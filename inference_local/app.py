
from __future__ import annotations
import csv, json, os, shutil, subprocess, sys, threading, time, uuid, zipfile
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image, ImageDraw
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

HERE=Path(__file__).resolve().parent
WORK=Path(os.getenv("SPINEMUSCLE_WORKDIR", HERE/"work"))
WORK.mkdir(parents=True,exist_ok=True)
EXACT=HERE/"exact"
CHECKPOINT=HERE/"frozen"/"level_v3"/"best_model.pt"
FROZEN_HASH="2EB1767910D1934F6CEA4152E09CADE95847ECE5777714BF63B9C1A9AFC125CC"
JOBS={}
LOCK=threading.Lock()
HEAVY_JOB_LOCK=threading.Lock()

JOB_RETENTION_HOURS=float(os.getenv("SPINEMUSCLE_JOB_RETENTION_HOURS","24"))
PURGE_HEAVY_ON_TERMINAL=os.getenv("SPINEMUSCLE_PURGE_HEAVY_ON_TERMINAL","1").strip().lower() in {"1","true","yes","on"}
HEAVY_DIR_NAMES={"dicom","duplicates","previews","musclemap_v4","mapping_v5","pvmq_dicom_batch","pvmq_v5_2"}

JOB_STATE_FILE="job.json"

def _safe_rmtree(path: Path) -> None:
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass

def _purge_heavy_job_data(jid: str) -> None:
    """Remove uploaded imaging and bulky intermediates, retaining job/report/log metadata."""
    if not PURGE_HEAVY_ON_TERMINAL:
        return
    wd = WORK / jid
    for name in HEAVY_DIR_NAMES:
        _safe_rmtree(wd / name)

def _expire_old_jobs_once() -> None:
    """Delete whole job folders after the configured retention window."""
    cutoff = time.time() - max(JOB_RETENTION_HOURS, 1.0) * 3600.0
    try:
        for wd in WORK.iterdir():
            if not wd.is_dir():
                continue
            state_file = wd / JOB_STATE_FILE
            try:
                stamp = state_file.stat().st_mtime if state_file.exists() else wd.stat().st_mtime
            except Exception:
                continue
            if stamp < cutoff:
                jid = wd.name
                _safe_rmtree(wd)
                with LOCK:
                    JOBS.pop(jid, None)
    except Exception:
        pass

def _janitor_loop() -> None:
    while True:
        _expire_old_jobs_once()
        time.sleep(3600)

threading.Thread(target=_janitor_loop, daemon=True, name="spinemuscle-job-janitor").start()

def _job_state_path(jid: str) -> Path:
    return WORK / jid / JOB_STATE_FILE

def _persist_job_unlocked(jid: str) -> None:
    state = JOBS.get(jid)
    if not state:
        return
    wd = WORK / jid
    wd.mkdir(parents=True, exist_ok=True)
    target = _job_state_path(jid)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)

def _load_job(jid: str):
    """Reload a persisted job after a worker restart when its work directory still exists.

    Processing work cannot be resumed safely mid-command, so a job found in the
    processing state after a restart is converted to an explicit interrupted state.
    Level-review and terminal jobs remain recoverable.
    """
    p = _job_state_path(jid)
    if not p.exists():
        return None
    try:
        state = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if state.get("stage") == "processing":
        reason = (
            "Analysis was interrupted because the inference service restarted. "
            "Please restart this study."
        )
        state.update({
            "stage": "blocked",
            "progress": 100,
            "message": reason,
            "qc_decision": "INTERRUPTED",
            "result": {
                "pvmq": None,
                "pvmq_issued": False,
                "qc_reason": reason,
            },
        })
        try:
            p.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    with LOCK:
        JOBS[jid] = state
    return state

def get_job_state(jid: str):
    with LOCK:
        state = JOBS.get(jid)
    if state is not None:
        return state
    return _load_job(jid)

sys.path.insert(0,str(EXACT/"level_verify_v1_5"))
import generate_level_review as lv
from level_localizer_v3 import PVMQ_LEVELS

app=FastAPI(title="SpineMuscle AI v15.1 Local Research Backend",version="15.1")
app.add_middleware(CORSMiddleware,allow_origins=os.getenv("CORS_ORIGINS","*").split(","),allow_methods=["*"],allow_headers=["*"],allow_credentials=False)

class ConfirmBody(BaseModel):
    selections: dict[str,str]

def sha256(path):
    import hashlib
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest().upper()


# Locked reference fingerprint for the validated research case 0271.
# Frozen PVMQ v5.2 may run only when the uploaded DICOM bytes match this reference.
VALIDATED_0271_HASHES = {
    "IM000023.dcm": "9A9989EFE1DFDC09FFB5E8BE42EAA0419415685F6D3DF6CCD6A0502C5C109675",
    "IM000018.dcm": "33F690CCBFD6974EA421CEB8C7871DCF85CF50CEFCE793A4641AA9A42B4BB2CF",
    "IM000013.dcm": "9180298E08625B5112BB1F2DBABB5DCFCC3BB609B6334DC03BB930A0F9426405",
    "IM000008.dcm": "E72E08A02E3FFD030C2B541713A0AD7CD7203B84BC4B1C660963F119EC184638",
}

def validated_0271_fingerprint(dicom_dir: Path) -> bool:
    """Exact-byte gate for the only study with a validated bundled CSF working mask."""
    try:
        files = [p for p in dicom_dir.iterdir() if p.is_file()]
        if len(files) != 25:
            return False
        for name, expected in VALIDATED_0271_HASHES.items():
            p = dicom_dir / name
            if (not p.is_file()) or sha256(p) != expected:
                return False
        return True
    except Exception:
        return False

def update(jid,**kw):
    with LOCK:
        if jid not in JOBS:
            loaded = None
            p = _job_state_path(jid)
            if p.exists():
                try:
                    loaded = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    loaded = None
            if loaded is None:
                raise KeyError(f"Unknown job {jid}")
            JOBS[jid] = loaded
        JOBS[jid].update(kw)
        _persist_job_unlocked(jid)

def conda_exe():
    cands=[os.getenv("CONDA_EXE"),
           str(Path.home()/"miniconda3_working"/"Scripts"/"conda.exe"),
           str(Path.home()/"miniconda3_working"/"condabin"/"conda.bat")]
    for x in cands:
        if x and Path(x).exists(): return x
    x=shutil.which("conda")
    return x

def pipeline_python_prefix():
    """Choose the Python used for app-owned orchestration scripts.

    Local validated runs can use the SpineLevels Conda environment. In the Docker
    deployment the web service itself already runs inside the pinned backend
    environment, while Conda is retained only because the frozen MuscleMap runner
    requires a separate environment literally named ``MuscleMap``. Setting
    SPINEMUSCLE_HOST_PIPELINE=1 therefore keeps app-owned orchestration on the
    current validated backend interpreter without changing any frozen method file.
    """
    if os.getenv("SPINEMUSCLE_HOST_PIPELINE", "").strip().lower() in {"1","true","yes","on"}:
        return [sys.executable]
    conda = conda_exe()
    if conda:
        return [conda, "run", "-n", "SpineLevels", "python"]
    return [sys.executable]

def run(cmd, log):
    """Run an app-owned subprocess while streaming output directly to disk.

    This avoids buffering potentially large stdout/stderr strings in the API process.
    """
    log_path=Path(log)
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with open(log_path,"w",encoding="utf-8",errors="ignore") as fh:
        p=subprocess.run([str(x) for x in cmd],text=True,stdout=fh,stderr=subprocess.STDOUT)
    if p.returncode!=0:
        raise RuntimeError(f"Command failed: {' '.join(map(str,cmd))}. See {log}")
    return p

@app.get("/api/health")
def health():
    ck_ok=CHECKPOINT.exists()
    ck_hash=sha256(CHECKPOINT) if ck_ok else None
    return {"ok":ck_ok and ck_hash==FROZEN_HASH,"service":"SpineMuscle AI v15.1 local research inference",
            "checkpoint_present":ck_ok,"checkpoint_hash":ck_hash,"checkpoint_hash_expected":FROZEN_HASH,
            "exact_component_hashes":json.loads((HERE/"EXACT_COMPONENT_HASHES.json").read_text()),
            "pvmq_0271_reproducibility_assets":(EXACT/"pvmq_v5_2"/"segmentation"/"0271").exists(),
            "runtime_profile":"CPU_ONLY_SERIAL_HEAVY_JOBS","job_retention_hours":JOB_RETENTION_HOURS,
            "purge_heavy_on_terminal":PURGE_HEAVY_ON_TERMINAL,
            "note":"New scans complete level confirmation + MuscleMap + muscle metrics. Frozen PVMQ v5.2 is issued only when a compatible validated thecal-sac mask exists."}

@app.post("/api/jobs")
async def create_job(case_id:str=Form(""), files:List[UploadFile]=File(...)):
    if not CHECKPOINT.exists(): raise HTTPException(503,"Frozen v3 checkpoint is not staged. Run STAGE_AND_VERIFY_ASSETS.ps1.")
    if sha256(CHECKPOINT)!=FROZEN_HASH: raise HTTPException(503,"Frozen v3 checkpoint hash does not match the locked model.")
    if not files: raise HTTPException(400,"Select one lumbar MRI study.")
    jid=uuid.uuid4().hex[:12]; wd=WORK/jid; dcm=wd/"dicom"; dcm.mkdir(parents=True)
    for i,f in enumerate(files):
        data=await f.read()
        if len(data)>80*1024*1024: raise HTTPException(413,"One uploaded file exceeds 80 MB.")
        (dcm/Path(f.filename or f"image_{i}.dcm").name).write_bytes(data)
    with LOCK:
        JOBS[jid]={"job_id":jid,"stage":"processing","progress":5,"message":"Validating the uploaded DICOM study...","case_id":case_id.strip(),"dicom_count":len(files)}
        _persist_job_unlocked(jid)
    threading.Thread(target=make_proposals,args=(jid,),daemon=True).start()
    return JOBS[jid]

@app.get("/api/jobs/{jid}")
def job(jid:str):
    state = get_job_state(jid)
    if state is None:
        raise HTTPException(404,"Job not found. The analysis session may have expired or the service may have restarted without persistent storage.")
    return state

@app.get("/api/jobs/{jid}/preview/{preview_id}")
def preview(jid:str,preview_id:str):
    p=WORK/jid/"previews"/Path(preview_id).name
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p,media_type="image/jpeg")

def save_preview(dicom_path:Path,label:str,out:Path):
    a=lv.load_pixel(dicom_path)
    im=Image.fromarray(a,"L").convert("RGB")
    im.thumbnail((420,420))
    im.save(out,"JPEG",quality=90)

def make_proposals(jid):
    wd=WORK/jid
    try:
        # Read headers and deduplicate by DICOM SOPInstanceUID.
        # Keep one copy deterministically; move extra copies out of the active DICOM folder.
        readable=[]; pids=set(); study_uids=set()
        for p in sorted((wd/"dicom").iterdir(), key=lambda x:(len(x.name),x.name.lower())):
            try:
                ds=lv.pydicom.dcmread(str(p),stop_before_pixels=True,force=True)
                sop=str(getattr(ds,"SOPInstanceUID","") or "").strip()
                study_uid=str(getattr(ds,"StudyInstanceUID","") or "").strip()
                if not sop:
                    continue
                r=lv.read_dicom_header(p)
                readable.append((p,sop,study_uid,r))
            except Exception:
                pass

        seen_sops=set()
        rows=[]
        duplicates=[]
        duplicate_dir=wd/"duplicates"

        for p,sop,study_uid,r in readable:
            if sop in seen_sops:
                duplicate_dir.mkdir(exist_ok=True)
                shutil.move(str(p),str(duplicate_dir/p.name))
                duplicates.append(p.name)
                continue

            seen_sops.add(sop)
            rows.append(r)
            if study_uid:
                study_uids.add(study_uid)
            if r.get("PatientID","").strip():
                pids.add(r["PatientID"].strip())

        update(
            jid,
            dicom_count=len(rows),
            dicom_count_unique=len(rows),
            duplicate_dicom_count=len(duplicates),
            duplicate_dicom_files=duplicates,
        )

        if not (5<=len(rows)<=500):
            raise RuntimeError(f"Expected 5-500 unique readable DICOMs; found {len(rows)}.")
        if len(study_uids)>1:
            raise RuntimeError("Upload contains DICOMs from more than one StudyInstanceUID.")
        if len(pids)>1:
            raise RuntimeError("Upload contains more than one distinct PatientID.")

        if duplicates:
            update(
                jid,
                progress=12,
                message=f"Removed {len(duplicates)} duplicate DICOM(s) by SOPInstanceUID; running frozen v3 level proposals...",
            )
        else:
            update(jid,progress=12,message="Running frozen v3 level proposals...")
        scores,ck=lv.score_all(rows,CHECKPOINT)
        prev=wd/"previews"; prev.mkdir()
        proposals={}
        for li,lev in enumerate(PVMQ_LEVELS):
            top=[int(x) for x in np.argsort(scores[:,li])[::-1][:5]]
            arr=[]
            for rank,idx in enumerate(top,1):
                r=rows[idx]; pid=f"{lev.replace('-','_')}_{rank}.jpg"
                save_preview(Path(r["Path"]),f"{lev} candidate {rank} | score {scores[idx,li]:.3f}",prev/pid)
                arr.append({"dicom_file":r["DICOM_File"],"score":float(scores[idx,li]),"preview_id":pid,
                            "instance_number":r["InstanceNumber"],"series_description":r["SeriesDescription"]})
            proposals[lev]=arr
        (wd/"level_proposals.json").write_text(json.dumps(proposals,indent=2))

        # Research-QA anatomical coverage gate.
        # Provisional threshold: validate on the intended dataset before treating as final.
        MIN_LEVEL_CONFIDENCE = 0.50
        failed_levels = [
            lev for lev in PVMQ_LEVELS
            if not proposals.get(lev)
            or float(proposals[lev][0]["score"]) < MIN_LEVEL_CONFIDENCE
        ]

        if failed_levels:
            reason = (
                "Unable to confidently identify all four required lumbar measurement planes. "
                "Upload the complete axial lumbar series. "
                f"Low-confidence levels: {', '.join(failed_levels)}."
            )
            update(
                jid,
                stage="blocked",
                progress=100,
                message=reason,
                qc_decision="INSUFFICIENT_ANATOMICAL_COVERAGE",
                level_proposals=proposals,
                result={
                    "qc_reason": reason,
                    "failed_levels": failed_levels,
                    "pvmq": None,
                    "pvmq_issued": False,
                },
            )
            return

        update(
            jid,
            stage="level_review",
            progress=25,
            message="Confirm the four PVMQ measurement planes.",
            level_proposals=proposals,
        )
    except Exception as e:
        update(jid,stage="blocked",progress=100,qc_decision="BLOCKED",result={"qc_reason":str(e),"pvmq":None})

@app.post("/api/jobs/{jid}/confirm-levels")
def confirm(jid:str,body:ConfirmBody):
    state = get_job_state(jid)
    if state is None:
        raise HTTPException(404,"Job not found. The analysis session may have expired.")
    if state.get("stage") != "level_review":
        raise HTTPException(409,"Level confirmation is not permitted for this job state.")
    required={"L1-L2","L2-L3","L3-L4","L4-L5"}
    if set(body.selections)!=required: raise HTTPException(400,"Exactly four levels must be confirmed.")
    wd=WORK/jid; prop=state.get("level_proposals",{})
    rows=[]
    for lev in ["L1-L2","L2-L3","L3-L4","L4-L5"]:
        f=body.selections[lev]
        if f not in {x["dicom_file"] for x in prop.get(lev,[])}: raise HTTPException(400,f"Invalid candidate for {lev}.")
        p=wd/"dicom"/f; rows.append({"Level":lev,"DICOM_File":p.name,"Path":str(p)})
    with open(wd/"CONFIRMED_LEVELS.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=["Level","DICOM_File","Path"]);w.writeheader();w.writerows(rows)
    update(jid,stage="processing",progress=30,message="Running whole-study MuscleMap segmentation...")
    threading.Thread(target=run_downstream,args=(jid,),daemon=True).start()
    return {"ok":True}

def run_downstream(jid):
    wd=WORK/jid; logs=wd/"logs"; logs.mkdir(exist_ok=True)
    update(jid,progress=32,message="Queued for memory-safe analysis...")
    with HEAVY_JOB_LOCK:
        try:
            py = pipeline_python_prefix(); muscle_dir=wd/"musclemap_v4"
            update(jid,progress=38,message="Converting the whole study and running MuscleMap v1.4...")
            run(py + [EXACT/"musclemap_v4"/"run_musclemap_v4.py",
                 "--confirmed-csv",wd/"CONFIRMED_LEVELS.csv","--output-dir",muscle_dir,"--model-version","1.4"],logs/"musclemap_v4.log")
    
            update(jid,progress=72,message="Mapping confirmed planes to MuscleMap segmentations...")
            map_dir=wd/"mapping_v5"; map_dir.mkdir()
            run(py + [EXACT/"mapping_v5"/"map_confirmed_planes_v5.py",
                 "--confirmed-csv",wd/"CONFIRMED_LEVELS.csv","--v4-result-dir",muscle_dir,"--output-dir",map_dir],logs/"mapping_v5.log")
            map_status=json.loads((map_dir/"RUN_STATUS.json").read_text())
            with open(map_dir/"CONFIRMED_PLANE_MAPPING_AND_MUSCLE_METRICS.csv",newline="",encoding="utf-8-sig") as f:
                mrows=list(csv.DictReader(f))
            levels=[{"level":r["Level"],"mean_si":float(r["Combined_Muscle_MeanSI"]),
                     "csa_mm2":float(r["Combined_Muscle_CSA_mm2"]),"geometry_qc":r["Geometry_QC"]} for r in mrows]
            numerator=map_status.get("four_level_mean_muscle_SI_numerator")
            result={"muscle_numerator":numerator,"csf_mean_si":None,"pvmq":None,"pvmq_issued":False,"csf_tier":None,
                    "qc_decision":"PVMQ_NOT_ISSUED",
                    "qc_reason":"Frozen PVMQ v5.2 not issued: the uploaded study does not match a validated CSF-mask reference fingerprint.",
                    "levels":levels,"method_versions":{"level_localizer":"v3 frozen","musclemap":"v1.4","plane_mapping":"v5 validated","pvmq":"v5.2 frozen"}}
    
            if validated_0271_fingerprint(wd/"dicom"):
                update(jid,progress=88,message="Running exact frozen PVMQ AUTO v5.2 for research case 0271...")
                pdir=wd/"pvmq_v5_2"; pdir.mkdir()
                archive_root=wd/"pvmq_dicom_batch"; archive_root.mkdir()
                with zipfile.ZipFile(archive_root/"251-300.zip","w",zipfile.ZIP_DEFLATED) as z:
                    for p in (wd/"dicom").iterdir():
                        if p.is_file(): z.write(p,Path("0271")/p.name)
                numcsv=wd/"numerator_0271.csv"
                with open(numcsv,"w",newline="",encoding="utf-8-sig") as f:
                    w=csv.DictWriter(f,fieldnames=["PatientID","Four_Level_Muscle_Mean_SI"]);w.writeheader()
                    w.writerow({"PatientID":"0271","Four_Level_Muscle_Mean_SI":repr(float(numerator))})
                run(py + [EXACT/"pvmq_v5_2"/"run_pvmq_v5_2.py",
                     "--dataset-dir",archive_root,"--segmentation-dir",EXACT/"pvmq_v5_2"/"segmentation",
                     "--manifest",EXACT/"pvmq_v5_2"/"manifest_0271.csv","--levels",EXACT/"pvmq_v5_2"/"levels_0271.csv",
                     "--numerator",numcsv,"--out-dir",pdir],logs/"pvmq_v5_2.log")
                with open(pdir/"pvmq_v5_2_results.csv",newline="",encoding="utf-8-sig") as f:
                    prow=list(csv.DictReader(f))[0]
                accepted=prow.get("Automatic_Decision")=="AUTO_ACCEPT" and prow.get("Status")=="OK"
                result.update({"csf_mean_si":float(prow["Final_Hierarchical_CSF_Mean_SI"]) if prow.get("Final_Hierarchical_CSF_Mean_SI") else None,
                               "pvmq":float(prow["Final_PVMQ"]) if accepted and prow.get("Final_PVMQ") else None,
                               "pvmq_issued":accepted,"csf_tier":("Tier "+prow["QC_Tier"]) if prow.get("QC_Tier") else None,
                               "qc_decision":prow.get("Automatic_Decision") or "AUTO_EXCLUDE",
                               "qc_reason":prow.get("Automatic_Exclusion_Reasons") or prow.get("QC_Warnings") or "Frozen v5.2 hierarchy passed."})
    
            (wd/"research_report.json").write_text(json.dumps(result,indent=2))
            update(jid,stage="complete" if result["pvmq_issued"] else "blocked",progress=100,message="Research workflow complete.",
                   qc_decision=result["qc_decision"],result=result,report_url=f"/api/jobs/{jid}/report")
        except Exception as e:
            update(jid,stage="blocked",progress=100,qc_decision="BLOCKED",
                   result={"pvmq":None,"pvmq_issued":False,"qc_reason":str(e)})
        finally:
            # Uploaded DICOMs and bulky NIfTI/segmentation intermediates are not retained
            # after a terminal result. Small report/job/log metadata remain temporarily.
            _purge_heavy_job_data(jid)

@app.get("/api/jobs/{jid}/report")
def report(jid:str):
    p=WORK/jid/"research_report.json"
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p,media_type="application/json",filename=f"SpineMuscle_{jid}_research_report.json")










