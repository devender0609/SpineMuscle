
from __future__ import annotations
import csv, json, os, shutil, subprocess, sys, threading, uuid, zipfile
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

def update(jid,**kw):
    with LOCK: JOBS[jid].update(kw)

def conda_exe():
    cands=[os.getenv("CONDA_EXE"),
           str(Path.home()/"miniconda3_working"/"Scripts"/"conda.exe"),
           str(Path.home()/"miniconda3_working"/"condabin"/"conda.bat")]
    for x in cands:
        if x and Path(x).exists(): return x
    x=shutil.which("conda")
    if x:return x
    raise RuntimeError("Conda executable not found.")

def run(cmd, log):
    p=subprocess.run([str(x) for x in cmd],text=True,capture_output=True)
    Path(log).write_text((p.stdout or "")+"\n[STDERR]\n"+(p.stderr or ""),encoding="utf-8",errors="ignore")
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
    JOBS[jid]={"job_id":jid,"stage":"processing","progress":5,"message":"Validating the uploaded DICOM study…","case_id":case_id.strip(),"dicom_count":len(files)}
    threading.Thread(target=make_proposals,args=(jid,),daemon=True).start()
    return JOBS[jid]

@app.get("/api/jobs/{jid}")
def job(jid:str):
    if jid not in JOBS: raise HTTPException(404,"Job not found.")
    return JOBS[jid]

@app.get("/api/jobs/{jid}/preview/{preview_id}")
def preview(jid:str,preview_id:str):
    p=WORK/jid/"previews"/Path(preview_id).name
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p,media_type="image/jpeg")

def save_preview(dicom_path:Path,label:str,out:Path):
    a=lv.load_pixel(dicom_path)
    im=Image.fromarray(a,"L").convert("RGB"); im.thumbnail((420,420))
    c=Image.new("RGB",(im.width,im.height+34),"white"); c.paste(im,(0,34))
    ImageDraw.Draw(c).text((7,9),label,fill="black")
    c.save(out,"JPEG",quality=90)

def make_proposals(jid):
    wd=WORK[jid]
    try:
        rows=[]; pids=set()
        for p in (wd/"dicom").iterdir():
            try:
                r=lv.read_dicom_header(p); rows.append(r)
                if r.get("PatientID","").strip(): pids.add(r["PatientID"].strip())
            except Exception: pass
        if not (5<=len(rows)<=500): raise RuntimeError(f"Expected 5–500 readable DICOMs; found {len(rows)}.")
        if len(pids)>1: raise RuntimeError("Upload contains more than one distinct PatientID.")
        update(jid,progress=12,message="Running frozen v3 level proposals…")
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
        update(jid,stage="level_review",progress=25,message="Confirm the four PVMQ measurement planes.",level_proposals=proposals)
    except Exception as e:
        update(jid,stage="blocked",progress=100,qc_decision="BLOCKED",result={"qc_reason":str(e),"pvmq":None})

@app.post("/api/jobs/{jid}/confirm-levels")
def confirm(jid:str,body:ConfirmBody):
    if jid not in JOBS: raise HTTPException(404)
    required={"L1-L2","L2-L3","L3-L4","L4-L5"}
    if set(body.selections)!=required: raise HTTPException(400,"Exactly four levels must be confirmed.")
    wd=WORK/jid; prop=JOBS[jid].get("level_proposals",{})
    rows=[]
    for lev in ["L1-L2","L2-L3","L3-L4","L4-L5"]:
        f=body.selections[lev]
        if f not in {x["dicom_file"] for x in prop.get(lev,[])}: raise HTTPException(400,f"Invalid candidate for {lev}.")
        p=wd/"dicom"/f; rows.append({"Level":lev,"DICOM_File":p.name,"Path":str(p)})
    with open(wd/"CONFIRMED_LEVELS.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=["Level","DICOM_File","Path"]);w.writeheader();w.writerows(rows)
    update(jid,stage="processing",progress=30,message="Running whole-study MuscleMap segmentation…")
    threading.Thread(target=run_downstream,args=(jid,),daemon=True).start()
    return {"ok":True}

def run_downstream(jid):
    wd=WORK/jid; logs=wd/"logs"; logs.mkdir(exist_ok=True)
    try:
        conda=conda_exe(); muscle_dir=wd/"musclemap_v4"
        update(jid,progress=38,message="Converting the whole study and running MuscleMap v1.4…")
        run([conda,"run","-n","SpineLevels","python",EXACT/"musclemap_v4"/"run_musclemap_v4.py",
             "--confirmed-csv",wd/"CONFIRMED_LEVELS.csv","--output-dir",muscle_dir,"--model-version","1.4"],logs/"musclemap_v4.log")

        update(jid,progress=72,message="Mapping confirmed planes to MuscleMap segmentations…")
        map_dir=wd/"mapping_v5"; map_dir.mkdir()
        run([conda,"run","-n","SpineLevels","python",EXACT/"mapping_v5"/"map_confirmed_planes_v5.py",
             "--confirmed-csv",wd/"CONFIRMED_LEVELS.csv","--v4-result-dir",muscle_dir,"--output-dir",map_dir],logs/"mapping_v5.log")
        map_status=json.loads((map_dir/"RUN_STATUS.json").read_text())
        with open(map_dir/"CONFIRMED_PLANE_MAPPING_AND_MUSCLE_METRICS.csv",newline="",encoding="utf-8-sig") as f:
            mrows=list(csv.DictReader(f))
        levels=[{"level":r["Level"],"mean_si":float(r["Combined_Muscle_MeanSI"]),
                 "csa_mm2":float(r["Combined_Muscle_CSA_mm2"]),"geometry_qc":r["Geometry_QC"]} for r in mrows]
        numerator=map_status.get("four_level_mean_muscle_SI_numerator")
        result={"muscle_numerator":numerator,"csf_mean_si":None,"pvmq":None,"pvmq_issued":False,"csf_tier":None,
                "qc_decision":"PVMQ_NOT_ISSUED",
                "qc_reason":"Frozen v5.2 requires a compatible validated thecal-sac working mask; none is installed for this new study.",
                "levels":levels,"method_versions":{"level_localizer":"v3 frozen","musclemap":"v1.4","plane_mapping":"v5 validated","pvmq":"v5.2 frozen"}}

        if JOBS[jid].get("case_id","").zfill(4)=="0271":
            update(jid,progress=88,message="Running exact frozen PVMQ AUTO v5.2 for research case 0271…")
            pdir=wd/"pvmq_v5_2"; pdir.mkdir()
            archive_root=wd/"pvmq_dicom_batch"; archive_root.mkdir()
            with zipfile.ZipFile(archive_root/"251-300.zip","w",zipfile.ZIP_DEFLATED) as z:
                for p in (wd/"dicom").iterdir():
                    if p.is_file(): z.write(p,Path("0271")/p.name)
            numcsv=wd/"numerator_0271.csv"
            with open(numcsv,"w",newline="",encoding="utf-8-sig") as f:
                w=csv.DictWriter(f,fieldnames=["PatientID","Four_Level_Muscle_Mean_SI"]);w.writeheader()
                w.writerow({"PatientID":"0271","Four_Level_Muscle_Mean_SI":repr(float(numerator))})
            run([conda,"run","-n","SpineLevels","python",EXACT/"pvmq_v5_2"/"run_pvmq_v5_2.py",
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
        update(jid,stage="blocked",progress=100,qc_decision="BLOCKED",result={"pvmq":None,"pvmq_issued":False,"qc_reason":str(e)})

@app.get("/api/jobs/{jid}/report")
def report(jid:str):
    p=WORK/jid/"research_report.json"
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p,media_type="application/json",filename=f"SpineMuscle_{jid}_research_report.json")
