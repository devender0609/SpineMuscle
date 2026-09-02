from __future__ import annotations
import io, json, os, shutil, subprocess, threading, time, uuid, zipfile
from pathlib import Path
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageDraw
import pydicom, numpy as np

ROOT=Path(os.getenv("SPINEMUSCLE_WORKDIR","/tmp/spinemuscle"))
ROOT.mkdir(parents=True,exist_ok=True)
FROZEN=Path(os.getenv("SPINEMUSCLE_FROZEN_DIR","/opt/spinemuscle/frozen"))
LEVEL_REVIEW=Path(os.getenv("LEVEL_REVIEW_SCRIPT",FROZEN/"generate_level_review.py"))
JOBS={}
LOCK=threading.Lock()

app=FastAPI(title="SpineMuscle AI Research Inference",version="15.0")
app.add_middleware(CORSMiddleware,allow_origins=os.getenv("CORS_ORIGINS","*").split(","),allow_credentials=False,allow_methods=["*"],allow_headers=["*"])

class ConfirmBody(BaseModel):
    selections: dict[str,str]

def update(j,**kw):
    with LOCK: JOBS[j].update(kw)

@app.get("/api/health")
def health():
    return {"ok":True,"service":"SpineMuscle AI inference","version":"15.0",
            "frozen_dir_exists":FROZEN.exists(),
            "level_checkpoint_exists":(FROZEN/"level_v3"/"best_model.pt").exists(),
            "pvmq_v52_exists":(FROZEN/"pvmq_v5_2"/"run_pvmq_v5_2.py").exists()}

@app.post("/api/jobs")
async def create_job(files: List[UploadFile]=File(...)):
    if not files: raise HTTPException(400,"Upload one DICOM study.")
    jid=uuid.uuid4().hex[:12]; wd=ROOT/jid; dcm=wd/"dicom"; dcm.mkdir(parents=True)
    count=0
    for f in files:
        data=await f.read()
        if len(data)>80*1024*1024: raise HTTPException(413,"Individual file too large.")
        p=dcm/Path(f.filename or f"image_{count}.dcm").name
        p.write_bytes(data); count+=1
    JOBS[jid]={"job_id":jid,"stage":"processing","progress":5,"message":"Validating DICOM study…","dicom_count":count}
    threading.Thread(target=propose_levels,args=(jid,),daemon=True).start()
    return JOBS[jid]

@app.get("/api/jobs/{jid}")
def status(jid:str):
    if jid not in JOBS: raise HTTPException(404,"Job not found")
    return JOBS[jid]

def propose_levels(jid):
    wd=ROOT/jid
    try:
        # Guard: readable DICOMs and one nonblank PatientID if present.
        readable=[]; pids=set()
        for p in (wd/"dicom").iterdir():
            try:
                ds=pydicom.dcmread(str(p),stop_before_pixels=True,force=True); readable.append(p)
                pid=str(getattr(ds,"PatientID","")).strip()
                if pid:pids.add(pid)
            except Exception: pass
        if not (5<=len(readable)<=500): raise RuntimeError(f"Expected 5–500 readable DICOMs; found {len(readable)}.")
        if len(pids)>1: raise RuntimeError("Upload contains more than one PatientID.")
        update(jid,progress=12,message="Running frozen v3 level proposals…")
        # Exact frozen localizer integration contract.
        loc=FROZEN/"level_v3"/"generate_level_review_api.py"
        ck=FROZEN/"level_v3"/"best_model.pt"
        if not loc.exists() or not ck.exists():
            raise RuntimeError("Frozen v3 API localizer assets are not installed on the inference service.")
        out=wd/"level_review"; out.mkdir()
        cp=subprocess.run(["python",str(loc),"--study-dir",str(wd/"dicom"),"--checkpoint",str(ck),"--output-dir",str(out),"--json-api"],
                          capture_output=True,text=True)
        (wd/"level_localizer.log").write_text((cp.stdout or "")+"\n"+(cp.stderr or ""))
        if cp.returncode!=0: raise RuntimeError("Frozen v3 level localizer failed.")
        proposals=json.loads((out/"level_proposals.json").read_text())
        update(jid,stage="level_review",progress=25,message="Confirm four PVMQ planes.",level_proposals=proposals)
    except Exception as e:
        update(jid,stage="blocked",progress=100,qc_decision="BLOCKED",result={"qc_reason":str(e)})

@app.get("/api/jobs/{jid}/preview/{preview_id}")
def preview(jid:str,preview_id:str):
    if jid not in JOBS: raise HTTPException(404)
    p=ROOT/jid/"level_review"/"previews"/Path(preview_id).name
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p,media_type="image/jpeg")

@app.post("/api/jobs/{jid}/confirm-levels")
def confirm(jid:str,body:ConfirmBody):
    if jid not in JOBS: raise HTTPException(404)
    if set(body.selections)!={"L1-L2","L2-L3","L3-L4","L4-L5"}:
        raise HTTPException(400,"Exactly four confirmed PVMQ levels are required.")
    wd=ROOT/jid
    proposal_files={c["dicom_file"] for cs in JOBS[jid].get("level_proposals",{}).values() for c in cs}
    for lev,f in body.selections.items():
        if f not in proposal_files: raise HTTPException(400,f"Invalid candidate for {lev}.")
    (wd/"confirmed_levels.json").write_text(json.dumps(body.selections,indent=2))
    update(jid,stage="processing",progress=30,message="Running MuscleMap and confirmed-plane metrics…")
    threading.Thread(target=downstream,args=(jid,),daemon=True).start()
    return {"ok":True}

def downstream(jid):
    wd=ROOT/jid
    try:
        runner=FROZEN/"pipeline"/"run_confirmed_pipeline.py"
        if not runner.exists(): raise RuntimeError("Integrated downstream runner is not installed.")
        out=wd/"analysis"; out.mkdir()
        cp=subprocess.run(["python",str(runner),"--study-dir",str(wd/"dicom"),"--confirmed-json",str(wd/"confirmed_levels.json"),
                           "--output-dir",str(out)],capture_output=True,text=True)
        (wd/"downstream.log").write_text((cp.stdout or "")+"\n"+(cp.stderr or ""))
        if cp.returncode!=0: raise RuntimeError("Downstream research pipeline failed. Review server audit log.")
        result=json.loads((out/"result.json").read_text())
        # Explicit research-governance gate.
        pvmq_ok=bool(result.get("pvmq_issued"))
        stage="complete" if pvmq_ok else "blocked"
        qc=result.get("qc_decision","AUTO_ACCEPT" if pvmq_ok else "PVMQ_NOT_ISSUED")
        report=out/"research_report.json"; report.write_text(json.dumps(result,indent=2))
        update(jid,stage=stage,progress=100,message="Analysis complete.",qc_decision=qc,result=result,report_url=f"/api/jobs/{jid}/report")
    except Exception as e:
        update(jid,stage="blocked",progress=100,qc_decision="BLOCKED",result={"qc_reason":str(e),"pvmq":None})

@app.get("/api/jobs/{jid}/report")
def report(jid:str):
    p=ROOT/jid/"analysis"/"research_report.json"
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p,filename=f"SpineMuscle_{jid}_research_report.json",media_type="application/json")
