
from __future__ import annotations
import argparse, base64, io, json, html
from pathlib import Path
import numpy as np
import pydicom
import torch
from PIL import Image, ImageDraw
from level_localizer_v3 import LEVELS, PVMQ_LEVELS, make_model, safe_float

def read_dicom_header(path):
    ds=pydicom.dcmread(str(path),stop_before_pixels=True,force=True)
    if getattr(ds,"SOPInstanceUID",None) is None:
        raise ValueError("not dicom")
    ipp=getattr(ds,"ImagePositionPatient",["","",""])
    return {
        "Path":str(path),"DICOM_File":path.name,
        "PatientID":str(getattr(ds,"PatientID","") or ""),
        "SeriesUID":str(getattr(ds,"SeriesInstanceUID","") or ""),
        "SeriesDescription":str(getattr(ds,"SeriesDescription","") or ""),
        "InstanceNumber":str(getattr(ds,"InstanceNumber","") or ""),
        "IPP_Z":str(ipp[2]) if len(ipp)>=3 else "",
        "Manufacturer":str(getattr(ds,"Manufacturer","") or ""),
        "FieldStrength_T":str(getattr(ds,"MagneticFieldStrength","") or ""),
    }

def load_pixel(path):
    ds=pydicom.dcmread(str(path),force=True)
    a=ds.pixel_array.astype(np.float32)
    slope=safe_float(getattr(ds,"RescaleSlope",1)) or 1.0
    intercept=safe_float(getattr(ds,"RescaleIntercept",0)) or 0.0
    a=a*slope+intercept
    lo,hi=np.percentile(a,[1,99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi<=lo:
        lo=float(a.min()); hi=float(a.max())+1e-6
    a=np.clip((a-lo)/(hi-lo),0,1)
    return (a*255).astype(np.uint8)

def to_tensor(arr,size):
    im=Image.fromarray(arr,"L").convert("RGB").resize((size,size))
    x=np.asarray(im).astype(np.float32)/255.0
    x=(x-np.array([0.485,0.485,0.485],np.float32))/np.array([0.229,0.229,0.229],np.float32)
    return torch.tensor(np.transpose(x,(2,0,1)),dtype=torch.float32)

def image_data_uri(path,label):
    a=load_pixel(path)
    im=Image.fromarray(a,"L").convert("RGB")
    im.thumbnail((380,380))
    canvas=Image.new("RGB",(im.width,im.height+32),"white")
    canvas.paste(im,(0,32))
    d=ImageDraw.Draw(canvas); d.text((6,8),label,fill="black")
    bio=io.BytesIO(); canvas.save(bio,format="JPEG",quality=88)
    return "data:image/jpeg;base64,"+base64.b64encode(bio.getvalue()).decode()

def score_all(rows,checkpoint,batch=32):
    ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
    size=int(ck.get("image_size",224))
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=make_model(pretrained=False)
    model.load_state_dict(ck["state_dict"]); model.to(dev); model.eval()
    scores=np.zeros((len(rows),len(LEVELS)),dtype=np.float32)
    with torch.no_grad():
        for s in range(0,len(rows),batch):
            xs=[to_tensor(load_pixel(Path(rows[i]["Path"])),size) for i in range(s,min(len(rows),s+batch))]
            xx=torch.stack(xs).to(dev)
            scores[s:s+len(xs)]=torch.sigmoid(model(xx)).cpu().numpy()
    return scores,ck


def make_html(rows,scores,out_html):
    sections=[]
    for li,lev in enumerate(PVMQ_LEVELS):
        top=[int(x) for x in np.argsort(scores[:,li])[::-1][:5]]
        opts=[]
        for rank,idx in enumerate(top,1):
            r=rows[idx]
            checked="checked" if rank==1 else ""
            label=f"{lev} candidate {rank} | score {scores[idx,li]:.3f}"
            uri=image_data_uri(Path(r["Path"]),label)
            opts.append(
                '<label class="candidate">'
                + '<input type="radio" name="' + html.escape(lev) + '" '
                + 'value="' + html.escape(r["Path"]) + '" '
                + 'data-file="' + html.escape(r["DICOM_File"]) + '" ' + checked + '>'
                + '<img src="' + uri + '">'
                + '<div><b>#' + str(rank) + '</b> score ' + f'{scores[idx,li]:.4f}'
                + '<br>' + html.escape(r["DICOM_File"])
                + '<br>Inst ' + html.escape(r["InstanceNumber"])
                + ' | ' + html.escape(r["SeriesDescription"]) + '</div>'
                + '</label>'
            )
        sections.append(
            '<section><h2>' + html.escape(lev) + '</h2>'
            '<p>Select the intended mid-disc axial plane. The top v3 prediction is preselected.</p>'
            '<div class="grid">' + ''.join(opts) + '</div></section>'
        )

    page = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>SpineMuscle Level Verification</title>
<style>
body{font-family:Arial,sans-serif;max-width:1500px;margin:auto;padding:24px;background:#f5f7fa;color:#17202a}
header,section{background:white;padding:18px;border-radius:12px;box-shadow:0 1px 5px #ccc;margin-bottom:18px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.candidate{border:2px solid #ddd;border-radius:10px;padding:8px;cursor:pointer}
.candidate:has(input:checked){border-color:#111;background:#eef4ff}
.candidate img{width:100%;height:auto;border-radius:6px}
button{font-size:18px;padding:12px 20px;margin-right:8px}
.warn{background:#fff4d6;border-left:5px solid #d99a00;padding:12px}
.status{margin-top:12px;font-weight:bold}
</style>
</head>
<body>
<header>
<h1>SpineMuscle — Level Verification</h1>
<div class="warn"><b>Research QC step.</b> Frozen v3 proposes candidate slices. Confirm the four PVMQ measurement levels before downstream PVMQ calculation.</div>
<p>DICOMs analyzed: __DICOM_COUNT__. This page is local and self-contained.</p>
<button type="button" id="csvBtn">Confirm & Download CSV</button>
<button type="button" id="jsonBtn">Confirm & Download JSON</button>
<div class="status" id="status"></div>
</header>
__SECTIONS__
<script>
const levels = __LEVELS_JSON__;

function getSelections(){
  const out = {};
  for (const lev of levels){
    const x = document.querySelector('input[name="' + lev + '"]:checked');
    if (!x){
      throw new Error('No candidate selected for ' + lev);
    }
    out[lev] = {path:x.value, file:x.dataset.file};
  }
  return out;
}

function triggerDownload(filename, text, mime){
  const blob = new Blob([text], {type:mime});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(function(){ URL.revokeObjectURL(url); }, 1500);
}

function makeCSV(){
  const o = getSelections();
  const rows = ['Level,DICOM_File,Path'];
  const q = function(v){ return '"' + String(v).replaceAll('"','""') + '"'; };
  for (const lev of levels){
    rows.push(q(lev) + ',' + q(o[lev].file) + ',' + q(o[lev].path));
  }
  return rows.join('\\r\\n') + '\\r\\n';
}

document.getElementById('csvBtn').addEventListener('click', function(){
  try{
    triggerDownload('SpineMuscle_level_confirmation.csv', makeCSV(), 'text/csv;charset=utf-8');
    document.getElementById('status').textContent = 'CSV download started.';
  }catch(e){
    document.getElementById('status').textContent = 'Error: ' + e.message;
    alert(e.message);
  }
});

document.getElementById('jsonBtn').addEventListener('click', function(){
  try{
    const payload = {
      status:'HUMAN_CONFIRMED',
      method:'Frozen v3 proposal + human level confirmation',
      selections:getSelections()
    };
    triggerDownload('SpineMuscle_level_confirmation.json', JSON.stringify(payload,null,2), 'application/json;charset=utf-8');
    document.getElementById('status').textContent = 'JSON download started.';
  }catch(e){
    document.getElementById('status').textContent = 'Error: ' + e.message;
    alert(e.message);
  }
});
</script>
</body>
</html>
"""
    page = page.replace("__DICOM_COUNT__", str(len(rows)))
    page = page.replace("__SECTIONS__", "".join(sections))
    page = page.replace("__LEVELS_JSON__", json.dumps(PVMQ_LEVELS))
    Path(out_html).write_text(page, encoding="utf-8")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--study-dir",required=True)
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--output-dir",required=True)
    args=ap.parse_args()
    rows=[]
    for p in Path(args.study_dir).rglob("*"):
        if p.is_file():
            try: rows.append(read_dicom_header(p))
            except: pass
    if not rows:
        raise SystemExit("No readable DICOM images found.")

    # Safety guard for one-patient review.
    # This public dataset does not reliably populate DICOM PatientID, so PatientID
    # cannot be the sole guard. The selected folder itself must look like a single
    # patient folder and contain a plausible number of DICOM images.
    study_path=Path(args.study_dir)
    leaf=study_path.name.strip()
    patient_ids=sorted(set((r.get("PatientID") or "").strip() for r in rows if (r.get("PatientID") or "").strip()))

    # If PatientID exists, multiple distinct IDs are definitely unsafe.
    if len(patient_ids) > 1:
        raise SystemExit(
            f"STOP: found {len(patient_ids)} different PatientIDs across {len(rows)} DICOMs. "
            "Select one individual patient folder only."
        )

    # Dataset patient folders are four-digit IDs such as 0271.
    folder_looks_patient = bool(__import__("re").fullmatch(r"\d{4}", leaf))
    if not folder_looks_patient:
        raise SystemExit(
            f"STOP: selected folder '{leaf}' does not look like one patient folder. "
            "Choose a path such as ...\\DICOM\\0271."
        )

    # Plausibility bound to prevent accidentally analyzing the full dataset.
    if len(rows) < 5 or len(rows) > 500:
        raise SystemExit(
            f"STOP: found {len(rows)} readable DICOMs. Expected a single-patient study "
            "(roughly tens to a few hundred images), not a dataset root."
        )

    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    scores,ck=score_all(rows,args.checkpoint)
    import csv
    proposals=[]
    for li,lev in enumerate(PVMQ_LEVELS):
        idx=int(np.argmax(scores[:,li])); r=rows[idx]
        proposals.append({"Level":lev,"DICOM_File":r["DICOM_File"],"Path":r["Path"],
                          "SeriesUID":r["SeriesUID"],"InstanceNumber":r["InstanceNumber"],
                          "CenterScore":float(scores[idx,li])})
    with open(out/"v3_machine_proposals.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(proposals[0].keys())); w.writeheader(); w.writerows(proposals)
    make_html(rows,scores,out/"REVIEW_LEVELS.html")
    (out/"RUN_INFO.json").write_text(json.dumps({
        "model_type":ck.get("model_type"),"checkpoint_epoch":ck.get("epoch"),
        "sigma_slices":ck.get("sigma_slices"),"dicom_count":len(rows),"selected_folder":str(Path(args.study_dir)),"folder_patient_id":Path(args.study_dir).name,
        "workflow":"Frozen v3 proposal + human confirmation before PVMQ"
    },indent=2,default=str),encoding="utf-8")
    print("Review page:",out/"REVIEW_LEVELS.html")

if __name__=="__main__":
    main()
