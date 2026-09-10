
from __future__ import annotations
import argparse, csv, json, os, shutil, subprocess, sys
from pathlib import Path

REQ=["L1-L2","L2-L3","L3-L4","L4-L5"]

def call(cmd, log=None):
    print(">", " ".join(map(str,cmd)), flush=True)
    p=subprocess.run(cmd,text=True,capture_output=True)
    text=(p.stdout or "") + ("\n[STDERR]\n"+p.stderr if p.stderr else "")
    if log: Path(log).write_text(text,encoding="utf-8",errors="ignore")
    if p.stdout: print(p.stdout,end="")
    if p.stderr: print(p.stderr,end="",file=sys.stderr)
    return p,text

def find_conda():
    for x in [os.environ.get("CONDA_EXE"),
              r"C:\Users\dpsingh\miniconda3_working\Scripts\conda.exe",
              r"C:\Users\dpsingh\miniconda3_working\condabin\conda.bat"]:
        if x and Path(x).exists(): return str(x)
    x=shutil.which("conda")
    if x: return x
    raise SystemExit("Conda executable not found.")

def read_confirmed(path):
    with open(path,newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    by={r["Level"].strip():r for r in rows}
    miss=[x for x in REQ if x not in by]
    if miss: raise SystemExit("Missing confirmed levels: "+", ".join(miss))
    return [by[x] for x in REQ]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--confirmed-csv",required=True)
    ap.add_argument("--output-dir",required=True)
    ap.add_argument("--model-version",default="1.4")
    a=ap.parse_args()

    rows=read_confirmed(a.confirmed_csv)
    parents={str(Path(r["Path"]).resolve().parent) for r in rows}
    if len(parents)!=1: raise SystemExit("Confirmed DICOMs span more than one source folder.")
    patient=Path(next(iter(parents)))

    out=Path(a.output_dir)
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(a.confirmed_csv,out/"CONFIRMED_LEVELS_INPUT.csv")
    (out/"SOURCE_PATIENT_FOLDER.txt").write_text(str(patient),encoding="utf-8")
    nifti=out/"nifti"; nifti.mkdir()
    segroot=out/"musclemap"; segroot.mkdir()
    logs=out/"logs"; logs.mkdir()

    conda=find_conda()

    # Environment proof
    p,envtxt=call([conda,"run","-n","MuscleMap","python","-c",
                   "import numpy,monai;print('NumPy='+numpy.__version__);print('MONAI='+monai.__version__)"],
                  logs/"environment.log")
    if p.returncode!=0: raise SystemExit("Could not inspect MuscleMap environment.")
    if "NumPy=2." in envtxt:
        raise SystemExit("STOP: MuscleMap still has NumPy 2.x. Run REPAIR_MUSCLEMAP_ENV.ps1 first.")

    # Whole-patient conversion
    p,txt=call([conda,"run","-n","MuscleMap","dcm2niix","-z","y","-f","%p_%s_i%5r",
                "-o",str(nifti),str(patient)],logs/"dcm2niix.log")
    if p.returncode!=0: raise SystemExit("dcm2niix failed.")
    niis=sorted(list(nifti.glob("*.nii"))+list(nifti.glob("*.nii.gz")))
    if not niis: raise SystemExit("No NIfTI created.")

    summary=[]
    for i,nii in enumerate(niis,1):
        sdir=segroot/f"series_{i:02d}"; sdir.mkdir()
        p,txt=call([conda,"run","-n","MuscleMap","mm_segment",
                    "-i",str(nii),"-c","auto","-s","75",
                    "--model_version",str(a.model_version),"-o",str(sdir)],
                   logs/f"mm_segment_series_{i:02d}.log")

        # Do NOT trust process exit code alone: mm_segment may log an internal error and still return 0.
        low=txt.lower()
        internal_error=(" - error - " in low or "traceback (most recent call last)" in low
                        or "error processing" in low)
        segs=sorted([x for x in sdir.rglob("*") if x.is_file() and (x.name.endswith(".nii") or x.name.endswith(".nii.gz"))])
        ok=(p.returncode==0 and not internal_error and len(segs)>0)
        status="OK" if ok else "FAIL"
        summary.append({"SeriesIndex":i,"InputNIfTI":str(nii),"Status":status,
                        "SegmentationFileCount":len(segs),
                        "SegmentationFiles":";".join(str(x) for x in segs),
                        "Log":str(logs/f"mm_segment_series_{i:02d}.log")})
        if not ok:
            with open(out/"MUSCLEMAP_SERIES_SUMMARY.csv","w",newline="",encoding="utf-8-sig") as f:
                w=csv.DictWriter(f,fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
            raise SystemExit(f"MuscleMap series {i} failed or produced no segmentation. See log.")

    with open(out/"MUSCLEMAP_SERIES_SUMMARY.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

    (out/"RUN_STATUS.json").write_text(json.dumps({
        "status":"MUSCLEMAP_ALL_SERIES_VERIFIED_OUTPUTS",
        "patient_folder":str(patient),
        "nifti_series_count":len(niis),
        "model_version":a.model_version,
        "confirmed_levels":REQ,
        "pvmq_v5_2_run":False,
        "next_step":"Verify segmentation geometry and map confirmed DICOM planes to segmentation slices."
    },indent=2),encoding="utf-8")
    print("\nSUCCESS: all MuscleMap series produced verified segmentation files.")

if __name__=="__main__":
    main()
