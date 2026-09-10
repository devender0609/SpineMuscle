
from __future__ import annotations
import argparse, csv, gzip, json, math, os, struct, sys
from pathlib import Path
import numpy as np
import pydicom

LEVELS=["L1-L2","L2-L3","L3-L4","L4-L5"]
MUSCLE_LABELS={5101:"L_multifidus",5102:"R_multifidus",5111:"L_erector_spinae",5112:"R_erector_spinae"}

def read_nii(path):
    opener=gzip.open if str(path).lower().endswith(".gz") else open
    with opener(path,"rb") as f:
        hdr=f.read(348)
        endian="<" if struct.unpack("<i",hdr[:4])[0]==348 else ">"
        dim=struct.unpack(endian+"8h",hdr[40:56])
        shape=tuple(dim[1:dim[0]+1])
        datatype=struct.unpack(endian+"h",hdr[70:72])[0]
        pixdim=struct.unpack(endian+"8f",hdr[76:108])
        vox_offset=struct.unpack(endian+"f",hdr[108:112])[0]
        scl_slope=struct.unpack(endian+"f",hdr[112:116])[0]
        scl_inter=struct.unpack(endian+"f",hdr[116:120])[0]
        sform_code=struct.unpack(endian+"h",hdr[254:256])[0]
        if sform_code<=0:
            raise RuntimeError(f"NIfTI has no usable sform: {path}")
        sx=struct.unpack(endian+"4f",hdr[280:296])
        sy=struct.unpack(endian+"4f",hdr[296:312])
        sz=struct.unpack(endian+"4f",hdr[312:328])
        aff=np.eye(4,dtype=float); aff[0]=sx; aff[1]=sy; aff[2]=sz
        dtmap={2:"u1",4:"i2",8:"i4",16:"f4",64:"f8",256:"i1",512:"u2",768:"u4"}
        if datatype not in dtmap: raise RuntimeError(f"Unsupported NIfTI datatype {datatype}: {path}")
        dt=np.dtype(endian+dtmap[datatype])
        f.seek(int(vox_offset))
        raw=f.read()
        arr=np.frombuffer(raw,dtype=dt,count=int(np.prod(shape))).reshape(shape,order="F")
        if np.issubdtype(arr.dtype,np.number) and scl_slope not in (0.0,1.0):
            arr=arr.astype(np.float32)*scl_slope+scl_inter
        return arr,aff,tuple(float(x) for x in pixdim[1:4])

def confirmed_rows(path):
    with open(path,newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    by={r.get("Level","").strip():r for r in rows}
    miss=[x for x in LEVELS if x not in by]
    if miss: raise SystemExit("Missing confirmed levels: "+", ".join(miss))
    return [by[x] for x in LEVELS]

def ras_position(ds):
    ipp=np.asarray(ds.ImagePositionPatient,dtype=float)
    return np.array([-ipp[0],-ipp[1],ipp[2]],dtype=float)

def plane_match(pos, affine, nz):
    step=affine[:3,2].astype(float)
    step_norm=float(np.linalg.norm(step))
    if step_norm<=0: raise RuntimeError("Invalid NIfTI slice step.")
    unit=step/step_norm
    best=None
    for k in range(nz):
        origin=affine[:3,3]+affine[:3,2]*k
        d=abs(float(np.dot(pos-origin,unit)))
        if best is None or d<best[0]:
            best=(d,k)
    return best

def find_seg_for_image(image_path, musclemap_root):
    stem=Path(image_path).name
    if stem.endswith(".nii.gz"): stem=stem[:-7]
    elif stem.endswith(".nii"): stem=stem[:-4]
    cands=list(Path(musclemap_root).rglob(stem+"_dseg.nii.gz"))+list(Path(musclemap_root).rglob(stem+"_dseg.nii"))
    if len(cands)!=1:
        raise RuntimeError(f"Expected one segmentation for {image_path}, found {len(cands)}")
    return cands[0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--confirmed-csv",required=True)
    ap.add_argument("--v4-result-dir",required=True)
    ap.add_argument("--output-dir",required=True)
    args=ap.parse_args()

    rows=confirmed_rows(args.confirmed_csv)
    root=Path(args.v4_result_dir)
    nifti_dir=root/"nifti"
    musclemap_root=root/"musclemap"
    if not nifti_dir.exists() or not musclemap_root.exists():
        raise SystemExit("Could not find v4 nifti/musclemap folders. Use the extracted v4 result folder.")

    imgs=sorted(list(nifti_dir.glob("*.nii"))+list(nifti_dir.glob("*.nii.gz")))
    if not imgs: raise SystemExit("No v4 NIfTI inputs found.")

    volumes=[]
    for p in imgs:
        a,aff,pix=read_nii(p)
        segp=find_seg_for_image(p,musclemap_root)
        s,saff,spix=read_nii(segp)
        if a.shape!=s.shape:
            raise RuntimeError(f"Image/seg shape mismatch: {p.name}")
        if np.max(np.abs(aff-saff))>1e-3:
            raise RuntimeError(f"Image/seg affine mismatch: {p.name}")
        volumes.append((p,a,aff,pix,segp,s))

    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    mapped=[]
    level_means=[]

    for r in rows:
        dcm=Path(r["Path"])
        if not dcm.exists(): raise SystemExit(f"Confirmed DICOM missing: {dcm}")
        ds=pydicom.dcmread(str(dcm),stop_before_pixels=True,force=True)
        if not hasattr(ds,"ImagePositionPatient"):
            raise RuntimeError(f"No ImagePositionPatient in {dcm}")
        pos=ras_position(ds)

        candidates=[]
        for vi,(p,a,aff,pix,segp,s) in enumerate(volumes):
            nz=a.shape[2] if a.ndim>=3 else 1
            dist,k=plane_match(pos,aff,nz)
            candidates.append((dist,k,vi))
        candidates.sort()
        dist,k,vi=candidates[0]
        p,a,aff,pix,segp,s=volumes[vi]

        # Conservative geometry gate.
        if dist>2.20:
            qc="FAIL"
        elif dist>0.75:
            qc="WARN"
        else:
            qc="PASS"

        im2=a[:,:,k].astype(np.float32)
        sg2=s[:,:,k].astype(np.int32)
        px_area=float(abs(pix[0]*pix[1]))
        muscle_mask=np.isin(sg2,list(MUSCLE_LABELS.keys()))
        n=int(muscle_mask.sum())
        if n==0:
            qc="FAIL"
            mean_si=float("nan")
            area=float("nan")
        else:
            vals=im2[muscle_mask]
            vals=vals[np.isfinite(vals)]
            mean_si=float(vals.mean()) if vals.size else float("nan")
            area=float(n*px_area)

        row={
            "Level":r["Level"],
            "Confirmed_DICOM":str(dcm),
            "Mapped_NIfTI":p.name,
            "Mapped_Segmentation":segp.name,
            "Mapped_Z_Index_0based":k,
            "PlaneDistance_mm":dist,
            "Geometry_QC":qc,
            "Combined_Muscle_Pixels":n,
            "Combined_Muscle_CSA_mm2":area,
            "Combined_Muscle_MeanSI":mean_si,
        }
        for lab,name in MUSCLE_LABELS.items():
            m=(sg2==lab)
            row[name+"_pixels"]=int(m.sum())
            row[name+"_CSA_mm2"]=float(m.sum()*px_area)
            row[name+"_MeanSI"]=float(im2[m].mean()) if m.any() else float("nan")
        mapped.append(row)
        if qc!="FAIL" and math.isfinite(mean_si):
            level_means.append(mean_si)

    fields=list(mapped[0].keys())
    with open(out/"CONFIRMED_PLANE_MAPPING_AND_MUSCLE_METRICS.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(mapped)

    all_pass=all(x["Geometry_QC"]=="PASS" for x in mapped)
    numerator=float(np.mean(level_means)) if len(level_means)==4 else None
    status="MAPPING_ALL_PASS" if all_pass else "MAPPING_REQUIRES_REVIEW"
    (out/"RUN_STATUS.json").write_text(json.dumps({
        "status":status,
        "levels":LEVELS,
        "all_four_exact_geometry_pass":all_pass,
        "four_level_mean_muscle_SI_numerator":numerator,
        "numerator_definition":"Mean of combined bilateral multifidus + erector-spinae mean SI across the four confirmed levels.",
        "pvmq_v5_2_run":False,
        "important":"This is the PVMQ muscle numerator only. No CSF denominator, PVMQ ratio, clinical cutoff, or PJK probability is generated."
    },indent=2),encoding="utf-8")

    print("\nMapping complete.")
    for x in mapped:
        print(f'{x["Level"]}: {x["Mapped_NIfTI"]} z={x["Mapped_Z_Index_0based"]} distance={x["PlaneDistance_mm"]:.4f} mm QC={x["Geometry_QC"]} meanSI={x["Combined_Muscle_MeanSI"]:.3f}')
    print("Status:",status)
    print("Four-level muscle SI numerator:",numerator)
    print("PVMQ v5.2 was NOT run.")

if __name__=="__main__":
    main()
