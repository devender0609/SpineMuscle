#!/usr/bin/env python
"""
PVMQ AUTO v5.2 — Freeze candidate
===============================================================

Purpose
-------
Preserve the quality advantage of v5 multi-slice CSF confirmation while reducing
the excessive attrition caused by requiring adjacent-slice confirmation in every
case.

Decision hierarchy
------------------
Tier A: target + BOTH neighboring segmentation slices confirm compatible
        high-signal CSF -> highest-confidence AUTO_ACCEPT.

Tier B: target + ONE neighboring slice confirms compatible high-signal CSF
        -> AUTO_ACCEPT if cross-slice QC is adequate.

Tier C: target-only rescue when neighboring confirmation is unavailable or fails.
        Rescue is deliberately stricter than Tier A/B:
        - strong high-signal GMM separation
        - high posterior membership
        - high fraction of ROI in high-signal compartment
        - good ROI homogeneity
        - adequate depth from TS boundary
        - CSF signal must exceed muscle signal
        -> AUTO_ACCEPT_TARGET_ONLY.

Tier D: automatic exclusion if none of the above is defensible.

No manual clicking is required.
No clinical PJK cutoff or probability is produced.
Locked-test patients remain excluded.
"""

import argparse, csv, io, math, re, tempfile, zipfile
from pathlib import Path
import numpy as np
import pydicom
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, label

LEVEL="L3-L4"

# Shared engineering/QC thresholds
MIN_TS_PIXELS=25
MIN_HIGH_COMPONENT_PX=20
MIN_POSTERIOR_PROB=0.70
MIN_TARGET_HIGH_FRAC=0.70
MIN_NEIGHBOR_HIGH_FRAC=0.50
MAX_NORM_POSITION_SHIFT=0.22
MAX_CROSS_SLICE_REL_RANGE=0.55
MAX_CROSS_SLICE_CV=0.35
ROI_DEPTH_EQR_MIN=0.10
MIN_GMM_SEPARATION_D=0.75

# Stricter target-only rescue thresholds
RESCUE_GMM_SEPARATION_D=1.25
RESCUE_POSTERIOR_HIGH=0.85
RESCUE_HIGH_FRAC=0.85
RESCUE_MAX_CV=0.25
RESCUE_DEPTH_EQR_MIN=0.15
RESCUE_REQUIRE_CSF_GT_MUSCLE=True

MAX_EM_ITER=200
EM_TOL=1e-6

def read_csv(p):
    with open(p,newline="",encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def write_csv(p,rows):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        p.write_text("",encoding="utf-8-sig"); return
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
    with open(p,"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def pid(v): return str(v).strip().zfill(4)

def index_seg(root,pids):
    root=Path(root); wanted=set(pids); idx={}
    for p in root.rglob("*_M.png"):
        m=re.search(r'(^|[\\/])(\d{4})([\\/])',str(p))
        if not m: continue
        pp=m.group(2)
        if pp not in wanted: continue
        stem=p.name[:-6]
        idx[(pp,stem)]={"mask":np.array(Image.open(p)),"source":str(p)}
    for zp in root.rglob("*.zip"):
        try:
            with zipfile.ZipFile(zp) as z:
                for n in z.namelist():
                    norm=n.replace("\\","/")
                    m=re.search(r'(^|/)(\d{4})/([^/]+)_M\.png$',norm,re.I)
                    if not m: continue
                    pp,stem=m.group(2),m.group(3)
                    if pp not in wanted or (pp,stem) in idx: continue
                    idx[(pp,stem)]={"mask":np.array(Image.open(io.BytesIO(z.read(n)))),"source":f"{zp}::{norm}"}
        except zipfile.BadZipFile:
            pass
    return idx

def archive_for(root,name):
    root=Path(root)
    p=root/name
    if p.exists(): return p
    alt=root/name.replace("(1)","")
    if alt.exists(): return alt
    raise FileNotFoundError(name)

def extract_patient(archive,pp,dest):
    dest.mkdir(parents=True,exist_ok=True); out={}
    with zipfile.ZipFile(archive) as z:
        for n in z.namelist():
            norm=n.replace("\\","/")
            m=re.search(r'(^|/)(\d{4})/([^/]+)\.dcm$',norm,re.I)
            if not m or m.group(2)!=pp: continue
            stem=m.group(3); fp=dest/(stem+".dcm")
            with z.open(n) as s, open(fp,"wb") as d: d.write(s.read())
            out[stem]=fp
    if not out: raise RuntimeError(f"No DICOMs found for {pp}")
    return out

def dgeom(path):
    ds=pydicom.dcmread(str(path),stop_before_pixels=True,force=True)
    ipp=np.array([float(x) for x in ds.ImagePositionPatient])
    iop=np.array([float(x) for x in ds.ImageOrientationPatient])
    n=np.cross(iop[:3],iop[3:]); n=n/np.linalg.norm(n)
    return float(np.dot(ipp,n))

def spacing(paths):
    vals=np.array(sorted({round(dgeom(p),6) for p in paths}),float)
    dif=np.abs(np.diff(vals)); dif=dif[dif>1e-4]
    return float(np.median(dif)) if len(dif) else np.nan

def choose_seg(pp,exact_stem,idx,dcm):
    if exact_stem not in dcm: raise RuntimeError(f"{exact_stem}.dcm missing")
    target=dgeom(dcm[exact_stem]); cand=[]
    for (p,stem),rec in idx.items():
        if p!=pp or stem not in dcm: continue
        cand.append((abs(dgeom(dcm[stem])-target),stem,rec))
    if not cand: raise RuntimeError("No linked segmentation mask")
    return sorted(cand,key=lambda x:x[0])[0]

def load_img(path):
    ds=pydicom.dcmread(str(path),force=True)
    arr=ds.pixel_array.astype(float)
    return arr*float(getattr(ds,"RescaleSlope",1) or 1)+float(getattr(ds,"RescaleIntercept",0) or 0)

def gaussian_pdf(x,mu,sigma):
    sigma=max(float(sigma),1e-6)
    z=(x-mu)/sigma
    return np.exp(-0.5*z*z)/(sigma*np.sqrt(2*np.pi))

def fit_gmm2(vals):
    x=np.asarray(vals,float); x=x[np.isfinite(x)]
    if len(x)<MIN_TS_PIXELS: raise RuntimeError("Too few TS pixels for GMM")
    lo,hi=np.percentile(x,[1,99]); xf=np.clip(x,lo,hi)
    q30,q70=np.percentile(xf,[30,70]); sd=max(float(np.std(xf,ddof=1)),1e-3)
    mu1,mu2=float(q30),float(q70); s1=s2=max(sd*0.65,1e-3); w1=w2=0.5
    prev=None
    for _ in range(MAX_EM_ITER):
        p1=w1*gaussian_pdf(xf,mu1,s1); p2=w2*gaussian_pdf(xf,mu2,s2)
        den=p1+p2+1e-300; r1=p1/den; r2=p2/den
        n1=max(r1.sum(),1e-8); n2=max(r2.sum(),1e-8)
        w1=float(n1/len(xf)); w2=float(n2/len(xf))
        mu1=float(np.sum(r1*xf)/n1); mu2=float(np.sum(r2*xf)/n2)
        s1=float(np.sqrt(max(np.sum(r1*(xf-mu1)**2)/n1,1e-6)))
        s2=float(np.sqrt(max(np.sum(r2*(xf-mu2)**2)/n2,1e-6)))
        ll=float(np.sum(np.log(den)))
        if prev is not None and abs(ll-prev)<EM_TOL*(1+abs(prev)): break
        prev=ll
    low,high=sorted([{"weight":w1,"mean":mu1,"sd":s1},{"weight":w2,"mean":mu2,"sd":s2}],key=lambda c:c["mean"])
    pooled=np.sqrt((low["sd"]**2+high["sd"]**2)/2)
    sep=(high["mean"]-low["mean"])/max(pooled,1e-8)
    return low,high,float(sep)

def high_prob(img,ts,low,high):
    p0=low["weight"]*gaussian_pdf(img,low["mean"],low["sd"])
    p1=high["weight"]*gaussian_pdf(img,high["mean"],high["sd"])
    prob=p1/(p0+p1+1e-300); prob[~ts]=0
    return prob

def clean_high(prob,ts):
    hm=(ts & np.isfinite(prob) & (prob>=MIN_POSTERIOR_PROB))
    lab,n=label(hm.astype(np.uint8)); clean=np.zeros_like(hm,dtype=bool)
    for k in range(1,n+1):
        if int(np.sum(lab==k))>=MIN_HIGH_COMPONENT_PX: clean|=(lab==k)
    return clean

def bbox(mask):
    ys,xs=np.where(mask)
    if not len(xs): return None
    return xs.min(),ys.min(),xs.max(),ys.max()

def circle_mask(shape,x,y,r):
    yy,xx=np.ogrid[:shape[0],:shape[1]]
    return (xx-x)**2+(yy-y)**2<=r*r

def load_seg_slice(pp,stem,segidx,dcm):
    rec=segidx.get((pp,stem))
    if rec is None or stem not in dcm: return None
    img=load_img(dcm[stem]); mask=rec["mask"]
    resampled=False
    if mask.ndim==3:
        if np.array_equal(mask[:,:,0],mask[:,:,1]) and np.array_equal(mask[:,:,0],mask[:,:,2]): mask=mask[:,:,0]
        else: return None
    if mask.shape!=img.shape:
        mar=mask.shape[1]/mask.shape[0]; iar=img.shape[1]/img.shape[0]
        if abs(mar-iar)<=0.01:
            mask=np.array(Image.fromarray(mask).resize((img.shape[1],img.shape[0]),resample=Image.Resampling.NEAREST))
            resampled=True
        else: return None
    ts=(mask==100)
    if int(ts.sum())<MIN_TS_PIXELS: return None
    low,high,sep=fit_gmm2(img[ts]); prob=high_prob(img,ts,low,high); hm=clean_high(prob,ts)
    if hm.sum()<MIN_HIGH_COMPONENT_PX: return None
    return {"img":img,"ts":ts,"sep":sep,"prob":prob,"highmask":hm,
            "low":low,"high":high,"resampled":resampled}

def target_candidates(img,ts,highmask,prob):
    bb=bbox(ts)
    if bb is None: return []
    x1,y1,x2,y2=bb; eqr=math.sqrt(float(ts.sum())/math.pi); dt=distance_transform_edt(ts)
    all_ranked=[]
    for radius in [3,2]:
        ranked=[]
        erosions=[2,1,0] if radius==3 else [1,0]
        for er in erosions:
            allowed=binary_erosion(ts,iterations=er) if er>0 else ts.copy()
            ys,xs=np.where(allowed)
            for y,x in zip(ys,xs):
                cm=circle_mask(img.shape,x,y,radius)
                if not np.all(allowed[cm]): continue
                vals=img[cm]; vals=vals[np.isfinite(vals)]
                if len(vals)<5: continue
                hf=float(np.mean(highmask[cm]))
                if hf<MIN_TARGET_HIGH_FRAC: continue
                mean=float(vals.mean()); sd=float(vals.std(ddof=1)) if len(vals)>1 else 0.0
                cv=sd/mean if mean else np.inf
                post=float(np.mean(prob[cm]))
                depth=float(dt[y,x])/max(eqr,1e-8)
                ranked.append({
                    "x":float(x),"y":float(y),"mean":mean,"sd":sd,"cv":cv,
                    "hi_frac":hf,"post":post,"depth_norm":depth,
                    "norm_x":float((x-x1)/max(x2-x1,1)),
                    "norm_y":float((y-y1)/max(y2-y1,1)),
                    "erosion":er,"radius":radius
                })
            if ranked: break
        if ranked:
            means=np.array([c["mean"] for c in ranked]); cvs=np.array([c["cv"] for c in ranked])
            posts=np.array([c["post"] for c in ranked]); hfs=np.array([c["hi_frac"] for c in ranked]); deps=np.array([c["depth_norm"] for c in ranked])
            def norm(v):
                lo,hi=np.percentile(v,[10,90]) if len(v)>2 else (v.min(),v.max())
                return np.clip((v-lo)/max(hi-lo,1e-8),0,1)
            nm=norm(means); nh=1-norm(cvs); npst=norm(posts); nf=norm(hfs); nd=norm(deps)
            for i,c in enumerate(ranked):
                c["score"]=float(.30*npst[i]+.25*nf[i]+.20*nm[i]+.15*nh[i]+.10*nd[i])
            all_ranked.extend(ranked)
            break
    return sorted(all_ranked,key=lambda c:c["score"],reverse=True)

def nearest_high_candidate(ns,nx,ny,radius):
    img,ts,highmask,prob=ns["img"],ns["ts"],ns["highmask"],ns["prob"]
    bb=bbox(ts)
    if bb is None: return None
    x1,y1,x2,y2=bb
    allowed=binary_erosion(ts,iterations=1)
    if allowed.sum()<8: allowed=ts.copy()
    ys,xs=np.where(allowed); best=None
    for y,x in zip(ys,xs):
        cm=circle_mask(img.shape,x,y,radius)
        if not np.all(allowed[cm]): continue
        vals=img[cm]; vals=vals[np.isfinite(vals)]
        if len(vals)<5: continue
        hf=float(np.mean(highmask[cm]))
        if hf<MIN_NEIGHBOR_HIGH_FRAC: continue
        nnx=(x-x1)/max(x2-x1,1); nny=(y-y1)/max(y2-y1,1)
        shift=math.hypot(nnx-nx,nny-ny)
        if shift>MAX_NORM_POSITION_SHIFT: continue
        mean=float(vals.mean()); sd=float(vals.std(ddof=1)) if len(vals)>1 else 0.0
        cv=sd/mean if mean else np.inf
        post=float(np.mean(prob[cm]))
        score=shift+0.2*cv-0.1*post
        if best is None or score<best["score"]:
            best={"mean":mean,"cv":cv,"hi_frac":hf,"post":post,
                  "shift":float(shift),"score":float(score)}
    return best

def neighboring_seg_stems(pp,target_stem,segidx,dcm):
    arr=[]
    for (p,stem) in segidx:
        if p!=pp or stem not in dcm: continue
        try: arr.append((dgeom(dcm[stem]),stem))
        except: pass
    arr=sorted(arr,key=lambda x:x[0]); stems=[s for _,s in arr]
    if target_stem not in stems: return None,None
    i=stems.index(target_stem)
    return (stems[i-1] if i>0 else None,
            stems[i+1] if i<len(stems)-1 else None)

def choose_hierarchical(target,candidates,neighbors,muscle_si):
    # First try multi-slice persistence for top candidates.
    best_multi=None
    for c in candidates[:25]:
        conf=[]
        for stem,ns in neighbors:
            nc=nearest_high_candidate(ns,c["norm_x"],c["norm_y"],c["radius"])
            if nc is not None: conf.append((stem,nc))
        if not conf:
            continue
        means=np.array([c["mean"]]+[x[1]["mean"] for x in conf],float)
        med=float(np.median(means))
        rel=float((means.max()-means.min())/max(abs(med),1e-8))
        cv=float(means.std(ddof=1)/max(abs(means.mean()),1e-8)) if len(means)>1 else 0.0
        if rel>MAX_CROSS_SLICE_REL_RANGE or cv>MAX_CROSS_SLICE_CV:
            continue
        shift=float(np.mean([x[1]["shift"] for x in conf]))
        score=c["score"]+.30*(len(conf)/2)-.25*shift-.20*rel-.15*cv
        tier="A" if len(conf)>=2 else "B"
        rec={"tier":tier,"c":c,"conf":conf,"rel":rel,"cv":cv,"shift":shift,"score":score}
        if best_multi is None or score>best_multi["score"]:
            best_multi=rec

    if best_multi is not None:
        c=best_multi["c"]
        vals=[c["mean"]]+[x[1]["mean"] for x in best_multi["conf"]]
        final_csf=float(np.median(vals))

        # v5.2 freeze-candidate rule:
        # A persistent presumed T2 CSF compartment must also be brighter than
        # the four-level paravertebral muscle reference. This is an engineering
        # tissue-identification sanity rule, not a clinical cutoff.
        if not (final_csf > muscle_si):
            return {
                "tier":"D",
                "candidate":c,
                "confirmations":best_multi["conf"],
                "cross_rel":best_multi["rel"],
                "cross_cv":best_multi["cv"],
                "cross_shift":best_multi["shift"],
                "final_csf":final_csf,
                "decision":"AUTO_EXCLUDE_LOW_CSF_SIGNAL"
            }

        return {
            "tier":best_multi["tier"],
            "candidate":c,
            "confirmations":best_multi["conf"],
            "cross_rel":best_multi["rel"],
            "cross_cv":best_multi["cv"],
            "cross_shift":best_multi["shift"],
            "final_csf":final_csf,
            "decision":"AUTO_ACCEPT"
        }

    # Tier C: target-only rescue, deliberately stricter than multi-slice tiers.
    for c in candidates[:25]:
        rescue_ok = (
            target["sep"] >= RESCUE_GMM_SEPARATION_D and
            c["post"] >= RESCUE_POSTERIOR_HIGH and
            c["hi_frac"] >= RESCUE_HIGH_FRAC and
            c["cv"] <= RESCUE_MAX_CV and
            c["depth_norm"] >= RESCUE_DEPTH_EQR_MIN
        )
        if RESCUE_REQUIRE_CSF_GT_MUSCLE:
            rescue_ok = rescue_ok and (c["mean"] > muscle_si)
        if rescue_ok:
            return {
                "tier":"C",
                "candidate":c,
                "confirmations":[],
                "cross_rel":np.nan,
                "cross_cv":np.nan,
                "cross_shift":np.nan,
                "final_csf":float(c["mean"]),
                "decision":"AUTO_ACCEPT_TARGET_ONLY"
            }

    return None

def process_patient(pp,mrow,ref,segidx,dataset_dir,muscle_si):
    with tempfile.TemporaryDirectory(prefix=f"pvmqv51_{pp}_") as td:
        td=Path(td)
        exact=Path(ref["DICOM_File"]).stem
        arch=archive_for(dataset_dir,mrow["Archive"])
        dcm=extract_patient(arch,pp,td/pp)
        sp=spacing(list(dcm.values()))
        dist,target_stem,_=choose_seg(pp,exact,segidx,dcm)

        target=load_seg_slice(pp,target_stem,segidx,dcm)
        if target is None: raise RuntimeError("Target segmentation slice unusable")
        if target["sep"]<MIN_GMM_SEPARATION_D:
            raise RuntimeError("Target GMM components poorly separated")

        candidates=target_candidates(target["img"],target["ts"],target["highmask"],target["prob"])
        if not candidates: raise RuntimeError("No valid target-slice high-signal candidate")

        prev_stem,next_stem=neighboring_seg_stems(pp,target_stem,segidx,dcm)
        neighbors=[]
        for stem in [prev_stem,next_stem]:
            if stem:
                ns=load_seg_slice(pp,stem,segidx,dcm)
                if ns is not None: neighbors.append((stem,ns))

        choice=choose_hierarchical(target,candidates,neighbors,muscle_si)
        if choice is None:
            raise RuntimeError("No defensible Tier A/B/C CSF measurement")

        c=choice["candidate"]
        final_csf=choice["final_csf"]

        if choice["decision"]=="AUTO_EXCLUDE_LOW_CSF_SIGNAL":
            pvmq=float(muscle_si/final_csf) if final_csf>0 else np.nan
            return {
                "QC_Tier":"D",
                "Target_Segmentation_DICOM":target_stem+".dcm",
                "Neighbor_Previous_DICOM":prev_stem+".dcm" if prev_stem else "",
                "Neighbor_Next_DICOM":next_stem+".dcm" if next_stem else "",
                "Available_Usable_Neighbors":len(neighbors),
                "Confirmed_Neighbor_Count":len(choice["confirmations"]),
                "Distance_to_Spacing_Ratio":dist/sp if np.isfinite(sp) and sp>0 else np.nan,
                "Target_ROI_X_px":c["x"],"Target_ROI_Y_px":c["y"],
                "Target_ROI_Radius_px":c["radius"],"Target_ROI_Erosion":c["erosion"],
                "Target_CSF_Mean_SI":c["mean"],"Target_CSF_CV":c["cv"],
                "Target_High_Component_Fraction":c["hi_frac"],
                "Target_Posterior_High":c["post"],
                "Target_ROI_Depth_Over_EqRadius":c["depth_norm"],
                "Target_GMM_Separation_D":target["sep"],
                "CrossSlice_MeanShift_Normalized":choice["cross_shift"],
                "CrossSlice_SI_RelRange":choice["cross_rel"],
                "CrossSlice_SI_CV":choice["cross_cv"],
                "Neighbor_CSF_Means":";".join(f"{s}:{n['mean']:.6f}" for s,n in choice["confirmations"]),
                "Final_Hierarchical_CSF_Mean_SI":final_csf,
                "CSF_to_Muscle_Ratio":final_csf/muscle_si if muscle_si>0 else np.nan,
                "Four_Level_Muscle_Mean_SI":muscle_si,
                "Final_PVMQ":pvmq,
                "Automatic_Decision":"AUTO_EXCLUDE",
                "Automatic_Exclusion_Reasons":"MULTISLICE_CSF_NOT_BRIGHTER_THAN_MUSCLE",
                "QC_Warnings":"CSF_NOT_BRIGHTER_THAN_MUSCLE"
            }
        pvmq=float(muscle_si/final_csf) if final_csf>0 else np.nan
        ratio=dist/sp if np.isfinite(sp) and sp>0 else np.nan
        csf_to_muscle=final_csf/muscle_si if muscle_si>0 else np.nan

        reasons=[]; warnings=[]
        if final_csf<=muscle_si:
            warnings.append("CSF_NOT_BRIGHTER_THAN_MUSCLE")
            # Tier C cannot get here by construction.
            if choice["tier"] in ("A","B"):
                if ((np.isfinite(choice["cross_rel"]) and choice["cross_rel"]>0.35) or
                    (np.isfinite(choice["cross_cv"]) and choice["cross_cv"]>0.22) or
                    (np.isfinite(choice["cross_shift"]) and choice["cross_shift"]>0.15)):
                    reasons.append("LOW_T2_SIGNAL_PLUS_WEAK_MULTISLICE_QC")

        if np.isfinite(pvmq) and pvmq>2.0:
            warnings.append("EXTREME_PVMQ_ENGINEERING_CHECK")
            reasons.append("EXTREME_PVMQ_ENGINEERING_EXCLUSION")

        decision=choice["decision"] if not reasons else "AUTO_EXCLUDE"

        return {
            "QC_Tier":choice["tier"],
            "Target_Segmentation_DICOM":target_stem+".dcm",
            "Neighbor_Previous_DICOM":prev_stem+".dcm" if prev_stem else "",
            "Neighbor_Next_DICOM":next_stem+".dcm" if next_stem else "",
            "Available_Usable_Neighbors":len(neighbors),
            "Confirmed_Neighbor_Count":len(choice["confirmations"]),
            "Distance_to_Spacing_Ratio":ratio,
            "Target_ROI_X_px":c["x"],"Target_ROI_Y_px":c["y"],
            "Target_ROI_Radius_px":c["radius"],"Target_ROI_Erosion":c["erosion"],
            "Target_CSF_Mean_SI":c["mean"],"Target_CSF_CV":c["cv"],
            "Target_High_Component_Fraction":c["hi_frac"],
            "Target_Posterior_High":c["post"],
            "Target_ROI_Depth_Over_EqRadius":c["depth_norm"],
            "Target_GMM_Separation_D":target["sep"],
            "CrossSlice_MeanShift_Normalized":choice["cross_shift"],
            "CrossSlice_SI_RelRange":choice["cross_rel"],
            "CrossSlice_SI_CV":choice["cross_cv"],
            "Neighbor_CSF_Means":";".join(f"{s}:{n['mean']:.6f}" for s,n in choice["confirmations"]),
            "Final_Hierarchical_CSF_Mean_SI":final_csf,
            "CSF_to_Muscle_Ratio":csf_to_muscle,
            "Four_Level_Muscle_Mean_SI":muscle_si,
            "Final_PVMQ":pvmq,
            "Automatic_Decision":decision,
            "Automatic_Exclusion_Reasons":";".join(reasons),
            "QC_Warnings":";".join(warnings)
        }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset-dir",required=True)
    ap.add_argument("--segmentation-dir",required=True)
    ap.add_argument("--manifest",default="manifest500.csv")
    ap.add_argument("--levels",default="levels.csv")
    ap.add_argument("--numerator",default="PVMQ_numerator_exact_levels.csv")
    ap.add_argument("--out-dir",default="PVMQ_AUTO_V5_2_output")
    args=ap.parse_args()

    manifest={pid(r["PatientID"]):r for r in read_csv(args.manifest)}
    refs={(pid(r["PatientID"]),r["Level"]):r for r in read_csv(args.levels)}
    nums=read_csv(args.numerator)

    targets=[]
    for r in nums:
        pp=pid(r["PatientID"])
        try: mus=float(r["Four_Level_Muscle_Mean_SI"])
        except: continue
        if manifest.get(pp,{}).get("Proposed_Split")=="Locked Test": continue
        targets.append((pp,mus))

    segidx=index_seg(args.segmentation_dir,{p for p,_ in targets})
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]

    for i,(pp,muscle_si) in enumerate(targets,1):
        print(f"[{i}/{len(targets)}] {pp}")
        row={"PatientID":pp}
        try:
            ref=refs.get((pp,LEVEL))
            if not ref or ref.get("Status")!="CONFIRMED_EXACT_MATCH":
                raise RuntimeError("No exact published L3-L4 mapping")
            row.update(process_patient(pp,manifest[pp],ref,segidx,args.dataset_dir,muscle_si))
            row.update({"Status":"OK","Error":""})
        except Exception as e:
            row.update({
                "QC_Tier":"D",
                "Automatic_Decision":"AUTO_EXCLUDE",
                "Automatic_Exclusion_Reasons":"PROCESSING_OR_HIERARCHICAL_QC_FAILURE",
                "Status":"FAILED","Error":str(e),
                "Four_Level_Muscle_Mean_SI":muscle_si
            })

        row["Manufacturer"]=manifest.get(pp,{}).get("Manufacturer","")
        row["FieldStrength_T"]=manifest.get(pp,{}).get("FieldStrength_T","")
        row["Proposed_Split"]=manifest.get(pp,{}).get("Proposed_Split","")
        rows.append(row)

        if i%10==0:
            write_csv(out/"pvmq_v5_2_results_partial.csv",rows)

    accepted=[r for r in rows if r.get("Automatic_Decision") in ("AUTO_ACCEPT","AUTO_ACCEPT_TARGET_ONLY")]
    excluded=[r for r in rows if r.get("Automatic_Decision")=="AUTO_EXCLUDE"]

    write_csv(out/"pvmq_v5_2_results.csv",rows)
    write_csv(out/"pvmq_v5_2_accepted.csv",accepted)
    write_csv(out/"pvmq_v5_2_excluded.csv",excluded)

    def vals(rs,key):
        arr=[]
        for r in rs:
            try:
                v=float(r.get(key,"nan"))
                if np.isfinite(v): arr.append(v)
            except: pass
        return np.array(arr,float)

    p=vals(accepted,"Final_PVMQ")
    tiers={t:sum(r.get("QC_Tier")==t for r in accepted) for t in ["A","B","C"]}
    write_csv(out/"pvmq_v5_2_summary.csv",[{
        "Target_With_4Level_Muscle_Numerator":len(rows),
        "AUTO_ACCEPT_TOTAL":len(accepted),
        "AUTO_EXCLUDE":len(excluded),
        "Acceptance_Rate":len(accepted)/len(rows) if rows else np.nan,
        "Tier_A_Accepted":tiers["A"],
        "Tier_B_Accepted":tiers["B"],
        "Tier_C_TargetOnly_Accepted":tiers["C"],
        "Tier_D_Excluded":len(excluded),
        "Processing_or_Hierarchical_QC_Failures":sum(r.get("Status")=="FAILED" for r in rows),
        "PVMQ_Mean":float(p.mean()) if len(p) else np.nan,
        "PVMQ_SD":float(p.std(ddof=1)) if len(p)>1 else np.nan,
        "PVMQ_Median":float(np.median(p)) if len(p) else np.nan,
        "PVMQ_Q1":float(np.percentile(p,25)) if len(p) else np.nan,
        "PVMQ_Q3":float(np.percentile(p,75)) if len(p) else np.nan,
        "PVMQ_Min":float(p.min()) if len(p) else np.nan,
        "PVMQ_Max":float(p.max()) if len(p) else np.nan,
        "PVMQ_Greater_Than_1":int(np.sum(p>1)) if len(p) else 0,
    }])

    # Tier-specific summary
    tier_rows=[]
    for tier in ["A","B","C"]:
        rs=[r for r in accepted if r.get("QC_Tier")==tier]
        pv=vals(rs,"Final_PVMQ")
        tier_rows.append({
            "QC_Tier":tier,"N":len(rs),
            "PVMQ_Median":float(np.median(pv)) if len(pv) else np.nan,
            "PVMQ_Mean":float(pv.mean()) if len(pv) else np.nan,
            "PVMQ_SD":float(pv.std(ddof=1)) if len(pv)>1 else np.nan,
            "PVMQ_Greater_Than_1":int(np.sum(pv>1)) if len(pv) else 0,
        })
    write_csv(out/"pvmq_v5_2_tier_summary.csv",tier_rows)

    groups={}
    for r in accepted:
        groups.setdefault((r.get("Manufacturer",""),r.get("FieldStrength_T","")),[]).append(r)
    gs=[]
    for (man,field),rs in groups.items():
        pv=vals(rs,"Final_PVMQ"); cs=vals(rs,"Final_Hierarchical_CSF_Mean_SI"); mu=vals(rs,"Four_Level_Muscle_Mean_SI")
        gs.append({
            "Manufacturer":man,"FieldStrength_T":field,"N":len(rs),
            "Tier_A":sum(r.get("QC_Tier")=="A" for r in rs),
            "Tier_B":sum(r.get("QC_Tier")=="B" for r in rs),
            "Tier_C":sum(r.get("QC_Tier")=="C" for r in rs),
            "PVMQ_Median":float(np.median(pv)) if len(pv) else np.nan,
            "PVMQ_Mean":float(pv.mean()) if len(pv) else np.nan,
            "PVMQ_SD":float(pv.std(ddof=1)) if len(pv)>1 else np.nan,
            "Muscle_SI_Median":float(np.median(mu)) if len(mu) else np.nan,
            "Hierarchical_CSF_SI_Median":float(np.median(cs)) if len(cs) else np.nan,
        })
    write_csv(out/"pvmq_v5_2_scanner_summary.csv",gs)

    print("Done.")
    print("Accepted:",len(accepted),"Excluded:",len(excluded),"Tiers:",tiers)

if __name__=="__main__":
    main()
