
from __future__ import annotations
import argparse, io, json, math, random, zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

from level_localizer_v2 import (
    LEVELS, read_csv, write_csv, split_rows, safe_float
)

PVMQ_LEVELS = ["L1-L2","L2-L3","L3-L4","L4-L5"]

def vec3(r):
    try:
        return np.array([float(r["IPP_X"]),float(r["IPP_Y"]),float(r["IPP_Z"])],dtype=float)
    except Exception:
        return None

def normal3(r):
    try:
        n=np.array([float(r["Normal_X"]),float(r["Normal_Y"]),float(r["Normal_Z"])],dtype=float)
        q=np.linalg.norm(n)
        return n/q if q>0 else None
    except Exception:
        return None

def plane_distance_mm(a,b):
    pa,pb,n=vec3(a),vec3(b),normal3(b)
    if pa is None or pb is None or n is None:
        return None
    return abs(float(np.dot(pa-pb,n)))

def native_spacing(patient_rows, gt):
    n=normal3(gt)
    if n is None:
        return safe_float(gt.get("SliceThickness"))
    vals=[]
    for r in patient_rows:
        if r.get("SeriesUID") != gt.get("SeriesUID"):
            continue
        p=vec3(r)
        if p is not None:
            vals.append(float(np.dot(p,n)))
    vals=sorted(set(round(v,5) for v in vals))
    d=np.abs(np.diff(vals))
    d=d[d>1e-5]
    if len(d):
        return float(np.median(d))
    return safe_float(gt.get("SliceThickness"))

def make_targets(index_rows, sigma_slices=1.25):
    """
    Continuous target for each level:
      y = exp(-0.5 * (distance_mm / (sigma_slices*native_spacing))^2)
    Only slices belonging to the exact reference SeriesUID get a nonzero target.
    This prevents geometrically unrelated series from being treated as near-center.
    """
    byp=defaultdict(list)
    for i,r in enumerate(index_rows):
        byp[r["PatientID"]].append(i)

    targets=np.zeros((len(index_rows),len(LEVELS)),dtype=np.float32)
    meta=[]
    for patient, inds in byp.items():
        pr=[index_rows[i] for i in inds]
        gtmap={r["Exact_Level"]:r for r in pr if r.get("Exact_Level") in LEVELS}
        for li,lev in enumerate(LEVELS):
            gt=gtmap.get(lev)
            if gt is None:
                continue
            sp=native_spacing(pr,gt) or safe_float(gt.get("SliceThickness")) or 4.0
            sigma=max(1.0,float(sp)*sigma_slices)
            for local_i,global_i in enumerate(inds):
                r=pr[local_i]
                if r.get("SeriesUID") != gt.get("SeriesUID"):
                    continue
                d=plane_distance_mm(r,gt)
                if d is None:
                    continue
                targets[global_i,li]=math.exp(-0.5*(d/sigma)**2)
    return targets

class ZipPixelStore:
    def __init__(self):
        self.handles={}
    def read(self, archive, member):
        if archive not in self.handles:
            self.handles[archive]=zipfile.ZipFile(archive)
        raw=self.handles[archive].read(member)
        ds=pydicom.dcmread(io.BytesIO(raw),force=True)
        a=ds.pixel_array.astype(np.float32)
        slope=safe_float(getattr(ds,"RescaleSlope",1)) or 1.0
        intercept=safe_float(getattr(ds,"RescaleIntercept",0)) or 0.0
        a=a*slope+intercept
        lo,hi=np.percentile(a,[1,99])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi<=lo:
            lo=float(np.min(a)); hi=float(np.max(a))+1e-6
        a=np.clip((a-lo)/(hi-lo),0,1)
        return (a*255).astype(np.uint8)

class ContinuousDataset(Dataset):
    def __init__(self, rows, targets, train=False, size=224):
        self.rows=rows
        self.targets=targets
        self.store=ZipPixelStore()
        if train:
            self.tf=transforms.Compose([
                transforms.Resize((size,size)),
                transforms.RandomRotation(5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.10,contrast=0.10),
                transforms.ToTensor(),
                transforms.Normalize([0.485]*3,[0.229]*3),
            ])
        else:
            self.tf=transforms.Compose([
                transforms.Resize((size,size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485]*3,[0.229]*3),
            ])
    def __len__(self):
        return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        a=self.store.read(r["Archive"],r["ZipMember"])
        im=Image.fromarray(a,"L").convert("RGB")
        return self.tf(im), torch.tensor(self.targets[i],dtype=torch.float32), i

def make_model(pretrained=True):
    try:
        weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        m=models.resnet18(weights=weights)
    except Exception as e:
        print("WARNING: pretrained initialization unavailable:",e)
        m=models.resnet18(weights=None)
    m.fc=nn.Linear(m.fc.in_features,len(LEVELS))
    return m

def split_with_targets(index, targets, split):
    if split=="train":
        mask=[r["Proposed_Split"]=="Fine-tuning/Training" for r in index]
    elif split=="val":
        mask=[r["Proposed_Split"]=="Development Validation" for r in index]
    elif split=="locked":
        mask=[r["Proposed_Split"]=="Locked Test" for r in index]
    else:
        raise ValueError(split)
    inds=[i for i,x in enumerate(mask) if x]
    return [index[i] for i in inds], targets[inds]

def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    index=read_csv(args.index)
    targets=make_targets(index,args.sigma_slices)
    tr,ty=split_with_targets(index,targets,"train")
    va,vy=split_with_targets(index,targets,"val")
    print(f"training slices={len(tr)}  validation slices={len(va)}")
    print(f"target sigma={args.sigma_slices} native slices")

    # Oversample slices near any exact disc center while retaining background.
    near=np.max(ty,axis=1)>=0.20
    w=np.where(near,4.0,1.0)
    sampler=WeightedRandomSampler(torch.tensor(w,dtype=torch.double),len(w),replacement=True)

    td=ContinuousDataset(tr,ty,train=True,size=args.image_size)
    vd=ContinuousDataset(va,vy,train=False,size=args.image_size)
    tl=DataLoader(td,batch_size=args.batch_size,sampler=sampler,num_workers=0)
    vl=DataLoader(vd,batch_size=args.batch_size,shuffle=False,num_workers=0)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:",device)
    model=make_model(pretrained=not args.no_pretrained).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)

    # Soft-label BCE makes the output interpretable as a center-likeness score.
    crit=nn.BCEWithLogitsLoss()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    best=float("inf"); hist=[]
    for epoch in range(1,args.epochs+1):
        model.train(); total=n=0
        for x,y,_ in tl:
            x=x.to(device); y=y.to(device)
            opt.zero_grad(set_to_none=True)
            z=model(x); loss=crit(z,y)
            loss.backward(); opt.step()
            total+=float(loss.item())*len(y); n+=len(y)
        sch.step()

        model.eval(); vt=vn=0
        with torch.no_grad():
            for x,y,_ in vl:
                x=x.to(device); y=y.to(device)
                loss=crit(model(x),y)
                vt+=float(loss.item())*len(y); vn+=len(y)
        row={"epoch":epoch,"train_loss":total/n,"val_loss":vt/vn,"lr":opt.param_groups[0]["lr"]}
        hist.append(row); write_csv(out/"training_history.csv",hist)
        print(f"epoch {epoch:02d}: train={row['train_loss']:.5f} val={row['val_loss']:.5f}",flush=True)
        if row["val_loss"]<best:
            best=row["val_loss"]
            torch.save({
                "state_dict":model.state_dict(),
                "levels":LEVELS,
                "image_size":args.image_size,
                "sigma_slices":args.sigma_slices,
                "epoch":epoch,
                "val_loss":best,
                "model_type":"continuous_disc_center_v3"
            },out/"best_model.pt")
    print("Best model:",out/"best_model.pt")

def score_rows(model, rows, targets, device, batch_size, image_size):
    ds=ContinuousDataset(rows,targets,train=False,size=image_size)
    dl=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0)
    scores=np.zeros((len(rows),len(LEVELS)),dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for x,_,idx in dl:
            s=torch.sigmoid(model(x.to(device))).cpu().numpy()
            scores[idx.numpy()]=s
    return scores

def independent_peaks(rows,scores):
    """
    v3 intentionally chooses each disc-center peak independently.
    Exact annotations in this dataset can reside in different axial series/stacks,
    so forcing all levels into one SeriesUID is not appropriate.
    """
    picks=[]
    for li,lev in enumerate(LEVELS):
        picks.append(int(np.argmax(scores[:,li])))
    return picks

def patient_qc_features(rows,scores,picks):
    peaks=[float(scores[k,i]) for i,k in enumerate(picks)]
    margins=[]
    for i,k in enumerate(picks):
        col=scores[:,i]
        order=np.argsort(col)[::-1]
        top=float(col[k])
        second=float(col[order[1]]) if len(order)>1 and order[0]==k else float(col[order[0]])
        margins.append(top-second)
    return {
        "min_peak_all5":float(np.min(peaks)),
        "min_peak_pvmq4":float(np.min(peaks[:4])),
        "mean_peak_pvmq4":float(np.mean(peaks[:4])),
        "min_margin_pvmq4":float(np.min(margins[:4])),
        "unique_series_pvmq4":len(set(rows[picks[i]].get("SeriesUID","") for i in range(4))),
    }

def evaluate(args, split):
    if split=="locked" and not args.confirm_locked:
        raise SystemExit("LOCKED TEST BLOCKED. Only run after v3 method freeze with --confirm-locked.")
    index=read_csv(args.index)
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    sigma=float(ck.get("sigma_slices",1.25))
    targets=make_targets(index,sigma)
    rows,yt=split_with_targets(index,targets,split)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=make_model(pretrained=False)
    model.load_state_dict(ck["state_dict"]); model.to(device)
    scores=score_rows(model,rows,yt,device,args.batch_size,int(ck.get("image_size",224)))

    byp=defaultdict(list)
    for i,r in enumerate(rows): byp[r["PatientID"]].append(i)

    level_out=[]; patient_out=[]
    for patient,ginds in sorted(byp.items()):
        pr=[rows[i] for i in ginds]; ps=scores[ginds]
        picks=independent_peaks(pr,ps)
        feat=patient_qc_features(pr,ps,picks)
        gtmap={r["Exact_Level"]:r for r in pr if r.get("Exact_Level") in LEVELS}
        pvmq_eval=pvmq_good=0
        all5_eval=all5_good=0
        for li,(lev,k) in enumerate(zip(LEVELS,picks)):
            pred=pr[k]; gt=gtmap.get(lev)
            if gt is None:
                level_out.append({"PatientID":patient,"Level":lev,"Status":"NO_REFERENCE"})
                continue
            d=plane_distance_mm(pred,gt)
            sp=native_spacing(pr,gt)
            ratio=d/sp if d is not None and sp else None
            within=bool(ratio is not None and ratio<=1.5)
            all5_eval+=1; all5_good+=int(within)
            if lev in PVMQ_LEVELS:
                pvmq_eval+=1; pvmq_good+=int(within)
            level_out.append({
                "PatientID":patient,"Level":lev,"Status":"OK",
                "Predicted_DICOM":pred["DICOM_File"],
                "Reference_DICOM":gt["DICOM_File"],
                "Predicted_SeriesUID":pred["SeriesUID"],
                "Reference_SeriesUID":gt["SeriesUID"],
                "Same_Series":"Yes" if pred["SeriesUID"]==gt["SeriesUID"] else "No",
                "Exact_Match":"Yes" if pred["DICOM_File"]==gt["DICOM_File"] else "No",
                "Distance_mm":d if d is not None else "",
                "Native_Spacing_mm":sp if sp else "",
                "Distance_in_Slices":ratio if ratio is not None else "",
                "Within_1_Slice":"Yes" if within else "No",
                "Center_Score":float(ps[k,li]),
                "Manufacturer":pred.get("Manufacturer",""),
                "FieldStrength_T":pred.get("FieldStrength_T",""),
            })
        patient_out.append({
            "PatientID":patient,
            "Manufacturer":pr[0].get("Manufacturer",""),
            "FieldStrength_T":pr[0].get("FieldStrength_T",""),
            **feat,
            "PVMQ4_Evaluable":pvmq_eval,
            "PVMQ4_Within1_Count":pvmq_good,
            "PVMQ4_All_Within1":int(pvmq_eval==4 and pvmq_good==4),
            "All5_Evaluable":all5_eval,
            "All5_Within1_Count":all5_good,
            "All5_All_Within1":int(all5_eval==5 and all5_good==5),
        })

    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/f"{split}_level_predictions.csv",level_out)
    write_csv(out/f"{split}_patient_results.csv",patient_out)

    ok=[r for r in level_out if r.get("Status")=="OK"]
    summary=[]
    for lev in LEVELS+["PVMQ4","ALL5"]:
        if lev=="PVMQ4":
            rr=[r for r in ok if r["Level"] in PVMQ_LEVELS]
        elif lev=="ALL5":
            rr=ok
        else:
            rr=[r for r in ok if r["Level"]==lev]
        if not rr: continue
        d=[float(r["Distance_mm"]) for r in rr if r["Distance_mm"]!=""]
        summary.append({
            "Scope":lev,"N":len(rr),
            "Exact_Match_Rate":sum(r["Exact_Match"]=="Yes" for r in rr)/len(rr),
            "Within_1_Slice_Rate":sum(r["Within_1_Slice"]=="Yes" for r in rr)/len(rr),
            "Median_Distance_mm":float(np.median(d)) if d else "",
            "P90_Distance_mm":float(np.percentile(d,90)) if d else "",
            "Max_Distance_mm":float(np.max(d)) if d else "",
        })

    full4=[r for r in patient_out if int(r["PVMQ4_Evaluable"])==4]
    full5=[r for r in patient_out if int(r["All5_Evaluable"])==5]
    summary.append({
        "Scope":"PATIENT_PVMQ4","N":len(full4),
        "Exact_Match_Rate":"",
        "Within_1_Slice_Rate":sum(int(r["PVMQ4_All_Within1"]) for r in full4)/len(full4) if full4 else "",
        "Median_Distance_mm":"","P90_Distance_mm":"","Max_Distance_mm":""
    })
    summary.append({
        "Scope":"PATIENT_ALL5","N":len(full5),
        "Exact_Match_Rate":"",
        "Within_1_Slice_Rate":sum(int(r["All5_All_Within1"]) for r in full5)/len(full5) if full5 else "",
        "Median_Distance_mm":"","P90_Distance_mm":"","Max_Distance_mm":""
    })
    write_csv(out/f"{split}_summary.csv",summary)
    print(json.dumps(summary,indent=2))
    print("Output:",out)

def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="cmd",required=True)

    p=sub.add_parser("train")
    p.add_argument("--index",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--image-size",type=int,default=224)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--sigma-slices",type=float,default=1.25)
    p.add_argument("--seed",type=int,default=20260831)
    p.add_argument("--no-pretrained",action="store_true")

    for name in ("validate","locked"):
        p=sub.add_parser(name)
        p.add_argument("--index",required=True)
        p.add_argument("--checkpoint",required=True)
        p.add_argument("--output-dir",required=True)
        p.add_argument("--batch-size",type=int,default=32)
        p.add_argument("--confirm-locked",action="store_true")

    args=ap.parse_args()
    if args.cmd=="train":
        train(args)
    elif args.cmd=="validate":
        evaluate(args,"val")
    elif args.cmd=="locked":
        evaluate(args,"locked")

if __name__=="__main__":
    main()
