#!/usr/bin/env python
from __future__ import annotations

import argparse, csv, io, json, math, os, random, re, shutil, sys, time, zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

LEVELS = ["L1-L2","L2-L3","L3-L4","L4-L5","L5-S1"]
CLASSES = LEVELS + ["OTHER"]
LEVEL_TO_ID = {x:i for i,x in enumerate(LEVELS)}
OTHER_ID = 5

def read_csv(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def write_csv(p, rows, fieldnames=None):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames: w.writeheader()
        w.writerows(rows)

def pid(v):
    return f"{int(float(v)):04d}"

def resolve_archive(root, archive_name):
    root = Path(root)
    direct = root/archive_name
    if direct.exists(): return direct
    wanted = Path(archive_name).name.lower()
    hits = [p for p in root.rglob("*.zip") if p.name.lower()==wanted]
    if len(hits)==1: return hits[0]
    norm = re.sub(r"\(\d+\)", "", Path(archive_name).stem).replace(" ","").lower()
    hits = [p for p in root.rglob("*.zip")
            if re.sub(r"\(\d+\)", "", p.stem).replace(" ","").lower()==norm]
    if len(hits)==1: return hits[0]
    raise FileNotFoundError(f"Could not uniquely resolve archive {archive_name} under {root}")

def patient_member(name, patient):
    parts=[x for x in name.replace("\\","/").split("/") if x]
    return patient in parts

def safe_float(v):
    try: return float(v)
    except: return None

def normal_and_position(ds):
    try:
        iop=np.asarray([float(x) for x in ds.ImageOrientationPatient],dtype=float)
        ipp=np.asarray([float(x) for x in ds.ImagePositionPatient],dtype=float)
        row=iop[:3]; col=iop[3:]
        normal=np.cross(row,col)
        normal=normal/(np.linalg.norm(normal)+1e-12)
        return normal, ipp, float(ipp[2])
    except:
        return None,None,None

def build_index(args):
    manifest=read_csv(args.manifest)
    refs=read_csv(args.levels)
    exact={}
    for r in refs:
        if r.get("Status")=="CONFIRMED_EXACT_MATCH":
            exact[(pid(r["PatientID"]), Path(r["DICOM_File"]).name)] = r["Level"]

    by_archive=defaultdict(list)
    for r in manifest:
        by_archive[r["Archive"]].append(r)

    rows=[]
    archive_cache={}
    for ai,(archive_name, pts) in enumerate(sorted(by_archive.items()),1):
        archive=resolve_archive(args.dataset_dir, archive_name)
        print(f"[archive {ai}/{len(by_archive)}] {archive.name}", flush=True)
        with zipfile.ZipFile(archive) as z:
            names=z.namelist()
            for m in pts:
                p=pid(m["PatientID"])
                pnames=[n for n in names if patient_member(n,p) and not n.endswith("/")]
                for name in pnames:
                    try:
                        raw=z.read(name)
                        ds=pydicom.dcmread(io.BytesIO(raw),stop_before_pixels=True,force=True)
                        if getattr(ds,"SOPInstanceUID",None) is None: continue
                        normal,ipp,zcoord=normal_and_position(ds)
                        fn=Path(name).name
                        lev=exact.get((p,fn),"")
                        rows.append({
                            "PatientID":p,
                            "Proposed_Split":m["Proposed_Split"],
                            "Manufacturer":m.get("Manufacturer",""),
                            "FieldStrength_T":m.get("FieldStrength_T",""),
                            "Archive":str(archive),
                            "ZipMember":name,
                            "DICOM_File":fn,
                            "SeriesUID":str(getattr(ds,"SeriesInstanceUID","") or ""),
                            "SOPUID":str(getattr(ds,"SOPInstanceUID","") or ""),
                            "InstanceNumber":str(getattr(ds,"InstanceNumber","") or ""),
                            "IPP_X":str(ipp[0]) if ipp is not None else "",
                            "IPP_Y":str(ipp[1]) if ipp is not None else "",
                            "IPP_Z":str(ipp[2]) if ipp is not None else "",
                            "Normal_X":str(normal[0]) if normal is not None else "",
                            "Normal_Y":str(normal[1]) if normal is not None else "",
                            "Normal_Z":str(normal[2]) if normal is not None else "",
                            "SliceThickness":str(getattr(ds,"SliceThickness","") or ""),
                            "Exact_Level":lev,
                            "Train_Label":"",
                            "Is_Exact":"Yes" if lev else "No",
                        })
                    except Exception:
                        continue

    # Label exact slices and immediate physical neighbors in same series.
    by_patient_series=defaultdict(list)
    for i,r in enumerate(rows):
        by_patient_series[(r["PatientID"],r["SeriesUID"])].append(i)

    for key, inds in by_patient_series.items():
        def order_value(i):
            r=rows[i]
            z=safe_float(r["IPP_Z"])
            inst=safe_float(r["InstanceNumber"])
            return z if z is not None else (inst if inst is not None else i)
        inds_sorted=sorted(inds,key=order_value)
        pos={idx:j for j,idx in enumerate(inds_sorted)}
        proposed={}
        for idx in inds_sorted:
            lev=rows[idx]["Exact_Level"]
            if lev:
                j=pos[idx]
                proposed[idx]=lev
                for jj in (j-1,j+1):
                    if 0<=jj<len(inds_sorted):
                        proposed.setdefault(inds_sorted[jj],lev)
        # If a slice is adjacent to two different exact targets, keep as OTHER.
        collisions=defaultdict(set)
        for idx in inds_sorted:
            lev=rows[idx]["Exact_Level"]
            if not lev: continue
            j=pos[idx]
            for jj in (j-1,j,j+1):
                if 0<=jj<len(inds_sorted):
                    collisions[inds_sorted[jj]].add(lev)
        for idx in inds_sorted:
            labs=collisions.get(idx,set())
            if len(labs)==1:
                rows[idx]["Train_Label"]=next(iter(labs))
            else:
                rows[idx]["Train_Label"]="OTHER"

    # Ensure every row labeled.
    for r in rows:
        if not r["Train_Label"]: r["Train_Label"]="OTHER"

    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"dicom_index.csv",rows)
    counts=defaultdict(int)
    for r in rows:
        counts[(r["Proposed_Split"],r["Train_Label"])]+=1
    summary=[{"Split":k[0],"Label":k[1],"N":v} for k,v in sorted(counts.items())]
    write_csv(out/"index_summary.csv",summary)
    print(f"Indexed {len(rows)} DICOMs -> {out/'dicom_index.csv'}")

class ZipPixelStore:
    def __init__(self):
        self.handles={}
    def read(self, archive, member):
        if archive not in self.handles:
            self.handles[archive]=zipfile.ZipFile(archive)
        raw=self.handles[archive].read(member)
        ds=pydicom.dcmread(io.BytesIO(raw),force=True)
        arr=ds.pixel_array.astype(np.float32)
        slope=safe_float(getattr(ds,"RescaleSlope",1)) or 1.0
        intercept=safe_float(getattr(ds,"RescaleIntercept",0)) or 0.0
        arr=arr*slope+intercept
        lo,hi=np.percentile(arr,[1,99])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi<=lo:
            lo=float(np.min(arr)); hi=float(np.max(arr))+1e-6
        arr=np.clip((arr-lo)/(hi-lo),0,1)
        return (arr*255).astype(np.uint8)

class SliceDataset(Dataset):
    def __init__(self, rows, train=False, image_size=224):
        self.rows=rows; self.store=ZipPixelStore()
        if train:
            self.tf=transforms.Compose([
                transforms.Resize((image_size,image_size)),
                transforms.RandomRotation(5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.12,contrast=0.12),
                transforms.ToTensor(),
                transforms.Normalize([0.485]*3,[0.229]*3),
            ])
        else:
            self.tf=transforms.Compose([
                transforms.Resize((image_size,image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485]*3,[0.229]*3),
            ])
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        arr=self.store.read(r["Archive"],r["ZipMember"])
        im=Image.fromarray(arr,mode="L").convert("RGB")
        x=self.tf(im)
        y=LEVEL_TO_ID.get(r["Train_Label"],OTHER_ID)
        return x,y,i

def make_model(pretrained=True):
    try:
        weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        model=models.resnet18(weights=weights)
    except Exception as e:
        print("WARNING: pretrained weights unavailable; training from random initialization:",e)
        model=models.resnet18(weights=None)
    model.fc=nn.Linear(model.fc.in_features,len(CLASSES))
    return model

def split_rows(index, split):
    if split=="train":
        return [r for r in index if r["Proposed_Split"]=="Fine-tuning/Training"]
    if split=="val":
        return [r for r in index if r["Proposed_Split"]=="Development Validation"]
    if split=="locked":
        return [r for r in index if r["Proposed_Split"]=="Locked Test"]
    raise ValueError(split)

def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    index=read_csv(args.index)
    tr=split_rows(index,"train"); va=split_rows(index,"val")
    print(f"train slices={len(tr)} val slices={len(va)}")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:",device)

    train_ds=SliceDataset(tr,train=True,image_size=args.image_size)
    val_ds=SliceDataset(va,train=False,image_size=args.image_size)

    counts=np.zeros(len(CLASSES),dtype=int)
    labels=[]
    for r in tr:
        y=LEVEL_TO_ID.get(r["Train_Label"],OTHER_ID); counts[y]+=1; labels.append(y)
    weights=1.0/np.maximum(counts,1)
    sample_weights=[weights[y] for y in labels]
    sampler=WeightedRandomSampler(sample_weights,num_samples=len(sample_weights),replacement=True)

    train_loader=DataLoader(train_ds,batch_size=args.batch_size,sampler=sampler,num_workers=0,pin_memory=False)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=0)

    model=make_model(pretrained=not args.no_pretrained).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    criterion=nn.CrossEntropyLoss()

    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    best=float("inf"); history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); total=correct=n=0
        for x,y,_ in train_loader:
            x=x.to(device); y=y.to(device)
            opt.zero_grad(set_to_none=True)
            logits=model(x); loss=criterion(logits,y)
            loss.backward(); opt.step()
            total+=float(loss.item())*len(y); correct+=int((logits.argmax(1)==y).sum()); n+=len(y)
        scheduler.step()

        model.eval(); vtotal=vcorrect=vn=0
        with torch.no_grad():
            for x,y,_ in val_loader:
                x=x.to(device); y=y.to(device)
                logits=model(x); loss=criterion(logits,y)
                vtotal+=float(loss.item())*len(y); vcorrect+=int((logits.argmax(1)==y).sum()); vn+=len(y)
        row={"epoch":epoch,"train_loss":total/n,"train_acc":correct/n,
             "val_loss":vtotal/vn,"val_acc":vcorrect/vn,"lr":opt.param_groups[0]["lr"]}
        history.append(row); write_csv(out/"training_history.csv",history)
        print(f"epoch {epoch:02d}: train loss={row['train_loss']:.4f} acc={row['train_acc']:.3f} | val loss={row['val_loss']:.4f} acc={row['val_acc']:.3f}",flush=True)
        if row["val_loss"]<best:
            best=row["val_loss"]
            torch.save({"state_dict":model.state_dict(),"classes":CLASSES,
                        "image_size":args.image_size,"epoch":epoch,"val_loss":best},
                       out/"best_model.pt")
    print("Best checkpoint:",out/"best_model.pt")

def predict_rows(model, rows, device, batch_size, image_size):
    ds=SliceDataset(rows,train=False,image_size=image_size)
    loader=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0)
    probs=np.zeros((len(rows),len(CLASSES)),dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for x,_,idx in loader:
            logits=model(x.to(device))
            p=torch.softmax(logits,dim=1).cpu().numpy()
            probs[idx.numpy()]=p
    return probs

def zval(r):
    z=safe_float(r.get("IPP_Z"))
    if z is not None: return z
    inst=safe_float(r.get("InstanceNumber"))
    return inst if inst is not None else 0.0

def ordered_decode(rows, probs):
    # Cranial -> caudal: larger patient-Z is usually more superior in standard lumbar MRI.
    order=sorted(range(len(rows)),key=lambda i:zval(rows[i]),reverse=True)
    n=len(order); k=len(LEVELS)
    score=np.log(np.clip(probs[:,0:k],1e-8,1.0))
    dp=np.full((k,n),-1e30,dtype=float)
    prev=np.full((k,n),-1,dtype=int)
    for j,idx in enumerate(order): dp[0,j]=score[idx,0]
    for lev in range(1,k):
        best=-1e30; bestj=-1
        for j,idx in enumerate(order):
            if j>0 and dp[lev-1,j-1]>best:
                best=dp[lev-1,j-1]; bestj=j-1
            if bestj>=0:
                dp[lev,j]=best+score[idx,lev]; prev[lev,j]=bestj
    j=int(np.argmax(dp[k-1]))
    picks=[None]*k
    for lev in range(k-1,-1,-1):
        picks[lev]=order[j]
        if lev>0: j=int(prev[lev,j])
    return picks

def plane_distance_mm(pred, gt):
    try:
        ipp=np.array([float(pred["IPP_X"]),float(pred["IPP_Y"]),float(pred["IPP_Z"])])
        gipp=np.array([float(gt["IPP_X"]),float(gt["IPP_Y"]),float(gt["IPP_Z"])])
        n=np.array([float(gt["Normal_X"]),float(gt["Normal_Y"]),float(gt["Normal_Z"])])
        return abs(float(np.dot(ipp-gipp,n)))
    except: return None

def native_spacing(patient_rows, gt):
    same=[r for r in patient_rows if r["SeriesUID"]==gt["SeriesUID"]]
    vals=[]
    try:
        n=np.array([float(gt["Normal_X"]),float(gt["Normal_Y"]),float(gt["Normal_Z"])])
        for r in same:
            ipp=np.array([float(r["IPP_X"]),float(r["IPP_Y"]),float(r["IPP_Z"])])
            vals.append(float(np.dot(ipp,n)))
        vals=sorted(set(round(v,5) for v in vals))
        diffs=np.abs(np.diff(vals))
        diffs=diffs[diffs>1e-5]
        if len(diffs): return float(np.median(diffs))
    except: pass
    return safe_float(gt.get("SliceThickness"))

def evaluate(args, split):
    if split=="locked" and not args.confirm_locked:
        raise SystemExit("LOCKED TEST BLOCKED. Re-run only after method freeze with --confirm-locked.")
    index=read_csv(args.index)
    rows=split_rows(index,split)
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=make_model(pretrained=False); model.load_state_dict(ck["state_dict"]); model.to(device)
    probs=predict_rows(model,rows,device,args.batch_size,int(ck.get("image_size",224)))

    byp=defaultdict(list)
    for i,r in enumerate(rows): byp[r["PatientID"]].append(i)
    results=[]
    for patient, global_inds in sorted(byp.items()):
        pr=[rows[i] for i in global_inds]
        pp=probs[global_inds]
        picks=ordered_decode(pr,pp)
        exact_by_level={r["Exact_Level"]:r for r in pr if r["Exact_Level"] in LEVELS}
        for lev,k in zip(LEVELS,picks):
            pred=pr[k]; gt=exact_by_level.get(lev)
            if gt is None:
                results.append({"PatientID":patient,"Level":lev,"Status":"NO_REFERENCE"})
                continue
            dist=plane_distance_mm(pred,gt)
            spacing=native_spacing(pr,gt)
            ratio=dist/spacing if dist is not None and spacing else None
            results.append({
                "PatientID":patient,"Level":lev,"Status":"OK",
                "Predicted_DICOM":pred["DICOM_File"],"Reference_DICOM":gt["DICOM_File"],
                "Exact_Match":"Yes" if pred["DICOM_File"]==gt["DICOM_File"] else "No",
                "Distance_mm":dist if dist is not None else "",
                "Native_Spacing_mm":spacing if spacing else "",
                "Distance_in_Slices":ratio if ratio is not None else "",
                "Within_1_Slice":"Yes" if ratio is not None and ratio<=1.5 else "No",
                "Predicted_Probability":float(pp[k,LEVEL_TO_ID[lev]]),
                "Manufacturer":pred["Manufacturer"],"FieldStrength_T":pred["FieldStrength_T"],
                "Proposed_Split":pred["Proposed_Split"],
            })
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/f"{split}_predictions.csv",results)
    ok=[r for r in results if r.get("Status")=="OK"]
    summary=[]
    for lev in LEVELS+["ALL"]:
        rr=ok if lev=="ALL" else [r for r in ok if r["Level"]==lev]
        if not rr: continue
        d=[float(r["Distance_mm"]) for r in rr if r["Distance_mm"]!=""]
        summary.append({
            "Level":lev,"N":len(rr),
            "Exact_Match_Rate":sum(r["Exact_Match"]=="Yes" for r in rr)/len(rr),
            "Within_1_Slice_Rate":sum(r["Within_1_Slice"]=="Yes" for r in rr)/len(rr),
            "Median_Distance_mm":float(np.median(d)) if d else "",
            "P90_Distance_mm":float(np.percentile(d,90)) if d else "",
        })
    write_csv(out/f"{split}_summary.csv",summary)
    print(json.dumps(summary,indent=2))
    print("Output:",out)

def infer(args):
    # New-study inference from an extracted DICOM folder.
    # This does not generate PVMQ; it produces candidate disc-level DICOM files.
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    tmp=[]
    for p in Path(args.study_dir).rglob("*"):
        if not p.is_file(): continue
        try:
            ds=pydicom.dcmread(str(p),stop_before_pixels=True,force=True)
            if getattr(ds,"SOPInstanceUID",None) is None: continue
            normal,ipp,zcoord=normal_and_position(ds)
            tmp.append({
                "Path":str(p),"Archive":"","ZipMember":"","DICOM_File":p.name,
                "PatientID":str(getattr(ds,"PatientID","") or ""),
                "SeriesUID":str(getattr(ds,"SeriesInstanceUID","") or ""),
                "InstanceNumber":str(getattr(ds,"InstanceNumber","") or ""),
                "IPP_X":str(ipp[0]) if ipp is not None else "",
                "IPP_Y":str(ipp[1]) if ipp is not None else "",
                "IPP_Z":str(ipp[2]) if ipp is not None else "",
                "Normal_X":str(normal[0]) if normal is not None else "",
                "Normal_Y":str(normal[1]) if normal is not None else "",
                "Normal_Z":str(normal[2]) if normal is not None else "",
                "SliceThickness":str(getattr(ds,"SliceThickness","") or ""),
                "Train_Label":"OTHER","Exact_Level":"",
            })
        except: pass
    if not tmp: raise SystemExit("No readable DICOMs.")

    class FolderDS(Dataset):
        def __init__(self,rows,size):
            self.rows=rows
            self.tf=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),
                                        transforms.Normalize([0.485]*3,[0.229]*3)])
        def __len__(self): return len(self.rows)
        def __getitem__(self,i):
            ds=pydicom.dcmread(self.rows[i]["Path"],force=True)
            a=ds.pixel_array.astype(np.float32)
            lo,hi=np.percentile(a,[1,99]); a=np.clip((a-lo)/(hi-lo+1e-6),0,1)
            im=Image.fromarray((a*255).astype(np.uint8),"L").convert("RGB")
            return self.tf(im),0,i

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=make_model(pretrained=False); model.load_state_dict(ck["state_dict"]); model.to(device); model.eval()
    ds=FolderDS(tmp,int(ck.get("image_size",224)))
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0)
    probs=np.zeros((len(tmp),len(CLASSES)),dtype=np.float32)
    with torch.no_grad():
        for x,_,idx in loader:
            probs[idx.numpy()]=torch.softmax(model(x.to(device)),1).cpu().numpy()
    picks=ordered_decode(tmp,probs)
    out=[]
    for lev,k in zip(LEVELS,picks):
        out.append({"Level":lev,"DICOM_File":tmp[k]["DICOM_File"],"Path":tmp[k]["Path"],
                    "Probability":float(probs[k,LEVEL_TO_ID[lev]])})
    write_csv(args.output,out)
    print(json.dumps(out,indent=2))

def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="cmd",required=True)

    p=sub.add_parser("index")
    p.add_argument("--dataset-dir",required=True); p.add_argument("--manifest",default="manifest500.csv")
    p.add_argument("--levels",default="patient_level_reference.csv"); p.add_argument("--output-dir",default="LEVEL_V2_INDEX")

    p=sub.add_parser("train")
    p.add_argument("--index",required=True); p.add_argument("--output-dir",default="LEVEL_V2_MODEL")
    p.add_argument("--epochs",type=int,default=20); p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--image-size",type=int,default=224); p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--seed",type=int,default=20260828); p.add_argument("--no-pretrained",action="store_true")

    for name in ("validate","locked"):
        p=sub.add_parser(name); p.add_argument("--index",required=True); p.add_argument("--checkpoint",required=True)
        p.add_argument("--output-dir",default=f"LEVEL_V2_{name.upper()}"); p.add_argument("--batch-size",type=int,default=32)
        p.add_argument("--confirm-locked",action="store_true")

    p=sub.add_parser("infer")
    p.add_argument("--study-dir",required=True); p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",default="predicted_levels.csv"); p.add_argument("--batch-size",type=int,default=32)

    args=ap.parse_args()
    if args.cmd=="index": build_index(args)
    elif args.cmd=="train": train(args)
    elif args.cmd=="validate": evaluate(args,"val")
    elif args.cmd=="locked": evaluate(args,"locked")
    elif args.cmd=="infer": infer(args)

if __name__=="__main__":
    main()
