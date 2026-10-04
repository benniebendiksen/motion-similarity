#!/usr/bin/env python3
"""Shared-finish Stage 1: flatten+intersect the 3 routes into one HumanML3D corpus (nonmirror).
Keyed by distinct (source_path, window): whole-clip windows symlinked, genuine sub-windows cropped.
--dry_run resolves + reports counts only (gate). Else materializes hml3d_cmu_flat + hml3d_captioned.json."""
import json,os,glob,argparse,shutil
TR="/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets"
ANN=f"{TR}/learned_baselines/tmr/repo/datasets/annotations/humanml3d/annotations.json"
R1=f"{TR}/datasets/amass_cmu_flat"; R2AMASS=f"{TR}/datasets/_amass_cmu_cmu"; R3=f"{TR}/datasets/_humanact12_cmu"
OUTDIR=f"{TR}/datasets/hml3d_cmu_flat"; OUTJSON=f"{TR}/datasets/hml3d_captioned.json"
_strip=lambda x: x[:-6] if x.endswith("_poses") else x
ap=argparse.ArgumentParser(); ap.add_argument("--dry_run",action="store_true"); a=ap.parse_args()
d=json.load(open(ANN))

def resolve(path):
    """annotations path -> absolute source BVH across routes, or None."""
    top=path.split("/")[0]
    if top=="CMU":
        clip=_strip(path.split("/")[-1])  # e.g. 80_63
        p=f"{R2AMASS}/{clip}_cmu33_30fps.bvh"
        return p if os.path.exists(p) else None
    if top=="humanact12":
        clip=path.split("/")[-1]
        p=f"{R3}/{clip}_cmu33_30fps.bvh"
        return p if os.path.exists(p) else None
    stem=_strip(path.replace("/","_"))  # R1 AMASS
    p=f"{R1}/{stem}_cmu33_30fps.bvh"
    return p if os.path.exists(p) else None

def canon(path):
    return _strip(path.replace("/","_"))

def nframes(bvh):
    for l in open(bvh):
        if l.strip().startswith("Frames:"): return int(l.split()[-1])
    return None

# group nonmirror ids by source path; collect windows + captions
from collections import defaultdict
by_src=defaultdict(lambda:{"ids":[],"windows":defaultdict(lambda:{"caps":[],"ids":[]})})
for i,v in d.items():
    if v["path"].startswith("M/"): continue
    anns=v["annotations"]; src=v["path"]
    by_src[src]["ids"].append(i)
    # overall id window = [min start, max end] across its segments; captions = all seg texts
    s=round(min(x["start"] for x in anns),3); e=round(max(x["end"] for x in anns),3)
    w=by_src[src]["windows"][(s,e)]
    w["caps"]+= [x["text"] for x in anns]; w["ids"].append(i)

routes={"R1":0,"R2":0,"R3":0}; missing=[]; clips_whole=0; clips_crop=0; manifest={}
for src,info in by_src.items():
    bvh=resolve(src)
    top=src.split("/")[0]
    rk="R2" if top=="CMU" else ("R3" if top=="humanact12" else "R1")
    if bvh is None: missing.append((rk,src)); continue
    routes[rk]+=1
    tot=nframes(bvh) if not a.dry_run else None
    dur=tot/30.0 if tot else None
    for (s,e),w in info["windows"].items():
        stem=canon(src)
        whole = True  # default; refined when we know frame count
        if not a.dry_run:
            fs=max(0,round(s*30)); fe=min(tot,round(e*30)) if tot else None
            whole = (fs<=0 and (tot is None or fe>=tot-1))
            key = stem if whole else f"{stem}__w{int(round(s*1000))}_{int(round(e*1000))}"
            out=f"{OUTDIR}/{key}_cmu33_30fps.bvh"
            if whole:
                if not os.path.lexists(out): os.symlink(bvh,out); 
                clips_whole+=1
            else:
                L=open(bvh).read().splitlines()
                fti=[k for k,l in enumerate(L) if l.strip().lower().startswith("frame time")][0]
                hdr=L[:fti+1]; fr=L[fti+1:]; fr=fr[fs:fe]
                hdr=[("Frames: %d"%len(fr)) if l.strip().startswith("Frames:") else l for l in hdr]
                open(out,"w").write("\n".join(hdr+fr)+"\n"); clips_crop+=1
            if key in manifest:
                manifest[key]["ann_ids"]+=w["ids"]; manifest[key]["captions"]+=w["caps"]
            else:
                manifest[key]={"ann_ids":w["ids"],"path":src,"window":[s,e],"captions":w["caps"]}
        else:
            # dry: just count distinct windows (whole vs maybe-sub by duration)
            whole = dur is None or (abs(s)<0.05 and abs(e-dur)<0.05)
            clips_whole+=1 if whole else 0; clips_crop+=0 if whole else 1
print("routes resolved:",routes,"| missing:",len(missing),missing[:6])
print(f"distinct source-windows: whole~{clips_whole} crop~{clips_crop} total~{clips_whole+clips_crop}")
if not a.dry_run:
    [m.update(captions=list(dict.fromkeys(m["captions"]))) for m in manifest.values()]
    json.dump(manifest,open(OUTJSON,"w"))
    print("WROTE",OUTDIR,"(",len(glob.glob(OUTDIR+"/*.bvh")),"bvh ) +",OUTJSON,"(",len(manifest),"entries )")
