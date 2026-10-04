#!/usr/bin/env python3
"""HA12 back-half Stage A: SMPL params (from ha12_batch_fit) -> SMPL-24 BVH @30fps.
Per clip: poses[F,72] -> [F,24,3] (already 24-joint SMPL axis-angle; NO palm pad, NO basis change —
HA12 was fit in Y-up), resample 20->30 (slerp rot / lerp trans), write smpl2bvh npz, call smpl2bvh.
Mirrors amass_smplh_to_bvh_batch.py Stage A. Array-shardable via --start/--end."""
import argparse, os, sys, glob, subprocess, numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
os.environ.setdefault("MKL_THREADING_LAYER","GNU")

def _t(n,fps): return np.arange(n,dtype=np.float64)/float(fps) if n>1 else np.array([0.0])
def resample_trans(tr,s,t):
    F=tr.shape[0]; to=_t(F,s); Fn=int(np.floor(to[-1]*t))+1; tn=np.arange(Fn)/float(t)
    return np.stack([np.interp(tn,to,tr[:,k]) for k in range(3)],1).astype(np.float32)
def resample_rotvec(rv,s,t):
    F,J,_=rv.shape; to=_t(F,s); Fn=int(np.floor(to[-1]*t))+1; tn=np.arange(Fn)/float(t)
    out=np.zeros((Fn,J,3),np.float32)
    for j in range(J):
        out[:,j]=Slerp(to,R.from_rotvec(rv[:,j])) (tn).as_rotvec().astype(np.float32)
    return out

ap=argparse.ArgumentParser()
ap.add_argument("--in_dir",default="/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets/datasets/_humanact12_smpl_scaled")
ap.add_argument("--out_npz_dir",default="/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets/datasets/_humanact12_bvhnpz")
ap.add_argument("--out_bvh_dir",default="/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets/datasets/_humanact12_smpl_bvh")
ap.add_argument("--smpl2bvh_repo",default="/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets/fit3d/third_party/smpl2bvh")
ap.add_argument("--python_exe",default="/home/p.bendiksen001/miniconda3/envs/j2s_gpu/bin/python")
ap.add_argument("--target_fps",type=float,default=30.0)
ap.add_argument("--start",type=int,default=0); ap.add_argument("--end",type=int,default=-1)
a=ap.parse_args()
os.makedirs(a.out_npz_dir,exist_ok=True); os.makedirs(a.out_bvh_dir,exist_ok=True)
clips=sorted(glob.glob(os.path.join(a.in_dir,"*.npz")))
end=len(clips) if a.end<0 else min(a.end,len(clips)); clips=clips[a.start:end]
print("clips:",len(clips),flush=True)
bad=[]
for ci,cf in enumerate(clips):
    name=os.path.splitext(os.path.basename(cf))[0]
    out_bvh=os.path.join(a.out_bvh_dir,name+".bvh")
    if os.path.exists(out_bvh): continue
    d=np.load(cf); poses=d["poses"].astype(np.float32); trans=d["trans"].astype(np.float32); src=float(d["fps"])
    F=poses.shape[0]; rot=poses.reshape(F,24,3)
    if abs(a.target_fps-src)>1e-6:
        rot=resample_rotvec(rot,src,a.target_fps); trans=resample_trans(trans,src,a.target_fps)
    npz=os.path.join(a.out_npz_dir,name+".npz")
    np.savez(npz,poses=rot[np.newaxis],trans=trans[np.newaxis])
    subprocess.run([a.python_exe,"smpl2bvh.py","--gender","NEUTRAL","--model_path","data/smpl",
        "--poses",os.path.abspath(npz),"--fps",str(int(round(a.target_fps))),"--output",os.path.abspath(out_bvh)],
        cwd=a.smpl2bvh_repo,check=True)
    # gate: 24 joints + 30fps
    L=open(out_bvh).read().splitlines()
    nj=sum(1 for l in L if l.strip().startswith(("ROOT","JOINT")))
    ft=[l for l in L if l.strip().lower().startswith("frame time")]
    ok=(nj==24 and ft and abs(float(ft[0].split()[-1])-1.0/a.target_fps)<1e-4)
    if not ok: bad.append((name,nj,ft[0].split()[-1] if ft else "noft"))
    if (ci+1)%100==0 or ci==0: print(f"[{ci+1}/{len(clips)}] {name} F{F}->{rot.shape[0]} joints={nj}",flush=True)
print("DONE bvh:",len(glob.glob(os.path.join(a.out_bvh_dir,"*.bvh"))),"| bad:",len(bad),bad[:5],flush=True)
