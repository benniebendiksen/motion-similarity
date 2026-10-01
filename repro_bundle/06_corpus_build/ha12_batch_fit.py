#!/usr/bin/env python3
"""HumanAct12 positions -> SMPL params via joints2smpl SMPLify3D (AMASS 22-joint), BATCHED.
Fits each clip in chunks of B frames in ONE batched SMPLify optimization (H200 parallelizes across
frames) -> seconds/clip instead of minutes. Frames fit independently (init from mean pose, seq_ind=0);
that is the standard MDM joints2smpl mode and is fine for a similarity corpus. Saves ONLY poses[F,72] +
trans[F,3] (smpl2bvh uses neutral betas, so betas do not reach the BVH) + reconstruction MPJPE. No .ply.
Array-shardable via --start/--end over the sorted clip list.
Grounding: subtract per-clip const [root_x0, min_y_all, root_z0] (frame0 root->XZ origin, lowest pt->floor)."""
import argparse, os, sys, glob, time
import numpy as np, torch
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from visualize.joints2smpl.src import config
config.SMPL_MODEL_DIR = os.path.join(HERE,"smpl_models")+"/"
config.GMM_MODEL_DIR  = os.path.join(HERE,"smpl_models")+"/"
config.SMPL_MEAN_FILE = os.path.join(HERE,"smpl_models","neutral_smpl_mean_params.h5")
import smplx, h5py
from visualize.joints2smpl.src.smplify import SMPLify3D

ap=argparse.ArgumentParser()
ap.add_argument("--src_dir",default="/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets/datasets/_humanact12_h3d/extracted/humanact12/humanact12")
ap.add_argument("--out_dir",default="/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets/datasets/_humanact12_smpl")
ap.add_argument("--num_iters",type=int,default=150)
ap.add_argument("--batch",type=int,default=128)
ap.add_argument("--start",type=int,default=0)
ap.add_argument("--end",type=int,default=-1)
ap.add_argument("--cuda",type=int,default=1)
a=ap.parse_args()
os.makedirs(a.out_dir,exist_ok=True)
dev=torch.device("cuda:0" if (a.cuda and torch.cuda.is_available()) else "cpu")
B=a.batch
print(f"device {dev} | iters {a.num_iters} | batch {B}",flush=True)

smpl=smplx.create(config.SMPL_MODEL_DIR,model_type="smpl",gender="neutral",ext="pkl",batch_size=B).to(dev)
mean=h5py.File(config.SMPL_MEAN_FILE,"r")
init_pose1=torch.from_numpy(mean["pose"][:]).float().to(dev).reshape(1,-1)
init_shape1=torch.from_numpy(mean["shape"][:]).float().to(dev).reshape(1,-1)
init_pose_B=init_pose1.repeat(B,1); init_shape_B=init_shape1.repeat(B,1)
smplify=SMPLify3D(smplxmodel=smpl,batch_size=B,joints_category="AMASS",num_iters=a.num_iters,device=dev)
conf=torch.ones(22).to(dev)

clips=sorted(glob.glob(os.path.join(a.src_dir,"*.npy")))
end=len(clips) if a.end<0 else min(a.end,len(clips))
clips=clips[a.start:end]
print("clips this shard:",len(clips),flush=True)

def ground(j):
    off=np.array([j[0,0,0], j[...,1].min(), j[0,0,2]],dtype=np.float32)
    return j-off[None,None,:]

def fit_chunk(kp_np):  # kp_np:[n<=B,22,3] -> pose[n,72],trans[n,3]
    n=kp_np.shape[0]
    pad=np.concatenate([kp_np, np.repeat(kp_np[-1:],B-n,axis=0)],axis=0) if n<B else kp_np
    kp=torch.tensor(pad,dtype=torch.float32).to(dev)
    _,_,npose,_,ncam,_=smplify(init_pose_B.detach(),init_shape_B.detach(),torch.zeros(B,3).to(dev),kp,conf_3d=conf,seq_ind=0)
    return npose.detach().cpu().numpy()[:n], ncam.detach().cpu().numpy().reshape(-1,3)[:n]

summ=[]
for ci,cf in enumerate(clips):
    name=os.path.splitext(os.path.basename(cf))[0]
    outp=os.path.join(a.out_dir,name+".npz")
    if os.path.exists(outp): continue
    raw=np.load(cf).astype(np.float32)[:,:22,:]
    sc=0.51/float(np.linalg.norm(raw[:,12]-raw[:,0],axis=-1).mean())  # normalize skeleton to SMPL scale
    j=ground(raw*sc); F=j.shape[0]
    poses=np.zeros((F,72),np.float32); trans=np.zeros((F,3),np.float32)
    t0=time.time()
    for c0 in range(0,F,B):
        p,t=fit_chunk(j[c0:c0+B]); poses[c0:c0+p.shape[0]]=p; trans[c0:c0+t.shape[0]]=t
    # MPJPE (chunked forward, neutral betas to match smpl2bvh)
    err=[]
    for c0 in range(0,F,B):
        n=min(B,F-c0); gp=torch.tensor(poses[c0:c0+n]).to(dev); tt=torch.tensor(trans[c0:c0+n]).to(dev)
        gp=torch.cat([gp,gp[-1:].repeat(B-n,1)],0); tt=torch.cat([tt,tt[-1:].repeat(B-n,1)],0)
        with torch.no_grad():
            o=smpl(betas=torch.zeros(B,10).to(dev),global_orient=gp[:,:3],body_pose=gp[:,3:],transl=tt,return_verts=False)
        rj=o.joints[:n,:22,:].cpu().numpy(); err.append(np.linalg.norm(rj-j[c0:c0+n],axis=-1))
    mpjpe=float(np.concatenate([e.reshape(-1) for e in err]).mean())*100.0
    np.savez(outp,poses=poses,trans=trans,mpjpe=mpjpe,fps=20,scale=sc)
    dt=time.time()-t0; summ.append((name,F,mpjpe,dt))
    print(f"[{ci+1}/{len(clips)}] {name} F={F} MPJPE={mpjpe:.2f}cm {dt:.1f}s",flush=True)
if summ:
    ms=np.array([s[2] for s in summ]); tt=np.array([s[3] for s in summ])
    print(f"DONE shard: n={len(summ)} MPJPE mean={ms.mean():.2f} med={np.median(ms):.2f} max={ms.max():.2f}cm | time/clip mean={tt.mean():.1f}s total={tt.sum():.0f}s",flush=True)
