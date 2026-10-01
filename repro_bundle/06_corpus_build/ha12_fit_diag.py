#!/usr/bin/env python3
"""Head-to-head: for given clips, fit with (a) batched@Niter and (b) per-frame warm-start@Niter.
Disentangles whether the tail MPJPE is fixed by warm-start or just by more iters. Reports MPJPE + time."""
import os,sys,glob,time,numpy as np,torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from visualize.joints2smpl.src import config
config.SMPL_MODEL_DIR=os.path.join(HERE,"smpl_models")+"/"; config.GMM_MODEL_DIR=os.path.join(HERE,"smpl_models")+"/"
config.SMPL_MEAN_FILE=os.path.join(HERE,"smpl_models","neutral_smpl_mean_params.h5")
import smplx,h5py
from visualize.joints2smpl.src.smplify import SMPLify3D
SRC="/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets/datasets/_humanact12_h3d/extracted/humanact12/humanact12"
dev=torch.device("cuda:0")
mean=h5py.File(config.SMPL_MEAN_FILE,"r")
ip=torch.from_numpy(mean["pose"][:]).float().to(dev).reshape(1,-1); ish=torch.from_numpy(mean["shape"][:]).float().to(dev).reshape(1,-1)
conf=torch.ones(22).to(dev)
def ground(j):
    off=np.array([j[0,0,0],j[...,1].min(),j[0,0,2]],dtype=np.float32); return j-off[None,None,:]
def mpjpe(smpl,poses,trans,j,B):
    F=len(j); err=[]
    for c0 in range(0,F,B):
        n=min(B,F-c0); gp=torch.tensor(poses[c0:c0+n]).to(dev); tt=torch.tensor(trans[c0:c0+n]).to(dev)
        gp=torch.cat([gp,gp[-1:].repeat(B-n,1)],0); tt=torch.cat([tt,tt[-1:].repeat(B-n,1)],0)
        with torch.no_grad(): o=smpl(betas=torch.zeros(B,10).to(dev),global_orient=gp[:,:3],body_pose=gp[:,3:],transl=tt,return_verts=False)
        err.append(np.linalg.norm(o.joints[:n,:22,:].cpu().numpy()-j[c0:c0+n],axis=-1))
    return float(np.concatenate([e.reshape(-1) for e in err]).mean())*100

def batched(j,niters,B=128):
    smpl=smplx.create(config.SMPL_MODEL_DIR,model_type="smpl",gender="neutral",ext="pkl",batch_size=B).to(dev)
    sm=SMPLify3D(smplxmodel=smpl,batch_size=B,joints_category="AMASS",num_iters=niters,device=dev)
    F=len(j); poses=np.zeros((F,72),np.float32); trans=np.zeros((F,3),np.float32)
    ipB=ip.repeat(B,1); ishB=ish.repeat(B,1)
    for c0 in range(0,F,B):
        ch=j[c0:c0+B]; n=len(ch); pad=np.concatenate([ch,np.repeat(ch[-1:],B-n,0)],0) if n<B else ch
        kp=torch.tensor(pad).to(dev)
        _,_,np_,_,nc,_=sm(ipB.detach(),ishB.detach(),torch.zeros(B,3).to(dev),kp,conf_3d=conf,seq_ind=0)
        poses[c0:c0+n]=np_.detach().cpu().numpy()[:n]; trans[c0:c0+n]=nc.detach().cpu().numpy().reshape(-1,3)[:n]
    return poses,trans,smpl
def warmstart(j,niters):
    smpl=smplx.create(config.SMPL_MODEL_DIR,model_type="smpl",gender="neutral",ext="pkl",batch_size=1).to(dev)
    sm=SMPLify3D(smplxmodel=smpl,batch_size=1,joints_category="AMASS",num_iters=niters,device=dev)
    F=len(j); poses=np.zeros((F,72),np.float32); trans=np.zeros((F,3),np.float32)
    pp=ip.clone(); pb=ish.clone(); pc=torch.zeros(1,3).to(dev); kp=torch.zeros(1,22,3).to(dev)
    for f in range(F):
        kp[0]=torch.tensor(j[f]).to(dev)
        _,_,npose,nbeta,nc,_=sm(pp.detach(),pb.detach(),pc.detach(),kp,conf_3d=conf,seq_ind=f)
        pp,pb,pc=npose,nbeta,nc; poses[f]=npose.detach().cpu().numpy()[0]; trans[f]=nc.detach().cpu().numpy().reshape(-1,3)[0]
    return poses,trans,smpl
smplB=smplx.create(config.SMPL_MODEL_DIR,model_type="smpl",gender="neutral",ext="pkl",batch_size=128).to(dev)
for name in sys.argv[1:]:
    j=ground(np.load(f"{SRC}/{name}.npy").astype(np.float32)[:,:22,:]); F=len(j)
    print(f"=== {name} F={F} ===",flush=True)
    for label,fn,args in [("batched@150",batched,(150,)),("batched@300",batched,(300,)),("warmstart@150",warmstart,(150,))]:
        t=time.time(); p,tr,_=fn(j,*args); dt=time.time()-t
        print(f"  {label:16s} MPJPE={mpjpe(smplB,p,tr,j,128):.2f}cm  {dt:.1f}s",flush=True)
