#!/usr/bin/env python3
"""Test: does per-clip scale-normalization to SMPL fix the high-MPJPE clips? Scale joints so
pelvis(0)->neck(12) == 0.51m (SMPL reference), batched@150 fit, report MPJPE (in normalized space)."""
import os,sys,glob,time,numpy as np,torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from visualize.joints2smpl.src import config
config.SMPL_MODEL_DIR=os.path.join(HERE,"smpl_models")+"/"; config.GMM_MODEL_DIR=os.path.join(HERE,"smpl_models")+"/"
config.SMPL_MEAN_FILE=os.path.join(HERE,"smpl_models","neutral_smpl_mean_params.h5")
import smplx,h5py
from visualize.joints2smpl.src.smplify import SMPLify3D
SRC="/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets/datasets/_humanact12_h3d/extracted/humanact12/humanact12"
dev=torch.device("cuda:0"); B=128
mean=h5py.File(config.SMPL_MEAN_FILE,"r")
ip=torch.from_numpy(mean["pose"][:]).float().to(dev).reshape(1,-1).repeat(B,1); ish=torch.from_numpy(mean["shape"][:]).float().to(dev).reshape(1,-1).repeat(B,1)
conf=torch.ones(22).to(dev)
smpl=smplx.create(config.SMPL_MODEL_DIR,model_type="smpl",gender="neutral",ext="pkl",batch_size=B).to(dev)
sm=SMPLify3D(smplxmodel=smpl,batch_size=B,joints_category="AMASS",num_iters=150,device=dev)
TARGET=0.51
def ground(j):
    off=np.array([j[0,0,0],j[...,1].min(),j[0,0,2]],dtype=np.float32); return j-off[None,None,:]
def fit(j):
    F=len(j); poses=np.zeros((F,72),np.float32); trans=np.zeros((F,3),np.float32)
    for c0 in range(0,F,B):
        ch=j[c0:c0+B]; n=len(ch); pad=np.concatenate([ch,np.repeat(ch[-1:],B-n,0)],0) if n<B else ch
        kp=torch.tensor(pad).to(dev)
        _,_,p,_,c,_=sm(ip.detach(),ish.detach(),torch.zeros(B,3).to(dev),kp,conf_3d=conf,seq_ind=0)
        poses[c0:c0+n]=p.detach().cpu().numpy()[:n]; trans[c0:c0+n]=c.detach().cpu().numpy().reshape(-1,3)[:n]
    err=[]
    for c0 in range(0,F,B):
        n=min(B,F-c0); gp=torch.tensor(poses[c0:c0+n]).to(dev); tt=torch.tensor(trans[c0:c0+n]).to(dev)
        gp=torch.cat([gp,gp[-1:].repeat(B-n,1)],0); tt=torch.cat([tt,tt[-1:].repeat(B-n,1)],0)
        with torch.no_grad(): o=smpl(betas=torch.zeros(B,10).to(dev),global_orient=gp[:,:3],body_pose=gp[:,3:],transl=tt,return_verts=False)
        err.append(np.linalg.norm(o.joints[:n,:22,:].cpu().numpy()-j[c0:c0+n],axis=-1))
    return float(np.concatenate([e.reshape(-1) for e in err]).mean())*100
for nm in ["P06G02R01F1742T1779A0901","P01G02R01F1140T1240A0801","P01G01R01F0001T0064A0101"]:
    j0=np.load(f"{SRC}/{nm}.npy").astype(np.float32)[:,:22,:]
    sp=np.linalg.norm(j0[:,12]-j0[:,0],axis=-1).mean(); s=TARGET/sp
    print(f"{nm}: pelvis-neck={sp:.3f} scale={s:.3f}",flush=True)
    print(f"  raw-scale   MPJPE={fit(ground(j0)):.2f}cm",flush=True)
    print(f"  norm-scale  MPJPE={fit(ground(j0*s)):.2f}cm",flush=True)
