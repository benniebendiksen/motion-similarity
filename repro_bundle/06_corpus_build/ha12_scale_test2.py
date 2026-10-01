#!/usr/bin/env python3
"""Compare scale-normalization methods vs SMPL rest bone lengths:
 A) single pelvis(0)->neck(12) -> 0.51
 B) multi-bone MEDIAN of per-bone ratios (len_smpl/len_clip)
 C) multi-bone LEAST-SQUARES scalar s = sum(lc*ls)/sum(lc*lc)
Report fitted MPJPE (normalized space) + scale. All target ~SMPL size so cm are comparable."""
import os,sys,numpy as np,torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from visualize.joints2smpl.src import config
config.SMPL_MODEL_DIR=os.path.join(HERE,"smpl_models")+"/"; config.GMM_MODEL_DIR=os.path.join(HERE,"smpl_models")+"/"
config.SMPL_MEAN_FILE=os.path.join(HERE,"smpl_models","neutral_smpl_mean_params.h5")
import smplx,h5py
from visualize.joints2smpl.src.smplify import SMPLify3D
SRC="/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets/datasets/_humanact12_h3d/extracted/humanact12/humanact12"
dev=torch.device("cuda:0"); B=128
PAR=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
mean=h5py.File(config.SMPL_MEAN_FILE,"r")
ip=torch.from_numpy(mean["pose"][:]).float().to(dev).reshape(1,-1).repeat(B,1); ish=torch.from_numpy(mean["shape"][:]).float().to(dev).reshape(1,-1).repeat(B,1)
conf=torch.ones(22).to(dev)
smpl=smplx.create(config.SMPL_MODEL_DIR,model_type="smpl",gender="neutral",ext="pkl",batch_size=B).to(dev)
sm=SMPLify3D(smplxmodel=smpl,batch_size=B,joints_category="AMASS",num_iters=150,device=dev)
# SMPL rest joints (zero pose/betas)
with torch.no_grad():
    o=smpl(betas=torch.zeros(B,10).to(dev),global_orient=torch.zeros(B,3).to(dev),body_pose=torch.zeros(B,69).to(dev),transl=torch.zeros(B,3).to(dev),return_verts=False)
Jr=o.joints[0,:22,:].cpu().numpy()
Lsmpl=np.array([np.linalg.norm(Jr[c]-Jr[PAR[c]]) for c in range(1,22)])
def bonelens(j): return np.array([np.linalg.norm(j[:,c]-j[:,PAR[c]],axis=-1).mean() for c in range(1,22)])
def ground(j):
    off=np.array([j[0,0,0],j[...,1].min(),j[0,0,2]],dtype=np.float32); return j-off[None,None,:]
def fit(j):
    F=len(j); poses=np.zeros((F,72),np.float32); trans=np.zeros((F,3),np.float32)
    for c0 in range(0,F,B):
        ch=j[c0:c0+B]; n=len(ch); pad=np.concatenate([ch,np.repeat(ch[-1:],B-n,0)],0) if n<B else ch
        _,_,p,_,c,_=sm(ip.detach(),ish.detach(),torch.zeros(B,3).to(dev),torch.tensor(pad).to(dev),conf_3d=conf,seq_ind=0)
        poses[c0:c0+n]=p.detach().cpu().numpy()[:n]; trans[c0:c0+n]=c.detach().cpu().numpy().reshape(-1,3)[:n]
    err=[]
    for c0 in range(0,F,B):
        n=min(B,F-c0); gp=torch.tensor(poses[c0:c0+n]).to(dev); tt=torch.tensor(trans[c0:c0+n]).to(dev)
        gp=torch.cat([gp,gp[-1:].repeat(B-n,1)],0); tt=torch.cat([tt,tt[-1:].repeat(B-n,1)],0)
        with torch.no_grad(): oo=smpl(betas=torch.zeros(B,10).to(dev),global_orient=gp[:,:3],body_pose=gp[:,3:],transl=tt,return_verts=False)
        err.append(np.linalg.norm(oo.joints[:n,:22,:].cpu().numpy()-j[c0:c0+n],axis=-1))
    return float(np.concatenate([e.reshape(-1) for e in err]).mean())*100
for nm in ["P06G02R01F1742T1779A0901","P01G02R01F1140T1240A0801","P01G01R01F0001T0064A0101"]:
    j0=np.load(f"{SRC}/{nm}.npy").astype(np.float32)[:,:22,:]
    lc=bonelens(j0)
    sA=0.51/np.linalg.norm(j0[:,12]-j0[:,0],axis=-1).mean()
    sB=float(np.median(Lsmpl/lc))
    sC=float(np.sum(lc*Lsmpl)/np.sum(lc*lc))
    print(f"=== {nm} F={len(j0)} | scales A={sA:.3f} B={sB:.3f} C={sC:.3f} ===",flush=True)
    print(f"  A single    MPJPE={fit(ground(j0*sA)):.2f}cm",flush=True)
    print(f"  B median    MPJPE={fit(ground(j0*sB)):.2f}cm",flush=True)
    print(f"  C lsq       MPJPE={fit(ground(j0*sC)):.2f}cm",flush=True)
