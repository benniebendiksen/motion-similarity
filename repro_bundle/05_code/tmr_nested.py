#!/usr/bin/env python3
"""Evaluate TMR (learned-similarity baseline) on the IDENTICAL 25 nested-CV folds as the
encoders and geometric baselines. TMR embedding L2 vs d_perc, Spearman, fold-averaged the
same way (baselines_nested.py structure). Conservative: TMR's input came via our BVH->SMPL
fit (~4cm), so a loss vs our encoder is a lower bound on our advantage."""
import sys, os, glob, numpy as np
from scipy import stats
MS="/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/motion-similarity"
TR="/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets"
sys.path.insert(0, MS)
os.environ.setdefault("MOTION_IS_REMOTE","1"); os.environ.setdefault("PYTHONWARNINGS","ignore")
from Config import Config
import cv_preliminary as cv

TMR_DIR=os.path.join(TR,"learned_baselines","tmr","encoded")  # Action_e0_e1_e2_e3.npy [256]
ACTIONS=["walking","pointing","picking"]
CAP={"walking":"Walking","pointing":"Pointing","picking":"Picking"}
BARS={"walking":0.478,"pointing":0.370,"picking":0.431}

def tmr_emb(action, key):
    fn=CAP[action]+"_"+"_".join(map(str,key))+".npy"
    p=os.path.join(TMR_DIR,fn)
    return np.load(p).astype(np.float64) if os.path.exists(p) else None

def spearman_on_keys(action, keys, module):
    """TMR-embedding L2 vs d_perc over pairs whose both keys are in `keys` (incl neutral)."""
    embs={}
    for k in keys:
        e=tmr_emb(action,k)
        if e is not None: embs[k]=e
    kl=sorted(embs); d=[]; p=[]
    for i in range(len(kl)):
        for j in range(i+1,len(kl)):
            dp=cv.get_inverse_direct_comparison_value((action,kl[i]),(action,kl[j]),module)
            if dp is None: continue
            d.append(float(np.linalg.norm(embs[kl[i]]-embs[kl[j]]))); p.append(dp)
    return stats.spearmanr(d,p).correlation if len(d)>3 else float("nan")

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--k",type=int,default=5); ap.add_argument("--repeats",type=int,default=5); ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args()
    cfg=Config(); cv.ACTIONS=ACTIONS
    modules,all_keys={},{}
    for act in ACTIONS: modules[act],all_keys[act]=cv.load_action_keys(act,cfg)
    per_action={act:[] for act in ACTIONS}
    for rep in range(a.repeats):
        folds={act:cv.make_folds(all_keys[act],a.k,a.seed+1000*rep) for act in ACTIONS}
        for f in range(a.k):
            for act in ACTIONS:
                test=folds[act][f]|{(0,0,0,0)}        # TEST fold (+neutral), IDENTICAL to encoder/baseline eval
                s=spearman_on_keys(act,test,modules[act])
                if s==s: per_action[act].append(s)
    print("=== TMR (learned-similarity baseline), raw embedding L2 vs d_perc, n=25 matched folds ===")
    print(f"{'action':<10} {'mean':>8} {'std':>7} {'n':>4}  clears bar?")
    for act in ACTIONS:
        v=np.array(per_action[act]); m=v.mean(); sd=v.std()
        sem=sd/np.sqrt(len(v)); clears=(m-1.96*sem)>BARS[act]
        print(f"{act:<10} {m:>8.3f} {sd:>7.3f} {len(v):>4}  bar={BARS[act]} {'YES' if clears else 'no'} (95%CI LB={m-1.96*sem:.3f})")
if __name__=="__main__": main()
