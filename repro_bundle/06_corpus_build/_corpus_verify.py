import json,os,sys,numpy as np
TR="/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets"; sys.path.insert(0,TR)
from common.MotionDataset import extract_root_and_rotations
from bvhReader.bvh import BVH
D=TR+"/datasets/hml3d_cmu_flat"
man=json.load(open(TR+"/datasets/hml3d_captioned.json"))
def route(p):
    t=p.split("/")[0]
    return "R2_CMU" if t=="CMU" else ("R3_HA12" if t=="humanact12" else "R1_AMASS")
acc={r:{"s":np.zeros((34,6)),"sq":np.zeros((34,6)),"n":0,"clips":0} for r in ["R1_AMASS","R2_CMU","R3_HA12"]}
nan=[]; empty_cap=0; vmin=1e9; vmax=-1e9; refj=None; hmis=[]; missing=0; jcounts=set()
for stem,e in man.items():
    f=D+"/"+stem+"_cmu33_30fps.bvh"
    if not os.path.exists(f): missing+=1; continue
    if not e.get("captions"): empty_cap+=1
    b=BVH(); b.load(f)
    jn=tuple(j.name for j in b.joints); jcounts.add(len(jn))
    if refj is None: refj=jn
    elif jn!=refj: hmis.append(stem)
    feats=extract_root_and_rotations(b).astype(np.float64)
    if not np.isfinite(feats).all(): nan.append(stem); continue
    vmin=min(vmin,float(feats.min())); vmax=max(vmax,float(feats.max()))
    a=acc[route(e["path"])]; a["s"]+=feats.sum(0); a["sq"]+=(feats**2).sum(0); a["n"]+=feats.shape[0]; a["clips"]+=1
print("=== FORMAT IDENTITY ===")
print("joint-count set:",jcounts," hierarchy mismatches vs ref:",len(hmis),hmis[:3])
print("missing files:",missing," empty-caption clips:",empty_cap)
print("=== MOTION VALUE SANITY (6D rot) ===")
print("global min=%.3f max=%.3f  NaN/Inf clips=%d %s"%(vmin,vmax,len(nan),nan[:3]))
print("=== PER-ROUTE DEGENERACY (std<0.05) ===")
for r,a in acc.items():
    if a["n"]==0: continue
    m=a["s"]/a["n"]; var=a["sq"]/a["n"]-m**2; std=np.sqrt(np.clip(var,0,None))
    print("  %-8s clips=%5d frames=%8d degenerate=%3d/204  std[%.3f,%.3f]"%(r,a["clips"],a["n"],int((std<0.05).sum()),std.min(),std.max()))
