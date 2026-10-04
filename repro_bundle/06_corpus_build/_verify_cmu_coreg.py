import json,glob,os
TR="/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets"
def fr(p):
    for l in open(p):
        if l.strip().startswith("Frames:"): return int(l.split()[-1])
d=json.load(open(TR+"/learned_baselines/tmr/repo/datasets/annotations/humanml3d/annotations.json"))
ann={v["path"]:v["duration"] for v in d.values() if not v["path"].startswith("M/")}
testB=glob.glob(TR+"/datasets/_amass_cmu_testB/**/*.bvh",recursive=True)
print("retargeted CMU/80 clips:",len(testB))
ok=0; tot=0; worse=[]
for f in testB:
    clip=os.path.basename(f).split("_smpl24")[0].split("_cmu33")[0]  # e.g. 80_63
    p=f"CMU/80/{clip}_poses"
    if p not in ann: continue
    tot+=1; new=fr(f)/30.0; a=ann[p]
    cgp=f"{TR}/datasets/cmu_all_perform/{clip}.bvh"
    cg=fr(cgp)/30.0 if os.path.exists(cgp) else None
    if abs(new-a)<=0.1: ok+=1
    if clip=="80_63":
        print(f"  [80_63] annotation={a:.3f}s  AMASS-CMU(new)={new:.3f}s (diff {abs(new-a):.3f})  cgspeed(old)={cg:.3f}s (diff {abs(cg-a):.3f})")
print(f"co-register (<=0.1s): {ok}/{tot} = {100*ok/tot:.0f}%  (cgspeed CMU was 74%)")
