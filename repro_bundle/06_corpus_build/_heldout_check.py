import glob,os
TR="/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets"
corpus=set(os.path.basename(f).replace("_cmu33_30fps.bvh","") for f in glob.glob(TR+"/datasets/hml3d_cmu_flat/*.bvh"))
evdirs=[d for d in glob.glob(TR+"/datasets/*") if "lma" in os.path.basename(d).lower() and "perform" in os.path.basename(d).lower()]
print("eval dirs:",[os.path.basename(d) for d in evdirs])
ev=set()
for d in evdirs:
    for f in glob.glob(d+"/**/*.bvh",recursive=True):
        ev.add(os.path.basename(f).replace("_cmu33_30fps.bvh","").replace(".bvh",""))
print("corpus clips=%d  eval clips=%d"%(len(corpus),len(ev)))
inter=corpus & ev
print("OVERLAP:",len(inter),list(inter)[:5])
toks=("perform","effort","walk_","point_","pick_")
hits=[c for c in corpus if any(t in c.lower() for t in toks)]
print("corpus stems w/ study-ish token:",len(hits),hits[:5])
