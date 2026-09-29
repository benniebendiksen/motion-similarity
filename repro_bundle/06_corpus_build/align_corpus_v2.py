#!/usr/bin/env python3
"""Corpus coverage v2: AMASS non-CMU -> amass_cmu_flat ; CMU -> cmu_all_perform (raw route).
Reports exact coverage per route and writes the list of MISSING CMU source ids to fetch from the free CMU DB."""
import json, os, re
TR = "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets"
ann = json.load(open(TR + "/learned_baselines/tmr/repo/datasets/annotations/humanml3d/annotations.json"))
amass_flat = set(f[:-4] for f in os.listdir(TR + "/datasets/amass_cmu_flat") if f.endswith(".bvh"))
cmu_flat = set(f[:-4] for f in os.listdir(TR + "/datasets/cmu_all_perform") if f.endswith(".bvh"))

def amass_key(p): return re.sub(r"_(poses|stageii|stage_ii)$", "", p).replace("/", "_")
def amass_filekey(s): return re.sub(r"_cmu(33)?_30fps$", "", re.sub(r"_30fps$", "", s))
amass_fk = set(amass_filekey(s) for s in amass_flat)
def cmu_id(p): return re.sub(r"_poses$", "", p.split("/")[-1])  # CMU/80/80_63_poses -> 80_63

amass_hit = amass_miss = cmu_hit = ha = 0
amass_missing, cmu_missing = [], set()
for k, v in ann.items():
    if k.startswith("M"): continue          # skip mirrors (augmentation)
    p = v["path"]; src = p.split("/")[0]
    if src.lower() == "humanact12": ha += 1; continue
    if src == "CMU":
        cid = cmu_id(p)
        if cid in cmu_flat: cmu_hit += 1
        else: cmu_missing.add(cid)
    else:
        if amass_key(p) in amass_fk: amass_hit += 1
        else: amass_miss += 1; amass_missing.append(p)

print("AMASS non-CMU:  hit %d  miss %d  (%.2f%%)" % (amass_hit, amass_miss, 100*amass_hit/max(1,amass_hit+amass_miss)))
if amass_missing: print("  amass miss sample:", amass_missing[:6])
print("CMU (raw route via cmu_all_perform):  hit %d  |  MISSING unique source ids: %d" % (cmu_hit, len(cmu_missing)))
print("  missing CMU ids sample:", sorted(cmu_missing)[:15])
print("HumanAct12 entries (downloading separately):", ha)
tot = amass_hit + cmu_hit
print("=> currently-covered non-mirror clips: %d  (of 14614 = %.1f%%); to-fetch: CMU %d ids + HumanAct12 %d"
      % (tot, 100*tot/14614, len(cmu_missing), ha))
open(TR + "/datasets/_cmu_missing_ids.txt", "w").write("\n".join(sorted(cmu_missing)) + "\n")
print("wrote datasets/_cmu_missing_ids.txt (%d ids)" % len(cmu_missing))
