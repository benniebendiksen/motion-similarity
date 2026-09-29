#!/usr/bin/env python3
"""HumanAct12 source-correctness gate — find the convention HumanML3D used (the code warns HA12 is L/R flipped).
Try 4 variants of the input joints, compute guofeats, compare to new_joint_vecs/000001.npy; the ~0-diff one is it."""
import sys, os, numpy as np
TR = "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets"
REPO = TR + "/learned_baselines/tmr/repo"
sys.path.insert(0, REPO)
from src.guofeats import joints_to_guofeats

def swap_left_right(data):
    data = data.copy(); data[..., 0] *= -1
    rc = [2, 5, 8, 11, 14, 17, 19, 21]; lc = [1, 4, 7, 10, 13, 16, 18, 20]
    tmp = data[:, rc].copy(); data[:, rc] = data[:, lc]; data[:, lc] = tmp
    return data

clip = TR + "/datasets/_humanact12_raw/HumanAct12/P11G01R02F1812T1847A0402.npy"
ref = "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/MotionDiffuse/MotionDiffuseDCT/text2motion/data/HumanML3D/new_joint_vecs/000001.npy"
j = np.load(clip).astype(np.float32)
r = np.load(ref)

def xinv(a): a = a.copy(); a[..., 0] *= -1; return a
variants = {
    "as-is": j,
    "x-inverted": xinv(j),
    "LR-swap(mirror)": swap_left_right(j),
    "x-inv + LR-swap": swap_left_right(xinv(j)),
}
print("ref shape:", r.shape)
best = None
for name, jj in variants.items():
    try:
        f = np.asarray(joints_to_guofeats(jj))
    except Exception as e:
        print("%-18s ERROR %s" % (name, str(e)[:50])); continue
    if f.shape != r.shape:
        print("%-18s shape %s != ref" % (name, f.shape)); continue
    md = float(np.abs(f - r).max())
    print("%-18s max_abs_diff = %.6g" % (name, md))
    if best is None or md < best[1]:
        best = (name, md)
if best:
    print("BEST:", best[0], "(max diff %.6g)" % best[1],
          "->", "MATCH" if best[1] < 1e-2 else "still off - investigate further")
