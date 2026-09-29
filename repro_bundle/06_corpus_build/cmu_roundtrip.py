#!/usr/bin/env python3
"""CMU round-trip fidelity gate: convert mb-source 80_63 through the exact cmu_all_perform pipeline
(prepare_files: drop Neck/RThumb/LThumb + adjustFrameRate(30) + fix_end_sites) and require it to reproduce
the existing cmu_all_perform/80_63.bvh. If it matches, the pipeline is proven faithful for the 248."""
import sys, os, shutil, numpy as np
TR = "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets"
sys.path.insert(0, TR)
import bvhConverterToPerform as C

src = TR + "/datasets/_cmu_mb_raw/extracted/80/80_63.bvh"
tmp = TR + "/datasets/_cmu_rt_src"; out = TR + "/datasets/_cmu_rt_out"
for d in (tmp, out):
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d, exist_ok=True)
shutil.copy(src, tmp + "/80_63.bvh")

C.prepare_files(tmp, out)
try:
    C.fix_end_sites(out)
except Exception as e:
    print("fix_end_sites note:", str(e)[:80])

def motion(path):
    lines = open(path).read().splitlines()
    i = [k for k, l in enumerate(lines) if l.strip().lower().startswith("frame time")][0]
    return np.array([list(map(float, l.split())) for l in lines[i + 1:] if l.strip()])

def joints(path):
    return [l.split()[1] for l in open(path).read().splitlines()
            if l.strip().startswith(("ROOT", "JOINT"))]

conv = out + "/80_63.bvh"; ref = TR + "/datasets/cmu_all_perform/80_63.bvh"
jc, jr = joints(conv), joints(ref)
print("converted joints:", len(jc), "| ref joints:", len(jr), "| joint names match:", jc == jr)
if jc != jr:
    print("  conv:", jc); print("  ref :", jr)
a, b = motion(conv), motion(ref)
print("converted motion shape:", a.shape, "| ref shape:", b.shape)
if a.shape == b.shape:
    d = np.abs(a - b)
    print("MAX abs diff: %.6g | MEAN abs diff: %.6g" % (d.max(), d.mean()))
    print("VERDICT:", "MATCH (pipeline faithful)" if d.max() < 1e-3 else "MISMATCH - investigate")
else:
    print("VERDICT: SHAPE MISMATCH - investigate (frames/cols)")
