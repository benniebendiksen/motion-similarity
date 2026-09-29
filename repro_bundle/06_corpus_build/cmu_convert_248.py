#!/usr/bin/env python3
"""Convert the 248 missing CMU clips through the round-trip-PROVEN pipeline (drop Neck/RThumb/LThumb ->
adjustFrameRate(30) -> fix_end_sites). Output verified against the cmu_all_perform format (28 joints, 30fps)."""
import sys, os, shutil, glob
TR = "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets"
sys.path.insert(0, TR)
import bvhconv_lib as C

miss = [l.strip() for l in open(TR + "/datasets/_cmu_missing_ids.txt") if l.strip()]
srcmap = {os.path.basename(f)[:-4]: f for f in glob.glob(TR + "/datasets/_cmu_mb_raw/extracted/**/*.bvh", recursive=True)}
tmp = TR + "/datasets/_cmu_248_src"; out = TR + "/datasets/_cmu_248_out"
for d in (tmp, out):
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d, exist_ok=True)

missing_src = []
for m in miss:
    if m in srcmap:
        shutil.copy(srcmap[m], tmp + "/" + m + ".bvh")
    else:
        missing_src.append(m)
print("to convert:", len(miss) - len(missing_src), "of", len(miss),
      ("| NO SOURCE for: " + str(missing_src[:8])) if missing_src else "| all sources found")

C.prepare_files(tmp, out)
try:
    C.fix_end_sites(out)
except Exception as e:
    print("fix_end_sites note:", str(e)[:80])

# verify each output: 28 joints, 568-style 30fps, parses
outfiles = sorted(glob.glob(out + "/*.bvh"))
bad = []
for f in outfiles:
    lines = open(f).read().splitlines()
    nj = sum(1 for l in lines if l.strip().startswith(("ROOT", "JOINT")))
    ft = [l for l in lines if l.strip().lower().startswith("frame time")]
    ok = (nj == 28 and ft and abs(float(ft[0].split()[-1]) - 0.033333) < 1e-4)
    if not ok:
        bad.append((os.path.basename(f), nj, ft[0].split()[-1] if ft else "no-ft"))
print("CONVERTED:", len(outfiles), "| joints=28 & 30fps verified:", len(outfiles) - len(bad), "| bad:", len(bad))
if bad:
    print("  bad sample:", bad[:5])
print("VERDICT:", "ALL 248 OK" if (len(outfiles) == 248 and not bad and not missing_src) else "CHECK ABOVE")
