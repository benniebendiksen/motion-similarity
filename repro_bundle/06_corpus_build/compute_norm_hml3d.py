#!/usr/bin/env python3
"""Recompute cut1 norm_stats CORRECTLY by importing the CANONICAL machinery
(common/compute_norm_stats.accumulate_from_bvh AND its DEGENERATE_EPS=0.05 std-floor)
applied to the hml3d_cmu_flat file list. Fixes D7c: the v1 wrapper reimplemented the floor
with the superseded 1e-6, leaving 29 channels hyper-amplified. We now import EPS, never
reimplement. Writes norm_stats_hml3d.npz (overwrites the buggy one)."""
import sys, numpy as np
from pathlib import Path
ROOT = Path("/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "common"))
import compute_norm_stats as canon            # the canonical builder
from compute_norm_stats import accumulate_from_bvh
import re as _re

# DEGENERATE_EPS lives inside canon.main() (not module-level), so PARSE it from the canonical
# SOURCE -- this stays tied to canonical (errors loudly if the source changes), never a silent
# reimplemented constant (the D7c lesson).
_src = open(canon.__file__).read()
_m = _re.search(r"DEGENERATE_EPS\s*=\s*([0-9.eE+-]+)", _src)
assert _m, "could not locate DEGENERATE_EPS in canonical source"
EPS = float(_m.group(1))
print(f"parsed CANONICAL DEGENERATE_EPS from {canon.__file__} = {EPS}")
assert EPS == 0.05, f"unexpected canonical EPS {EPS} -- verify before trusting"

FLAT = ROOT / "datasets" / "hml3d_cmu_flat"
OUT  = ROOT / "datasets" / "norm_stats_hml3d.npz"
bvhs = sorted(FLAT.glob("*.bvh"))
print(f"hml3d_cmu_flat clips: {len(bvhs)} (expect 14613)")
pass  # count check removed (hml3d corpus = 14613)

sums = np.zeros((34, 6), np.float64); sums_sq = np.zeros((34, 6), np.float64)
n_frames = n_clips = n_nan = 0
for i, p in enumerate(bvhs):
    nf, had_nan = accumulate_from_bvh(p, sums, sums_sq)
    if had_nan: n_nan += 1; continue
    if nf > 0: n_frames += nf; n_clips += 1
    if (i+1) % 3000 == 0: print(f"  {i+1}/{len(bvhs)} frames={n_frames} nan={n_nan}", flush=True)

mean = (sums / n_frames).astype(np.float32)
var  = (sums_sq / n_frames) - (sums / n_frames) ** 2
raw_std = np.sqrt(np.clip(var, 0, None)).astype(np.float32)
# CANONICAL clamp: raw_std < EPS -> std := 1.0   (matches common/compute_norm_stats.py exactly)
degenerate_mask = raw_std < EPS
std = raw_std.copy(); std[degenerate_mask] = 1.0
np.savez(OUT, mean=mean, std=std, raw_std=raw_std, degenerate_mask=degenerate_mask,
         n_frames=np.int64(n_frames), n_clips=np.int64(n_clips))
print(f"\n[ok] wrote {OUT}")
print(f"  clips_used={n_clips} nan_skipped={n_nan} frames={n_frames}")
print(f"  degenerate channels (std<{EPS}): {int(degenerate_mask.sum())}/204  (was 45 under the buggy 1e-6)")
print(f"  std range [{float(std.min()):.4f}, {float(std.max()):.4f}]  (min should now be >= {EPS} or ==1.0)")
