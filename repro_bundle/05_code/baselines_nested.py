#!/usr/bin/env python
"""
Re-evaluate the DTW and (DTW-corrected, unpadded) geodesic baselines on the EXACT SAME
nested-CV TEST folds as the MAMP+pose encoder, fold-averaged identically. Apples-to-apples:
the encoder mean was a fold-average over these folds, so the baselines must be too.

Per fold: restrict raw motion features to the fold's TEST keys, compute pairwise DTW and
geodesic distances, pair with d_perc = 1 - count_normalized[(0,2)], Spearman. Average over
all repeats*K folds. Reports baseline mean +/- std (+ SEM) on the identical partitions used
for the encoder, so "encoder vs baseline" is a like-for-like comparison.

Uses the LOCAL corrected geodesic compute_geodesic_distances_dtw (DTW-aligned, raw unpadded,
num_joints=28) -- NOT the legacy padded one. See memory project_geodesic_baseline_provenance.
"""
import os, sys, argparse, time, json, random
import numpy as np
from scipy import stats

MS_ROOT = "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/motion-similarity"
# pipelines/ FIRST so the corrected-geodesic copy wins over the stale root infer_emb_vec.py
sys.path.insert(0, MS_ROOT)
sys.path.insert(0, os.path.join(MS_ROOT, "pipelines"))
os.chdir(MS_ROOT)

from Config import Config
import cv_preliminary as cv
# import the PIPELINES copy explicitly (has corrected DTW-geodesic; root copy is stale)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("infer_emb_vec_pipe",
                                     os.path.join(MS_ROOT, "pipelines", "infer_emb_vec.py"))
_iev = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_iev)
calculate_real_variable_length_dtw = _iev.calculate_real_variable_length_dtw
compute_geodesic_distances_dtw = _iev.compute_geodesic_distances_dtw
get_raw_features_without_dataloader = _iev.get_raw_features_without_dataloader

ACTIONS = ["walking", "pointing", "picking"]
BARS = {"walking": 0.478, "pointing": 0.370, "picking": 0.431}


def spearman_from_distances(dist_tuples, action, module):
    """dist_tuples: list of (dist, key1, key2). Pair with d_perc, Spearman over GT pairs."""
    d, p = [], []
    for dist, k1, k2 in dist_tuples:
        # keys may be (action, effort) or bare effort; normalize to bare effort for lookup
        e1 = k1[1] if isinstance(k1, tuple) and len(k1) == 2 and isinstance(k1[0], str) else k1
        e2 = k2[1] if isinstance(k2, tuple) and len(k2) == 2 and isinstance(k2[0], str) else k2
        v = cv.get_inverse_direct_comparison_value((action, e1), (action, e2), module)
        if v is not None:
            d.append(float(dist)); p.append(v)
    if len(d) < 4:
        return float("nan")
    return stats.spearmanr(d, p).correlation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(MS_ROOT, "baselines_nested_results.json"))
    args = ap.parse_args()
    cv.ACTIONS = ACTIONS
    def ts(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    cfg = Config()
    modules, all_keys = {}, {}
    for a in ACTIONS:
        modules[a], all_keys[a] = cv.load_action_keys(a, cfg)

    # preload full raw features once per action (we subset per fold via key filtering)
    raw_full = {}
    for a in ACTIONS:
        raw_full[a] = get_raw_features_without_dataloader(a, cfg, valid_indices=None,
                                                          balance_class_frame_counts=False)
        # normalize keys to bare effort tuples
        rb = {}
        for k, v in raw_full[a].items():
            e = k[1] if isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], str) else k
            rb[e] = v
        raw_full[a] = rb
        ts(f"  {a}: {len(raw_full[a])} raw clips loaded")

    rows = {"dtw": [], "geo": []}
    for rep in range(args.repeats):
        folds = {a: cv.make_folds(all_keys[a], args.k, args.seed + 1000 * rep) for a in ACTIONS}
        for f in range(args.k):
            for a in ACTIONS:
                test = folds[a][f]   # TEST keys for this fold (bare effort tuples)
                sub = {e: raw_full[a][e] for e in test if e in raw_full[a] and e != (0, 0, 0, 0)}
                if len(sub) < 3:
                    continue
                dtw_d = calculate_real_variable_length_dtw(sub)
                geo_d = compute_geodesic_distances_dtw(sub, num_joints=28)
                s_dtw = spearman_from_distances(dtw_d, a, modules[a])
                s_geo = spearman_from_distances(geo_d, a, modules[a])
                rows["dtw"].append((a, s_dtw)); rows["geo"].append((a, s_geo))
            ts(f"[r{rep}f{f}] done")
        json.dump(rows, open(args.out, "w"), default=list)

    ts("===== BASELINES on identical nested-CV folds (fold-mean +/- std, SEM) =====")
    agg = {}
    for name in ("dtw", "geo"):
        agg[name] = {}
        for a in ACTIONS:
            vals = [s for (act, s) in rows[name] if act == a and s == s]
            m, sd = float(np.mean(vals)), float(np.std(vals))
            sem = sd / np.sqrt(len(vals)) if vals else float("nan")
            agg[name][a] = {"mean": round(m, 4), "std": round(sd, 4), "sem": round(sem, 4), "n": len(vals)}
            ts(f"  {name.upper():4} {a:9} {m:.3f} +/- {sd:.3f} (SEM {sem:.3f}, n={len(vals)})")
    # encoder (from the repeated run) for side-by-side reference -- filled in by user/report
    json.dump({"rows": rows, "aggregate": agg, "bars": BARS}, open(args.out, "w"), default=list)
    ts("done")


if __name__ == "__main__":
    main()
