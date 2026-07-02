#!/usr/bin/env python
"""
Leakage-clean nested cross-validation for the checkpoint-fairness re-analysis.

3-WAY SPLIT per outer fold (user-specified, no selection leakage):
  * TEST   = outer held-out fold motions. NEVER touched during selection or training.
             Final raw + triplet Spearman reported here.
  * SELECT = a disjoint inner fold, used ONLY to pick the encoder checkpoint by RAW
             embedding-Spearman (the early-stop / model-selection criterion). Metric-
             independent, so it does not bias the triplet comparison.
  * TRAIN  = the remaining motions, used for triplet metric-learning.
  All three disjoint -> checkpoint selection never sees TEST -> publishable.

Protocol: K=5 outer folds (magnitude-stratified per action). For each outer fold f:
  TEST = fold[f]. Of the remaining K-1 folds, one (fold[(f+1)%K]) = SELECT, rest = TRAIN.
  1. SELECT step: encode SELECT motions with each candidate checkpoint; pick the checkpoint
     maximising MEAN raw-Spearman across the 3 actions on SELECT (action-agnostic).
  2. Encode TRAIN+TEST with the selected checkpoint.
  3. RAW report: raw-Spearman on TEST (the encoder's own number).
  4. TRIPLET report: train triplet MLP (n seeds) on TRAIN, eval refined-Spearman on TEST.
  Average raw & triplet over the K outer folds -> fold-mean +/- std per action.

Reports BOTH raw and triplet so the triplet contribution (delta) is visible per action,
with proper fold error bars. Bars to beat: walking>0.478, pointing>0.370, picking>0.431
(max of DTW/geodesic per action).
"""
import sys, os, glob, random, argparse, time, json, shutil
import numpy as np

MS_ROOT = "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/motion-similarity"
TR_ROOT = "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets"
sys.path.insert(0, MS_ROOT); os.chdir(MS_ROOT)

import torch
from Config import Config
import cv_preliminary as cv

ACTIONS = ["walking", "pointing", "picking"]
BARS = {"walking": 0.478, "pointing": 0.370, "picking": 0.431}


def selected_keys_union(folds, idxs):
    return {a: set().union(*[folds[a][i] for i in idxs]) for a in ACTIONS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["MAMP+pose"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt-stride", type=int, default=3)     # MAMP trajectory subsample
    ap.add_argument("--n-triplet-seeds", type=int, default=5)
    ap.add_argument("--triplet-epochs", type=int, default=200)
    ap.add_argument("--ckpt-epochs", nargs="+", type=int, default=None,
                    help="explicit candidate epochs to select among (overrides stride scan)")
    ap.add_argument("--work-dir", default="/tmp/cv_nested")
    ap.add_argument("--out", default=os.path.join(MS_ROOT, "cv_nested_results.json"))
    ap.add_argument("--repeats", type=int, default=1,
                    help="REPEATED K-fold: number of independent fold partitions (different "
                         "seeds). Pools all repeats*K test estimates -> tightens SEM of the mean.")
    args = ap.parse_args()
    cv.ACTIONS = ACTIONS
    os.makedirs(args.work_dir, exist_ok=True)

    def ts(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    cfg = Config()
    modules, all_keys = {}, {}
    for a in ACTIONS:
        modules[a], all_keys[a] = cv.load_action_keys(a, cfg)

    results = {}
    for enc in args.encoders:
        ts(f"===== ENCODER {enc} ({args.repeats} repeat(s) x K={args.k}) =====")
        # candidate checkpoints
        if args.ckpt_epochs:
            cks = [c for c in cv.list_checkpoints(enc, 1) if c[0] in args.ckpt_epochs]
        else:
            cks = cv.list_checkpoints(enc, args.ckpt_stride)
        ts(f"  {len(cks)} candidate checkpoints: {[e for e,_ in cks]}")
        fold_rows = []
        for rep in range(args.repeats):
          # distinct fold partition per repeat -> averages over the partition lottery
          folds = {a: cv.make_folds(all_keys[a], args.k, args.seed + 1000 * rep) for a in ACTIONS}
          for f in range(args.k):
            t0 = time.time()
            tag = f"r{rep}f{f}"
            test = {a: folds[a][f] | {(0, 0, 0, 0)} for a in ACTIONS}
            sel_i = (f + 1) % args.k
            select = {a: folds[a][sel_i] | {(0, 0, 0, 0)} for a in ACTIONS}
            train_i = [i for i in range(args.k) if i != f and i != sel_i]
            train = {a: set().union(*[folds[a][i] for i in train_i]) | {(0, 0, 0, 0)}
                     for a in ACTIONS}

            # ---- SELECT: pick checkpoint by mean raw-Spearman on SELECT (never TEST) ----
            best = (None, -2.0, None, None)
            for ep, path in cks:
                per = {}
                for a in ACTIONS:
                    od = os.path.join(args.work_dir, f"{enc}_{tag}_ep{ep}_{a}")
                    cv.encode(enc, path, a, od)
                    per[a] = cv.raw_spearman(od, a, select[a], modules[a])
                vals = [v for v in per.values() if v == v]
                ms = float(np.mean(vals)) if vals else -2.0
                if ms > best[1]:
                    best = (ep, ms, per, path)
            sel_ep, sel_meanS, sel_per, sel_path = best
            ts(f"[{tag}] {enc} selected ep={sel_ep} (SELECT mean-S={sel_meanS:.3f} {({a:round(sel_per[a],3) for a in ACTIONS})})")

            # ---- encode TRAIN+TEST with selected checkpoint ----
            emb_dirs = {}
            for a in ACTIONS:
                od = os.path.join(args.work_dir, f"{enc}_{tag}_SEL_{a}")
                cv.encode(enc, sel_path, a, od)
                emb_dirs[a] = od

            # ---- RAW report on TEST ----
            raw_test = {a: cv.raw_spearman(emb_dirs[a], a, test[a], modules[a]) for a in ACTIONS}

            # ---- TRIPLET: train on TRAIN, eval refined on TEST ----
            trip = {a: [] for a in ACTIONS}
            for s in range(args.n_triplet_seeds):
                per = cv.train_triplet_on_fold(emb_dirs, train, test, seed=42 + s,
                                               n_epochs=args.triplet_epochs, modules=modules)
                for a in ACTIONS:
                    if per.get(a) is not None:
                        trip[a].append(per[a])
            trip_test = {a: (float(np.mean(v)) if v else float("nan")) for a, v in trip.items()}

            fold_rows.append({"repeat": rep, "fold": f, "selected_epoch": sel_ep,
                              "raw_test": raw_test, "triplet_test": trip_test,
                              "seconds": round(time.time() - t0)})
            json.dump({enc: {"folds": fold_rows}}, open(args.out, "w"), indent=2)
            ts(f"[{tag}] raw_test={ {a:round(raw_test[a],3) for a in ACTIONS} }  trip_test={ {a:round(trip_test[a],3) for a in ACTIONS} }")

        # ---- aggregate over all repeats*K test estimates ----
        agg = {"raw": {}, "triplet": {}, "bars": BARS, "n_estimates": len(fold_rows)}
        for a in ACTIONS:
            rv = [r["raw_test"][a] for r in fold_rows if r["raw_test"][a] == r["raw_test"][a]]
            tv = [r["triplet_test"][a] for r in fold_rows if r["triplet_test"][a] == r["triplet_test"][a]]
            rsem = float(np.std(rv) / np.sqrt(len(rv))) if rv else float("nan")
            tsem = float(np.std(tv) / np.sqrt(len(tv))) if tv else float("nan")
            agg["raw"][a] = {"mean": round(float(np.mean(rv)), 4), "std": round(float(np.std(rv)), 4),
                             "sem": round(rsem, 4), "n": len(rv)}
            agg["triplet"][a] = {"mean": round(float(np.mean(tv)), 4), "std": round(float(np.std(tv)), 4),
                                 "sem": round(tsem, 4), "n": len(tv)}
        results[enc] = {"folds": fold_rows, "aggregate": agg}
        json.dump(results, open(args.out, "w"), indent=2)
        ts(f"===== {enc} AGGREGATE (n={len(fold_rows)} test estimates = {args.repeats}x{args.k}) =====")
        for a in ACTIONS:
            r = agg["raw"][a]; t = agg["triplet"][a]; bar = BARS[a]
            # 'beats robustly' = mean - 1.96*SEM > bar (95% CI lower bound clears the bar)
            r_rob = (r["mean"] - 1.96 * r["sem"]) > bar
            ts(f"  {a:9} RAW {r['mean']:.3f} +/- {r['std']:.3f} (SEM {r['sem']:.3f}) "
               f"| mean {'>' if r['mean']>bar else '<='} bar {bar} | 95%-CI-clears-bar: {r_rob} "
               f"| +TRIP {t['mean']:.3f} +/- {t['std']:.3f} (delta {t['mean']-r['mean']:+.3f})")


if __name__ == "__main__":
    main()
