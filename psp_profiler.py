#!/usr/bin/env python
"""
Perceptual-Spearman Plateau (PSP) profiler.

For each encoder, walk its checkpoint trajectory and measure raw held-out perceptual
Spearman as a function of epoch, on the SAME SELECT folds the nested-CV uses for checkpoint
selection. This locates each model's PSP -- the epoch beyond which perceptual signal no longer
improves (and may degrade, per the over-training divergence in the paper's S4.4.1) -- which is
what bounds that model's [init, PSP] candidate window for the matched 7-candidate sweep.

WHY SELECT folds, repeated, averaged:
  Selection picks a checkpoint by raw Spearman on a disjoint inner (SELECT) fold. So PSP must be
  measured on SELECT folds, mean over the 3 actions, AVERAGED over the repeated-CV seeds -- not a
  single split (single-split perceptual Spearman has fold-std ~0.15; a single curve is noise).
  This yields the same quantity selection optimizes, so the window we derive is the one that is
  actually searched.

OUTPUT per model:
  - curve: list of (epoch, mean_select_spearman, std_over_seeds)
  - psp_epoch: detected plateau epoch (argmax of smoothed curve; see _detect_psp)
  - reached_psp: bool -- False if the curve is still RISING at the last checkpoint (=> the model
    has NOT reached its PSP yet and REQUIRES MORE TRAINING before its window can be set).

This is READ-ONLY on checkpoints. It only encodes + correlates; it never trains or deletes.
Run on chimera (paths + encode scripts live there). Results -> psp_profile_results.json.
"""
import os, sys, argparse, time, json, tempfile, shutil
import numpy as np

MS_ROOT = "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/motion-similarity"
sys.path.insert(0, MS_ROOT)
os.chdir(MS_ROOT)
os.environ.setdefault("MOTION_IS_REMOTE", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

from Config import Config
import cv_preliminary as cv

ACTIONS = cv.ACTIONS  # walking, pointing, picking


def _detect_psp(epochs, means, stds=None, n_splits=None, min_band=0.02):
    """Locate PSP (plateau ONSET) and whether the plateau was reached -- NOISE-AWARE.

    Perceptual Spearman on SELECT folds carries fold-noise (std ~0.14); a fixed-threshold
    "still rising" test mistakes noise-level drift for genuine improvement (it falsely flags a
    plateaued model as needing more training -- the opposite of saving compute). So we judge
    against the curve's OWN noise band, not an absolute epsilon.

    band = a noise margin = max(SEM_of_the_peak, MIN_BAND). SEM = std/sqrt(n_splits) at the peak
    if available, else a conservative default.
      PSP   = the EARLIEST epoch whose smoothed mean reaches (peak_mean - band). This is the
              plateau ONSET -- candidates past it gain only noise, so it is the right window end.
      reached = False ONLY if the curve is still climbing by MORE THAN the band over its final
              leg AND the peak sits at the very end -- i.e. genuine, above-noise improvement that
              has not turned over. Noise-level end-drift => reached=True.
    """
    e = np.asarray(epochs, float)
    m = np.asarray(means, float)
    ok = ~np.isnan(m)
    e, m = e[ok], m[ok]
    if stds is not None:
        s = np.asarray(stds, float)[ok]
    else:
        s = np.full_like(m, np.nan)
    if len(m) < 3:
        return (int(e[-1]) if len(e) else -1), False
    sm = np.convolve(m, np.ones(3) / 3, mode="same")
    sm[0], sm[-1] = m[0], m[-1]
    i_max = int(np.argmax(sm))
    peak = sm[i_max]
    # noise band from the peak checkpoint's SEM (std over splits / sqrt(n_splits))
    sem = (s[i_max] / np.sqrt(n_splits)) if (n_splits and not np.isnan(s[i_max])) else np.nan
    band = max(min_band, sem if not np.isnan(sem) else 0.0)
    # PSP = earliest epoch within `band` of the peak (plateau onset)
    within = np.where(sm >= peak - band)[0]
    psp_epoch = int(e[int(within[0])]) if len(within) else int(e[i_max])
    # reached = not (peak-at-end AND end-leg rising ABOVE the noise band)
    end_leg_gain = (sm[-1] - sm[-2]) if len(sm) >= 2 else 0.0
    peak_at_end = i_max >= len(e) - 2
    reached = not (peak_at_end and end_leg_gain > band)
    return psp_epoch, bool(reached)


def profile_model(enc, k, repeats, seed, epoch_grid, ts, workroot, min_band=0.02):
    cfg = Config()
    # UNIFORM epoch grid (identical across models): keep only checkpoints whose epoch is a
    # multiple of `epoch_grid`. This is the pre-registered, non-cherry-pick sampling rule --
    # NOT per-model strides, NOT "every checkpoint" (save intervals differ across models).
    # epoch_grid must be a multiple of every model's save interval (100 = LCM of 20/25/100).
    all_ckpts = cv.list_checkpoints(enc, 1)            # stride 1 = every saved checkpoint
    ckpts = [(ep, p) for (ep, p) in all_ckpts if ep > 0 and ep % epoch_grid == 0]
    if not ckpts:
        ts(f"  [{enc}] NO checkpoints on the {epoch_grid}-epoch grid -- skipping")
        return None
    # sanity: warn if this model's native interval does not divide the grid (=> sparse sampling)
    if len(all_ckpts) >= 2:
        interval = all_ckpts[1][0] - all_ckpts[0][0]
        if interval > 0 and epoch_grid % interval != 0:
            ts(f"  [{enc}] WARNING: save interval {interval} does not divide grid {epoch_grid} "
               f"-- grid points may miss checkpoints (non-uniform!)")
    ts(f"  [{enc}] {len(ckpts)} checkpoints on {epoch_grid}-grid, epochs "
       f"{ckpts[0][0]}..{ckpts[-1][0]} (stride {ckpt_stride})")

    # action modules + full key sets (canonical), once
    modules, all_keys = {}, {}
    for a in ACTIONS:
        modules[a], all_keys[a] = cv.load_action_keys(a, cfg)

    # precompute the SELECT folds per (repeat, action) -- identical seeds to cv_nested
    # cv_nested: outer fold f -> SELECT = inner fold (f+1)%k. We average perceptual Spearman over
    # ALL outer folds' SELECT sets (every fold serves as a SELECT once), per repeat, to match the
    # selection signal's distribution without privileging one split.
    select_sets = {}  # (rep) -> list over outer-folds of {action: set(keys incl neutral)}
    for rep in range(repeats):
        folds = {a: cv.make_folds(all_keys[a], k, seed + 1000 * rep) for a in ACTIONS}
        per_fold = []
        for f in range(k):
            sel_i = (f + 1) % k
            per_fold.append({a: folds[a][sel_i] | {(0, 0, 0, 0)} for a in ACTIONS})
        select_sets[rep] = per_fold

    curve = []
    for epoch, path in ckpts:
        # encode all 3 actions for this checkpoint into a temp dir, score, discard
        emb_dir = tempfile.mkdtemp(prefix=f"psp_{enc}_{epoch}_", dir=workroot)
        try:
            try:
                for a in ACTIONS:
                    cv.encode(enc, path, a, emb_dir)
            except Exception as ex:
                ts(f"    ep{epoch}: ENCODE FAILED ({str(ex).splitlines()[0]}) -- nan")
                curve.append((epoch, float("nan"), float("nan")))
                continue
            # mean-over-actions Spearman on each SELECT set; collect across all rep*fold splits
            per_split = []
            for rep in range(repeats):
                for sel in select_sets[rep]:
                    sp = [cv.raw_spearman(emb_dir, a, sel[a], modules[a]) for a in ACTIONS]
                    sp = [x for x in sp if not np.isnan(x)]
                    if sp:
                        per_split.append(float(np.mean(sp)))
            if per_split:
                mean, std = float(np.mean(per_split)), float(np.std(per_split))
            else:
                mean, std = float("nan"), float("nan")
            curve.append((epoch, mean, std))
            ts(f"    ep{epoch:>6}: SELECT-meanS={mean:.4f} +/-{std:.4f} (n_splits={len(per_split)})")
        finally:
            shutil.rmtree(emb_dir, ignore_errors=True)

    epochs = [c[0] for c in curve]
    means = [c[1] for c in curve]
    stds = [c[2] for c in curve]
    n_splits_total = repeats * k
    psp_epoch, reached = _detect_psp(epochs, means, stds, n_splits_total, min_band)
    ts(f"  [{enc}] PSP={psp_epoch}  reached={reached}"
       + ("" if reached else "  <-- STILL RISING: NEEDS MORE TRAINING before window is valid"))
    return dict(encoder=enc, epoch_grid=epoch_grid, last_epoch=epochs[-1],
                psp_epoch=psp_epoch, reached_psp=reached,
                curve=[dict(epoch=e, mean=m, std=s) for e, m, s in curve])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", required=True,
                    help="encoder names from cv_preliminary.ENCODERS")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epoch-grid", type=int, default=100,
                    help="UNIFORM epoch sampling grid applied identically to ALL models (default "
                         "100 = LCM of save intervals 20/25/100). Keeps only checkpoints whose "
                         "epoch is a multiple of this. This is the pre-registered non-cherry-pick "
                         "sampling rule; do NOT use per-model values.")
    ap.add_argument("--min-band", type=float, default=0.02,
                    help="noise band floor for PSP/plateau detection (sensitivity-check over "
                         "{0.01,0.02,0.03}; default 0.02 is the pre-registered value).")
    ap.add_argument("--out", default=os.path.join(MS_ROOT, "psp_profile_results.json"))
    args = ap.parse_args()

    def ts(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    cv.ACTIONS = ACTIONS
    workroot = tempfile.mkdtemp(prefix="psp_work_")
    ts(f"PSP profiler: encoders={args.encoders} k={args.k} repeats={args.repeats} "
       f"epoch_grid={args.epoch_grid} min_band={args.min_band}")
    results = {}
    try:
        for enc in args.encoders:
            r = profile_model(enc, args.k, args.repeats, args.seed, args.epoch_grid, ts, workroot,
                              min_band=args.min_band)
            if r is not None:
                results[enc] = r
                json.dump(results, open(args.out, "w"), indent=2)
    finally:
        shutil.rmtree(workroot, ignore_errors=True)

    ts("==== PSP SUMMARY ====")
    for enc, r in results.items():
        flag = "" if r["reached_psp"] else "  *** NEEDS MORE TRAINING ***"
        ts(f"  {enc:<18} PSP={r['psp_epoch']:>6}  last_ckpt={r['last_epoch']:>6}  "
           f"reached={r['reached_psp']}{flag}")
    ts(f"results -> {args.out}")


if __name__ == "__main__":
    main()
