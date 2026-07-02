#!/usr/bin/env python
"""
Preliminary nested cross-validation (4 non-MLD encoders) for the checkpoint-fairness
re-analysis. Runs on chimera (motion-similarity root) where the trainer + loaders import.
Encode scripts live in the sibling `triplets` root and are shelled out with absolute paths.

Protocol:
  * K=5 outer folds over each action's effort classes, magnitude-stratified, seeded.
  * INNER (per outer fold, over outer-TRAIN keys split into inner-train/inner-val): for each
    encoder, scan stored checkpoints; encode inner-VAL clips; per checkpoint compute raw
    Spearman (embedding-L2 vs d_perc) PER ACTION on inner-val; pick the SINGLE checkpoint that
    maximises MEAN-Spearman across the 3 actions (action-agnostic selection).
  * OUTER: encode all clips with the selected checkpoint; train the LayerNorm triplet MLP
    (n seeds) on outer-TRAIN; evaluate raw-Spearman and triplet-Spearman on outer-TEST.
  * Average over the 5 folds -> fair preliminary numbers.

Selection signal d_perc = 1 - count_normalized[(selected0=0, selected1=2)], reused from
embedding_inference_autoencoder.get_inverse_direct_comparison_value.

Validates the pipeline before MLD trajectories land, and gives an early read on whether the
MAMP+pose headline survives FAIR checkpoint selection (minus only the MLD comparators).
"""
import os, sys, json, glob, random, argparse, subprocess, time
import numpy as np
from scipy import stats

# --- roots -------------------------------------------------------------------------------
MS_ROOT = "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/motion-similarity"
TR_ROOT = "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets"
sys.path.insert(0, MS_ROOT)
os.chdir(MS_ROOT)

import torch
import pandas as pd
from Config import Config
from networks.triplet_mining import TripletMining
from run_enhanced_embedding_triplet_training import (
    run_enhanced_triplet_training, EnhancedEmbeddingTrainer)


# Inlined from embedding_inference_autoencoder to avoid its unguarded `import tensorflow`
# (that module pulls TF at load time; TF is absent from torch_gpu_cu12). These two helpers
# have no TF dependency themselves.
def create_triplet_module(anim_name, bool_drop, bool_fixed, sq_lr, sq_cn, config,
                          valid_indices=None):
    return TripletMining(bool_drop, bool_fixed, sq_lr, sq_cn, anim_name, config,
                         valid_indices=valid_indices)


def get_inverse_direct_comparison_value(key1, key2, triplet_modules):
    if not isinstance(triplet_modules, list):
        triplet_modules = [triplet_modules]
    action1, effort1 = key1
    action2, effort2 = key2
    if effort1 == (0, 0, 0, 0) or effort2 == (0, 0, 0, 0):
        return None
    if action1 != action2:
        return None
    for module in triplet_modules:
        if module.anim_name == action1:
            if not hasattr(module, "df_comparisons") or module.df_comparisons is None:
                return None
            df = module.df_comparisons
            pair = df[df["efforts_tuples"].apply(
                lambda x: set(x) == set([effort1, effort2]) if isinstance(x, list) else False)]
            tgt = pair[(pair["selected0"] == 0) & (pair["selected1"] == 2)]
            if not tgt.empty and "count_normalized" in tgt.columns \
                    and not pd.isna(tgt["count_normalized"].iloc[0]):
                return 1 - tgt["count_normalized"].iloc[0]
            return None
    return None

ACTIONS = ["walking", "pointing", "picking"]          # trainer/d_perc convention (lowercase)
ACTION_CAP = {"walking": "Walking", "pointing": "Pointing", "picking": "Picking"}  # file glob

# --- encoder registry --------------------------------------------------------------------
# family, ckpt_dir (under TR_ROOT), and encode invocation knobs.
ENCODERS = {
    "MAMP":      dict(family="mamp",    ckpt_dir="MAMP/output_dir/lma_mamp_holdout_1200",
                      config="config/lma_mamp_pretrain.yaml", stride_default=3),
    "MAMP+pose": dict(family="mamp",    ckpt_dir="MAMP/output_dir/lma_mamp_pose_holdout",
                      config="config/lma_mamp_pretrain.yaml", stride_default=3),
    "AE-vel":    dict(family="vanilla", ckpt_dir="checkpoints/v3_ablation_PROV",       n_windows=1,
                      stride_default=1),
    "AE-recon":  dict(family="vanilla", ckpt_dir="checkpoints/v3_ablation_PROV_noVel", n_windows=1,
                      stride_default=1),
    # MLD-AE families (new trajectory dirs; epoch_*.pt; encode via encode_sweep.py --kind ae)
    "MLD-AE-holdout": dict(family="mld", kind="ae", ckpt_dir="checkpoints_ae_mld/v0_ae_3ds_seed0_traj",
                           n_windows=1, stride_default=3),
    "MLD-AE-vel":     dict(family="mld", kind="ae", ckpt_dir="checkpoints_ae_mld/v0_ae_vel_3ds_seed0_traj",
                           n_windows=1, stride_default=3),
    "MLD-AE-prov":    dict(family="mld", kind="ae", ckpt_dir="checkpoints_ae_mld/v0_ae_provloss_3ds_seed0_traj",
                           n_windows=1, stride_default=3),
    # MLD-VAE (deterministic-mu encode via encode_sweep.py --kind vae); re-trained traj dir
    "MLD-VAE-holdout": dict(family="mld", kind="vae",
                            ckpt_dir="checkpoints_vae_mld/v0_mld_holdout_seed0_3ds",
                            n_windows=1, stride_default=3),
}
ENCODE_ENV = ("source $(conda info --base)/etc/profile.d/conda.sh; "
              "conda activate torch_gpu_cu12; ")


def list_checkpoints(enc, stride):
    cfg = ENCODERS[enc]
    d = os.path.join(TR_ROOT, cfg["ckpt_dir"])
    import re
    def _trailing_int(p):
        m = re.findall(r"(\d+)", os.path.basename(p))
        return int(m[-1]) if m else -1
    if cfg["family"] == "mamp":
        paths = glob.glob(os.path.join(d, "checkpoint-*.pth"))
        key = lambda p: int(p.split("checkpoint-")[1].split(".pth")[0])
    elif cfg["family"] == "mld":
        paths = glob.glob(os.path.join(d, "epoch_*.pt"))
        key = _trailing_int          # epoch_06700.pt -> 6700
    else:  # vanilla: cp_PROV_noVel_2100.pth / cp_PROV_10000.pth -> trailing int = epoch
        paths = glob.glob(os.path.join(d, "cp_*.pth"))
        key = _trailing_int
    paths = sorted(paths, key=key)
    if stride > 1:
        # keep endpoints, stride the interior
        idx = list(range(0, len(paths), stride))
        if (len(paths) - 1) not in idx:
            idx.append(len(paths) - 1)
        paths = [paths[i] for i in idx]
    return [(key(p), p) for p in paths]


def encode(enc, ckpt_path, action, out_dir):
    """Run the encode script (in TR_ROOT) for one ckpt/action into out_dir (absolute)."""
    cfg = ENCODERS[enc]
    os.makedirs(out_dir, exist_ok=True)
    cap = ACTION_CAP[action]
    if cfg["family"] == "mamp":
        # ckpt/config resolve against MAMP/ (_HERE) -> strip the leading "MAMP/"
        ck = os.path.relpath(ckpt_path, os.path.join(TR_ROOT, "MAMP"))
        py = (f"python MAMP/encode_mamp_lma.py --ckpt {ck} --config {cfg['config']} "
              f"--action {cap} --out-dir {out_dir}")
    elif cfg["family"] == "mld":
        ck = os.path.relpath(ckpt_path, TR_ROOT)
        kind = cfg.get("kind", "ae")
        py = (f"python probes/encode_sweep.py --kind {kind} --ckpt {ck} --action {cap} "
              f"--n-windows {cfg['n_windows']} --out-dir {out_dir}")
    else:  # vanilla
        ck = os.path.relpath(ckpt_path, TR_ROOT)
        py = (f"python probes/encode_aetransformer.py --ckpt {ck} --action {cap} "
              f"--n-windows {cfg['n_windows']} --out-dir {out_dir}")
    cmd = ["bash", "-lc", ENCODE_ENV + f"cd {TR_ROOT}; " + py]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    n_pt = len(glob.glob(os.path.join(out_dir, "*_emb.pt")))
    if r.returncode != 0 or n_pt == 0:
        raise RuntimeError(
            f"encode failed: enc={enc} action={action} ckpt={ckpt_path}\n"
            f"  returncode={r.returncode} files_written={n_pt}\n"
            f"  cmd={py}\n  output tail:\n" + "\n".join(r.stdout.splitlines()[-15:]))


# --- folds -------------------------------------------------------------------------------
def make_folds(keys, k, seed):
    """Magnitude-stratified K-fold partition (neutral excluded; callers re-add it)."""
    rng = random.Random(seed)
    by_mag = {}
    for key in sorted(keys):
        if key == (0, 0, 0, 0):
            continue
        by_mag.setdefault(sum(abs(e) for e in key), []).append(key)
    folds = [set() for _ in range(k)]
    for mag in sorted(by_mag):
        grp = by_mag[mag][:]
        rng.shuffle(grp)
        for i, key in enumerate(grp):
            folds[i % k].add(key)
    return folds


# --- raw-perceptual Spearman -------------------------------------------------------------
def load_action_keys(action, config):
    """All effort keys for an action, via a triplet module over the full set."""
    m = create_triplet_module(action, False, True, False, False, config, valid_indices=None)
    return m, sorted(k for k in _module_keys(m) if k != (0, 0, 0, 0))


def _module_keys(module):
    # TripletMining exposes its class keys; fall back to df if needed.
    if hasattr(module, "exemplar_dict") and module.exemplar_dict:
        return list(module.exemplar_dict.keys())
    if hasattr(module, "df_comparisons") and module.df_comparisons is not None:
        ks = set()
        for tup in module.df_comparisons["efforts_tuples"]:
            if isinstance(tup, list):
                ks.update(tup)
        return list(ks)
    return []


def raw_spearman(emb_dir, action, keys, module):
    """Spearman(embedding L2, d_perc) over pairs whose BOTH keys are in `keys`."""
    cap = ACTION_CAP[action]
    embs = {}
    for key in keys:
        if key == (0, 0, 0, 0):
            continue
        fp = os.path.join(emb_dir, f"{cap}_{'_'.join(map(str, key))}_emb.pt")
        if os.path.exists(fp):
            embs[key] = torch.load(fp, map_location="cpu", weights_only=True).flatten().numpy()
    dists, percs = [], []
    klist = sorted(embs)
    for i in range(len(klist)):
        for j in range(i + 1, len(klist)):
            dp = get_inverse_direct_comparison_value(
                (action, klist[i]), (action, klist[j]), module)
            if dp is None:
                continue
            dists.append(float(np.linalg.norm(embs[klist[i]] - embs[klist[j]])))
            percs.append(dp)
    if len(dists) < 4:
        return float("nan")
    return stats.spearmanr(dists, percs).correlation


# --- inner-loop checkpoint selection -----------------------------------------------------
def select_checkpoint(enc, inner_val_keys, modules, work_dir, stride):
    """Encode inner-val with each ckpt; pick the one with max MEAN-Spearman over actions."""
    ckpts = list_checkpoints(enc, stride)
    best = (None, -2.0, None)  # (epoch, mean_spearman, per_action)
    for ep, path in ckpts:
        per = {}
        ok = True
        for a in ACTIONS:
            od = os.path.join(work_dir, f"{enc}_ep{ep}_{a}")
            try:
                encode(enc, path, a, od)
            except subprocess.CalledProcessError:
                ok = False; break
            per[a] = raw_spearman(od, a, inner_val_keys[a], modules[a])
        if not ok:
            continue
        vals = [v for v in per.values() if v == v]  # drop nan
        if not vals:
            continue
        mean_s = float(np.mean(vals))
        if mean_s > best[1]:
            best = (ep, mean_s, per)
    return best


# --- triplet training with injected fold split -------------------------------------------
def train_triplet_on_fold(emb_dirs, train_keys, test_keys, seed, n_epochs, modules):
    """Patch the trainer's split to our fold, train, return per-action TEST results."""
    cfg = Config()
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    def _fold_split(self, similarity_dicts, val_ratio=0.4):
        # similarity_dicts is a list in ACTIONS order (built from embedding_dirs).
        # train_keys/test_keys are {action: set_of_effort_tuples}. Index by position so
        # each action's dict is filtered against THAT action's fold keys (the earlier bug:
        # `k in train_keys` tested effort-tuple against the action-name dict keys -> always
        # False -> only neutral survived -> zero triplets).
        tr, va = [], []
        for i, d in enumerate(similarity_dicts):
            a = ACTIONS[i]
            akeys = set(d.keys())
            t = {k for k in akeys if k in train_keys[a] or k == (0, 0, 0, 0)}
            v = {k for k in akeys if k in test_keys[a] or k == (0, 0, 0, 0)}
            tr.append(t); va.append(v)
        return tr, va

    EnhancedEmbeddingTrainer._create_stratified_split = _fold_split

    # CRITICAL: load_similarity_data_from_embeddings caches the per-action similarity dict
    # at {action}_embeddings_{method}_..._local.pickle, keyed ONLY on action+method (NOT the
    # embedding dir). Across folds/checkpoints that cache would be silently reused, making
    # every fold train on fold-0's embeddings and invalidating the CV. Clear it each time so
    # EmbeddingDataset regenerates from THIS fold's emb_dirs.
    _se = cfg.similarity_exemplars_dir
    for a in ACTIONS:
        for meth in ("rots_only", "concat"):
            p = os.path.join(_se, f"{a}_embeddings_{meth}_{cfg.similarity_dict_file_name}")
            if os.path.exists(p):
                os.remove(p)

    net, _results = run_enhanced_triplet_training(
        embedding_dirs=emb_dirs, combination_method="rots_only",
        clustering_config={"n_clusters": 5, "selection_strategy": "centroid",
                           "random_state": 42},
        training_config={"n_epochs": n_epochs, "scheduler": "plateau",
                         "use_perception_loss": False, "use_adaptive_distance": False,
                         "seed": seed})

    # Per-action TEST Spearman on REFINED embeddings. EmbeddingRefiningSimilarityNetwork has
    # no correlation method (its evaluate() is a stub); the local refinement pattern
    # (similarity_network.py L533/549, infer_emb_vec.py) is: push each clip embedding through
    # the trained MLP `net.network`, then correlate pairwise L2 vs d_perc -- IDENTICAL pairing
    # to raw_spearman, just on refined vectors. This guarantees raw/triplet equivalence.
    return refined_spearman(net, emb_dirs, test_keys, modules)


def refined_spearman(net, emb_dirs, test_keys, modules):
    """Apply trained MLP to each test _emb.pt, write refined vectors, reuse raw_spearman."""
    net.network.eval()
    dev = next(net.network.parameters()).device
    per_action = {}
    for a in ACTIONS:
        cap = ACTION_CAP[a]
        rdir = emb_dirs[a] + "_refined"
        os.makedirs(rdir, exist_ok=True)
        for fp in glob.glob(os.path.join(emb_dirs[a], "*_emb.pt")):
            v = torch.load(fp, map_location="cpu", weights_only=True).flatten().float()
            with torch.no_grad():
                r = net.network(v.unsqueeze(0).to(dev)).squeeze(0).cpu()
            torch.save(r, os.path.join(rdir, os.path.basename(fp)))
        per_action[a] = raw_spearman(rdir, a, test_keys[a] | {(0, 0, 0, 0)}, modules[a])
    return per_action


def seed42_split(all_keys, val_ratio=0.4, seed=42):
    """EXACT replica of pipelines/infer_emb_vec.create_train_val_split: per action,
    sorted keys, random.Random(seed) shuffle, first val_ratio -> val; neutral in both."""
    train, val = {}, {}
    for a in ACTIONS:
        rng = random.Random(seed)
        keys = sorted(k for k in all_keys[a] if k != (0, 0, 0, 0))
        rng.shuffle(keys)
        vs = max(1, int(len(keys) * val_ratio))
        val[a] = set(keys[:vs]) | {(0, 0, 0, 0)}
        train[a] = set(keys[vs:]) | {(0, 0, 0, 0)}
    return train, val


def repro_mode(args, cfg, modules, all_keys, _ts):
    """EQUIVALENCE CHECK: reproduce draft §6.6 for MAMP+pose at a single epoch on the fixed
    seed-42 split. Targets (draft): raw 0.525/0.263/0.581, +triplet 0.527/0.465/0.451."""
    enc = "MAMP+pose"
    ep = args.repro_epoch
    _, path = next(c for c in list_checkpoints(enc, 1) if c[0] == ep)
    _ts(f"REPRO mode: {enc} @ epoch {ep}  (seed-42 val_ratio=0.4 split, NOT k-fold)")

    train_keys, val_keys = seed42_split(all_keys)
    for a in ACTIONS:
        _ts(f"  {a}: {len(val_keys[a])-1} val / {len(train_keys[a])-1} train classes")

    # encode @ep for all actions
    emb_dirs = {}
    for a in ACTIONS:
        od = os.path.join(args.work_dir, f"repro_{enc}_{ep}_{a}")
        encode(enc, path, a, od)
        emb_dirs[a] = od

    # RAW on the VAL split (exactly the §6.6 raw column)
    raw = {a: raw_spearman(emb_dirs[a], a, val_keys[a], modules[a]) for a in ACTIONS}
    _ts(f"RAW   (val split): {raw}")

    # +TRIPLET: train on TRAIN split, eval refined on VAL split, averaged over seeds
    trip = {a: [] for a in ACTIONS}
    for s in range(args.n_triplet_seeds):
        per = train_triplet_on_fold(emb_dirs, train_keys, val_keys, seed=42 + s,
                                    n_epochs=args.triplet_epochs, modules=modules)
        for a in ACTIONS:
            if per.get(a) is not None:
                trip[a].append(per[a])
        _ts(f"  seed {s}: {per}")
    trip_mean = {a: (float(np.mean(v)) if v else float('nan')) for a, v in trip.items()}

    print("\n===== EQUIVALENCE: remote harness vs draft §6.6 (MAMP+pose@%d) =====" % ep)
    tgt_raw = {"walking": 0.525, "pointing": 0.263, "picking": 0.581}
    tgt_trip = {"walking": 0.527, "pointing": 0.465, "picking": 0.451}
    print(f"{'action':9} {'raw(mine)':>10} {'raw(draft)':>11} | {'trip(mine)':>11} {'trip(draft)':>12}")
    for a in ACTIONS:
        print(f"{a:9} {raw[a]:10.3f} {tgt_raw[a]:11.3f} | {trip_mean[a]:11.3f} {tgt_trip[a]:12.3f}")
    import json
    json.dump({"epoch": ep, "raw": raw, "triplet": trip_mean,
               "draft_raw": tgt_raw, "draft_triplet": tgt_trip},
              open(args.out, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-triplet-seeds", type=int, default=5)
    ap.add_argument("--triplet-epochs", type=int, default=200)
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS))
    ap.add_argument("--work-dir", default="/tmp/cv_prelim")
    ap.add_argument("--out", default=os.path.join(MS_ROOT, "cv_preliminary_results.json"))
    ap.add_argument("--ckpt-stride", type=int, default=0,
                    help="override per-encoder stride_default (0 = use registry default)")
    ap.add_argument("--repro-epoch", type=int, default=0,
                    help="EQUIVALENCE MODE: reproduce draft §6.6 on this single MAMP+pose epoch "
                         "using the fixed seed-42 val_ratio=0.4 shuffle split (NOT k-fold).")
    args = ap.parse_args()

    def _ts(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    cfg = Config()
    os.makedirs(args.work_dir, exist_ok=True)

    # full keys + d_perc module per action
    modules, all_keys = {}, {}
    for a in ACTIONS:
        modules[a], all_keys[a] = load_action_keys(a, cfg)
        print(f"[keys] {a}: {len(all_keys[a])} effort classes")

    if args.repro_epoch:
        return repro_mode(args, cfg, modules, all_keys, _ts)

    # outer folds per action
    folds = {a: make_folds(all_keys[a], args.k, args.seed) for a in ACTIONS}

    results = {enc: {"folds": []} for enc in args.encoders}
    for enc in args.encoders:
        stride = args.ckpt_stride if args.ckpt_stride > 0 else ENCODERS[enc]["stride_default"]
        n_ck = len(list_checkpoints(enc, stride))
        _ts(f"========== ENCODER {enc} (stride {stride}, {n_ck} ckpts) ==========")
        for f in range(args.k):
            t0 = time.time()
            test_keys = {a: folds[a][f] for a in ACTIONS}
            train_keys = {a: set().union(*[folds[a][g] for g in range(args.k) if g != f])
                          for a in ACTIONS}
            # inner-val = one of the outer-train folds (the next index) for ckpt selection
            inner_val = {a: folds[a][(f + 1) % args.k] for a in ACTIONS}

            _ts(f"[fold {f}] {enc} START selection ({n_ck} ckpts x 3 actions)")
            ep, mean_s, per = select_checkpoint(enc, inner_val, modules, args.work_dir, stride)
            _ts(f"[fold {f}] {enc} selected ckpt ep={ep} inner mean-S={mean_s:.3f} {per} "
                f"(selection {time.time()-t0:.0f}s)")

            # encode all clips with selected ckpt -> per-action emb dirs for triplet training
            _, sel_path = next((c for c in list_checkpoints(enc, stride) if c[0] == ep), (None, None))
            emb_dirs = {}
            for a in ACTIONS:
                od = os.path.join(args.work_dir, f"{enc}_SEL_f{f}_{a}")
                encode(enc, sel_path, a, od)
                emb_dirs[a] = od

            # raw-Spearman on OUTER-TEST (the fair raw number)
            raw_test = {a: raw_spearman(emb_dirs[a], a, test_keys[a] | {(0,0,0,0)}, modules[a])
                        for a in ACTIONS}
            _ts(f"[fold {f}] {enc} raw_test={raw_test} (start triplet, {args.n_triplet_seeds} seeds)")

            # triplet on OUTER-TRAIN, eval OUTER-TEST, averaged over seeds
            trip = {a: [] for a in ACTIONS}
            for s in range(args.n_triplet_seeds):
                per_action = train_triplet_on_fold(emb_dirs, train_keys, test_keys,
                                                   seed=42 + s, n_epochs=args.triplet_epochs,
                                                   modules=modules)
                for a in ACTIONS:
                    sp = per_action.get(a)
                    if sp is not None:
                        trip[a].append(sp)
            trip_mean = {a: (float(np.mean(v)) if v else float("nan")) for a, v in trip.items()}

            results[enc]["folds"].append({
                "fold": f, "selected_epoch": ep, "inner_mean_spearman": mean_s,
                "raw_test_spearman": raw_test, "triplet_test_spearman": trip_mean,
                "seconds": round(time.time() - t0, 1)})
            json.dump(results, open(args.out, "w"), indent=2)
            print(f"[fold {f}] {enc} raw={raw_test} trip={trip_mean} ({results[enc]['folds'][-1]['seconds']}s)")

    # aggregate
    for enc in args.encoders:
        agg = {"raw": {}, "triplet": {}}
        for a in ACTIONS:
            rv = [fd["raw_test_spearman"][a] for fd in results[enc]["folds"]
                  if fd["raw_test_spearman"][a] == fd["raw_test_spearman"][a]]
            tv = [fd["triplet_test_spearman"][a] for fd in results[enc]["folds"]
                  if fd["triplet_test_spearman"][a] == fd["triplet_test_spearman"][a]]
            agg["raw"][a] = [float(np.mean(rv)), float(np.std(rv))] if rv else None
            agg["triplet"][a] = [float(np.mean(tv)), float(np.std(tv))] if tv else None
        results[enc]["aggregate"] = agg
    json.dump(results, open(args.out, "w"), indent=2)
    print("\n===== AGGREGATE (mean±std over folds) =====")
    print(json.dumps({e: results[e]["aggregate"] for e in args.encoders}, indent=2))


if __name__ == "__main__":
    main()
