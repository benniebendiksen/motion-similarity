#!/usr/bin/env python3
"""
Embedding-Only Motion Similarity Analysis (No Triplet/Refinement Models)
+ Triplet ground-truth integration

- Loads single 6D AE embeddings from *_encoded_2 directories via SingleEmbeddingDataset
- Builds per-animation TripletMining to access human comparison alphas (df_comparisons)
- Computes pairwise Embedding L2, Geodesic (raw), and DTW (raw)
- Correlates each distance set against human alphas on the validation subset
- Formats output into uniform "framework-style" sections with a per-section winner
"""

import os
import sys
import time
import pickle
import warnings
from pathlib import Path
from itertools import combinations
from typing import Dict, Any, Tuple, List

# Resolve project root from this file's location so the script works
# whether invoked from pipelines/ or from the project root.
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'networks'))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Optional: TF appears in your raw pickles
# TensorFlow is optional — only used for the GPU availability check helper.
try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

# --- Project-specific imports (REQUIRED) ---
from Config import Config
import src.organize_synthetic_data as osd

# New: our single-embedding dataset loader
from embedding_dataset_single import SingleEmbeddingDataset

# New: Triplet ground truth module (provides df_comparisons with human judgements)
from networks.triplet_mining import TripletMining

# -----------------------
# Pretty reporting helpers
# -----------------------

def _h(title: str, char: str = "=") -> str:
    bar = char * 70
    return f"\n{bar}\n{title}\n{bar}\n"

def _summarize_distances_from_pairs(pairs: List[Tuple[float, tuple, tuple]]) -> Dict[str, float]:
    vals = np.asarray([d for d, _, _ in pairs], dtype=float)
    if vals.size == 0:
        return dict(min=np.nan, median=np.nan, mean=np.nan, max=np.nan)
    return dict(min=float(vals.min()),
                median=float(np.median(vals)),
                mean=float(vals.mean()),
                max=float(vals.max()))

def _winner_by_strength(left: Dict[str, Any] | None,
                        right: Dict[str, Any] | None,
                        left_name: str, right_name: str) -> str:
    """Compare absolute Pearson, absolute Spearman, and R^2; return winner label."""
    l_abs_p = abs(left["pearson"]["r"]) if left else -1
    r_abs_p = abs(right["pearson"]["r"]) if right else -1
    l_abs_s = abs(left["spearman"]["r"]) if left else -1
    r_abs_s = abs(right["spearman"]["r"]) if right else -1
    l_r2    = left["linreg"]["r2"]       if left else -1
    r_r2    = right["linreg"]["r2"]      if right else -1

    score_l = (l_abs_p > r_abs_p) + (l_abs_s > r_abs_s) + (l_r2 > r_r2)
    score_r = (r_abs_p > l_abs_p) + (r_abs_s > l_abs_s) + (r_r2 > l_r2)
    if score_l > score_r:
        return left_name
    if score_r > score_l:
        return right_name
    return "Tie"

def render_module_report(
    anim_name: str,
    subset_name: str,
    method_name_left: str,
    method_name_right: str,
    corr_left: Dict[str, Any] | None,
    corr_right: Dict[str, Any] | None,
    dist_left: Dict[str, float],
    dist_right: Dict[str, float],
    valid_pairs_left: int,
    valid_pairs_right: int,
) -> None:
    print(_h(f"{anim_name.upper()} ({subset_name}) — {method_name_left} vs {method_name_right}"))

    print("[CORRELATIONS vs human judgements]")
    if corr_left:
        print(f"[{anim_name} | {method_name_left}]  "
              f"Pearson r={corr_left['pearson']['r']:.4f} (p={corr_left['pearson']['p']:.2e}),  "
              f"Spearman r={corr_left['spearman']['r']:.4f} (p={corr_left['spearman']['p']:.2e}),  "
              f"Slope={corr_left['linreg']['slope']:.4f},  R^2={corr_left['linreg']['r2']:.4f}  "
              f"(n={valid_pairs_left})")
    else:
        print(f"[{anim_name} | {method_name_left}]  No valid human-labelled pairs (n=0)")

    if corr_right:
        print(f"[{anim_name} | {method_name_right}] "
              f"Pearson r={corr_right['pearson']['r']:.4f} (p={corr_right['pearson']['p']:.2e}), "
              f"Spearman r={corr_right['spearman']['r']:.4f} (p={corr_right['spearman']['p']:.2e}), "
              f"Slope={corr_right['linreg']['slope']:.4f},  R^2={corr_right['linreg']['r2']:.4f}  "
              f"(n={valid_pairs_right})")
    else:
        print(f"[{anim_name} | {method_name_right}] No valid human-labelled pairs (n=0)")

    print("\n[Distance summaries]")
    print(f"{method_name_left}:  min={dist_left['min']:.4f}, median={dist_left['median']:.4f}, "
          f"mean={dist_left['mean']:.4f}, max={dist_left['max']:.4f}")
    print(f"{method_name_right}: min={dist_right['min']:.4f}, median={dist_right['median']:.4f}, "
          f"mean={dist_right['mean']:.4f}, max={dist_right['max']:.4f}")

    winner = _winner_by_strength(corr_left, corr_right, method_name_left, method_name_right)
    print(f"\n→ Stronger method (this section): {winner}\n")


# -----------------------
# Utility / helper blocks
# -----------------------

def create_train_val_split(similarity_dicts, val_ratio=0.4):
    """Deterministic train/val split by sorted keys, same logic as before."""
    train_indices, val_indices = [], []
    for anim_dict in similarity_dicts:
        keys = sorted(k for k in anim_dict.keys() if k != (0, 0, 0, 0))
        val_size = max(1, int(len(keys) * val_ratio))
        val_keys = set(keys[:val_size])
        train_keys = set(keys[val_size:])
        if (0, 0, 0, 0) in anim_dict:
            train_keys.add((0, 0, 0, 0))
            val_keys.add((0, 0, 0, 0))
        train_indices.append(train_keys)
        val_indices.append(val_keys)
    return train_indices, val_indices


def generate_embeddings_without_refinement(similarity_dict, anim_name):
    """
    Input:  {effort_tuple: [embedding_np_or_tensor, ...]}
    Output: {(anim_name, effort_tuple): 1D np.ndarray}
    """
    print(f"[{anim_name}] Extracting pretrained AE embeddings (no refinement)...")
    out = {}
    for class_tuple, exemplars in similarity_dict.items():
        if not exemplars:
            continue
        emb = exemplars[0]
        if hasattr(emb, "numpy"):  # TF tensor
            emb = emb.numpy()
        elif hasattr(emb, "detach"):  # torch tensor
            emb = emb.detach().cpu().numpy()
        elif not isinstance(emb, np.ndarray):
            emb = np.array(emb)
        if emb.ndim > 1:
            emb = emb.flatten()
        out[(anim_name, class_tuple)] = emb
    print(f"[{anim_name}] -> {len(out)} embeddings")
    if out:
        any_key = next(iter(out.keys()))
        print(f"[{anim_name}] sample embedding shape: {out[any_key].shape}")
    return out


def calculate_pairwise_distances(embeddings_dict):
    """Pairwise Euclidean, normalized to [0,1]. Returns [(norm_d, k1, k2), ...] sorted asc."""
    keys = list(embeddings_dict.keys())
    n = len(keys)
    print(f"Calculating pairwise L2 on {n} embeddings; {n*(n-1)//2} pairs...")

    raw = []
    start = time.time()
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(embeddings_dict[keys[i]] - embeddings_dict[keys[j]])
            raw.append((d, keys[i], keys[j]))
    if not raw:
        return []

    maxd = max(d for d, _, _ in raw)
    norm = [(d / maxd if maxd > 0 else 0.0, k1, k2) for (d, k1, k2) in raw]
    norm.sort(key=lambda t: t[0])
    print(f"Done pairwise L2 in {time.time()-start:.2f}s (normalized to [0,1])")
    return norm


def get_raw_features_without_dataloader(anim_name, config, valid_indices=None, balance_class_frame_counts=True):
    """
    Read raw variable-length motion exemplars directly from the pickle (no dataloader),
    preserving original frame counts. Returns {(anim_name, effort_tuple): np.ndarray[T, ...]}
    """
    pkl_path = config.similarity_exemplars_dir + anim_name + "_" + config.similarity_dict_file_name
    print(f"[{anim_name}] Loading raw motion directly: {pkl_path}")
    with open(pkl_path, "rb") as f:
        raw_dict = pickle.load(f)

    if balance_class_frame_counts:
        raw_dict = osd.balance_single_exemplar_similarity_classes_by_frame_count([raw_dict], 137)[0]

    if valid_indices is not None:
        raw_dict = {k: v for k, v in raw_dict.items() if k in valid_indices}

    out = {}
    for class_tuple, exemplars in raw_dict.items():
        if not exemplars:
            continue
        x = exemplars[0]
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        elif not isinstance(x, np.ndarray):
            x = np.array(x)
        out[(anim_name, class_tuple)] = x
    print(f"[{anim_name}] -> {len(out)} raw sequences (variable lengths OK)")
    return out


def compute_geodesic_distances(dict_raw_features, num_frames=137, num_joints=28):
    """
    Mean geodesic distance over all frames/joints for quaternionized raw features.
    Assumes reshape to (num_frames, num_joints, 4). Normalizes quats first.
    """
    print("Computing geodesic distances on normalized quaternions...")
    normed = {}
    for key, sample in dict_raw_features.items():
        q = sample.reshape(num_frames, num_joints, 4)
        q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-10)
        normed[key] = q

    keys = list(normed.keys())
    pairs = []
    for i, j in combinations(range(len(keys)), 2):
        q1 = normed[keys[i]]
        q2 = normed[keys[j]]
        dots = np.sum(q1 * q2, axis=-1)
        ang = 2.0 * np.arccos(np.clip(np.abs(dots), -1.0, 1.0))
        pairs.append((ang.mean(), keys[i], keys[j]))
    pairs.sort(key=lambda t: t[0])
    print(f"Computed {len(pairs)} geodesic distances.")
    return pairs


def calculate_real_variable_length_dtw(dict_raw_features):
    """DTW for variable-length, multi-dim sequences (flattened per frame)."""
    try:
        from dtaidistance import dtw_ndim
    except Exception as e:
        raise RuntimeError("Please `pip install dtaidistance` to compute DTW.") from e

    print("Computing DTW distances on variable-length raw sequences...")
    keys = list(dict_raw_features.keys())
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = dict_raw_features[keys[i]]
            b = dict_raw_features[keys[j]]
            a2 = a.reshape(a.shape[0], -1) if a.ndim > 2 else a
            b2 = b.reshape(b.shape[0], -1) if b.ndim > 2 else b
            a2 = a2.astype(np.float64)
            b2 = b2.astype(np.float64)
            d = dtw_ndim.distance(a2, b2)
            pairs.append((d, keys[i], keys[j]))
    pairs.sort(key=lambda t: t[0])
    print(f"Computed {len(pairs)} DTW distances.")
    return pairs


def _correlate(distances, alphas):
    """
    Compute Pearson, Spearman, and linear fit stats.
    Returns a dict (or None if no valid labels).
    """
    valid = [(d, a) for d, a in zip(distances, alphas) if a is not None]
    if not valid:
        return None
    d, a = zip(*valid)
    d = np.array(d, dtype=float)
    a = np.array(a, dtype=float)

    pear = stats.pearsonr(d, a)
    spear = stats.spearmanr(d, a)
    model = LinearRegression().fit(d.reshape(-1, 1), a)
    r2 = r2_score(a, model.predict(d.reshape(-1, 1)))

    return {
        "pearson":  {"r": float(pear[0]),   "p": float(pear[1])},
        "spearman": {"r": float(spear.correlation), "p": float(spear.pvalue)},
        "linreg":   {"slope": float(model.coef_[0]), "intercept": float(model.intercept_), "r2": float(r2)},
        "n":        int(len(valid)),
    }


# -----------------------
# Triplet ground-truth glue
# -----------------------

def create_triplet_module(anim_name, bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
                          squared_left_right_euc_dist, squared_class_neut_euc_dist, config, valid_indices=None):
    """
    Create TripletMining for this animation; exposes .df_comparisons with human labels.
    """
    return TripletMining(
        bool_drop_neutral_exemplar,
        bool_fixed_neutral_embedding,
        squared_left_right_euc_dist,
        squared_class_neut_euc_dist,
        anim_name,
        config,
        valid_indices=valid_indices
    )


def get_inverse_direct_comparison_value(key1, key2, triplet_module):
    """
    key = (anim_name, effort_tuple). Return 1 - count_normalized for the pair,
    using df_comparisons inside the module. Skip neutral and cross-animation pairs.
    """
    action1, effort1 = key1
    action2, effort2 = key2
    if action1 != action2:
        return None
    if effort1 == (0, 0, 0, 0) or effort2 == (0, 0, 0, 0):
        return None

    df = getattr(triplet_module, "df_comparisons", None)
    if df is None or df.empty:
        return None

    try:
        pair_match = df[df['efforts_tuples'].apply(
            lambda x: set(x) == {effort1, effort2} if isinstance(x, list) else False
        )]
        target = pair_match[(pair_match['selected0'] == 0) & (pair_match['selected1'] == 2)]
        if not target.empty and 'count_normalized' in target.columns:
            val = target['count_normalized'].iloc[0]
            return None if pd.isna(val) else (1.0 - float(val))
    except Exception:
        pass
    return None


def collect_distance_inverse_pairs(distance_tuples, triplet_module):
    """Return (distances, alphas) lists aligned with pairs in distance_tuples."""
    dists, alphas = [], []
    for d, k1, k2 in distance_tuples:
        dists.append(d)
        alphas.append(get_inverse_direct_comparison_value(k1, k2, triplet_module))
    return dists, alphas


# -----------------------
# Main
# -----------------------

def main():
    cfg = Config()
    animations = ["walking", "pointing", "picking"]
    evaluate_only_validation = True

    for anim in animations:
        print("\n" + "=" * 70)
        print(f"PROCESSING: {anim.upper()} (AE embeddings + Triplet ground truth)")
        print("=" * 70)

        # 1) Build val split from RAW similarity dict (as before)
        part = osd.load_similarity_data(True, anim, cfg)  # True = drop neutral exemplar like before
        base = part["train"]
        base.update(part["test"])
        balanced = osd.balance_single_exemplar_similarity_classes_by_frame_count([base], 137)
        train_idx, val_idx = create_train_val_split(balanced)
        valid_indices = val_idx[0] if evaluate_only_validation else None
        subset_name = "VALIDATION SUBSET" if evaluate_only_validation else "ALL DATA"
        print(f"Evaluating on: {subset_name}")

        # 2) Load AE embeddings via SingleEmbeddingDataset from *_encoded_2 dirs
        if anim == "walking":
            emb_dir = "../datasets/lma_perform_walking_ae_combined"
        elif anim == "pointing":
            emb_dir = "../datasets/lma_perform_pointing_ae_combined"
        elif anim == "picking":
            emb_dir = "../datasets/lma_perform_picking_ae_combined"
        else:
            raise ValueError(f"Unknown animation {anim}")

        try:
            ds = SingleEmbeddingDataset(emb_dir)     # filters invalid effort tuples, parses filenames
            emb_sim_dict = ds.to_similarity_dict()   # {effort_tuple: [np_embedding, ...]}
        except Exception as e:
            print(f"[{anim}] ERROR building embedding similarity dict from {emb_dir}: {e}")
            continue

        if evaluate_only_validation and valid_indices is not None:
            before = len(emb_sim_dict)
            emb_sim_dict = {k: v for k, v in emb_sim_dict.items() if k in valid_indices}
            print(f"[{anim}] Filtered embedding classes by validation set: {before} -> {len(emb_sim_dict)}")

        emb_dict = generate_embeddings_without_refinement(emb_sim_dict, anim)

        # 3) Raw features for geodesic/DTW (validation subset to match embeddings)
        raw_dict = get_raw_features_without_dataloader(anim, cfg, valid_indices=valid_indices)
        # For DTW we’ll reuse raw_dict but without balancing if needed later.

        # 4) Build TripletMining to access human alpha ground truths
        triplet_module = create_triplet_module(
            anim_name=anim,
            bool_drop_neutral_exemplar=True,
            bool_fixed_neutral_embedding=True,
            squared_left_right_euc_dist=False,
            squared_class_neut_euc_dist=False,
            config=cfg,
            valid_indices=valid_indices
        )
        if getattr(triplet_module, "df_comparisons", None) is None or triplet_module.df_comparisons.empty:
            print(f"[{anim}] WARNING: Triplet module has no df_comparisons; correlations will be skipped.")

        # 5) Distances
        emb_L2 = calculate_pairwise_distances(emb_dict)      # normalized [0,1]
        geo = compute_geodesic_distances(raw_dict)           # mean geodesic
        dtw = calculate_real_variable_length_dtw(raw_dict)   # DTW

        # 6) Collect alphas from triplet ground truth and correlate (structured)
        emb_d, emb_a = collect_distance_inverse_pairs(emb_L2, triplet_module)
        geo_d, geo_a = collect_distance_inverse_pairs(geo, triplet_module)
        dtw_d, dtw_a = collect_distance_inverse_pairs(dtw, triplet_module)

        emb_stats = _correlate(emb_d, emb_a)
        geo_stats = _correlate(geo_d, geo_a)
        dtw_stats = _correlate(dtw_d, dtw_a)

        # Distance summaries for the pretty report
        emb_summary = _summarize_distances_from_pairs(emb_L2)
        geo_summary = _summarize_distances_from_pairs(geo)
        dtw_summary = _summarize_distances_from_pairs(dtw)

        # 6a) Embedding L2 vs Geodesic (raw)
        render_module_report(
            anim_name=anim,
            subset_name=subset_name,
            method_name_left="Embedding L2",
            method_name_right="Geodesic (raw)",
            corr_left=emb_stats,
            corr_right=geo_stats,
            dist_left=emb_summary,
            dist_right=geo_summary,
            valid_pairs_left=(emb_stats["n"] if emb_stats else 0),
            valid_pairs_right=(geo_stats["n"] if geo_stats else 0),
        )

        # 6b) Embedding L2 vs DTW (raw)
        render_module_report(
            anim_name=anim,
            subset_name=subset_name,
            method_name_left="Embedding L2",
            method_name_right="DTW (raw)",
            corr_left=emb_stats,
            corr_right=dtw_stats,
            dist_left=emb_summary,
            dist_right=dtw_summary,
            valid_pairs_left=(emb_stats["n"] if emb_stats else 0),
            valid_pairs_right=(dtw_stats["n"] if dtw_stats else 0),
        )

    print("\n" + "=" * 70)
    print("EMBEDDING-ONLY ANALYSIS COMPLETE (with Triplet ground-truth integration)")
    print("=" * 70)


if __name__ == "__main__":
    main()
