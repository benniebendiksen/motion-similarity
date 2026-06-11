"""
DTW-fusion regularization for triplet metric learning.

Adds an auxiliary term that pulls the embedding's pairwise L2 distance matrix
toward a frozen, precomputed DTW distance matrix on raw motion features —
*after rank-normalizing both* so only DTW's ORDERING (its temporal-alignment
structure) is transferred, never its scale.

    L_total = L_triplet  +  lambda * L_dtw_align
    L_dtw_align = mean_{i<j} ( rank_norm(D_emb)[i,j] - rank_norm(D_dtw)[i,j] )^2

Design notes
------------
- The DTW matrix is computed once at setup from raw BVH features, indexed in
  the SAME class order the triplet module uses (insertion order of
  dict_similarity_classes_exemplars), frozen, and attached to the module as
  `module.dtw_distance_matrix`.  No gradient flows into it.
- Rank-normalization to [0,1] is applied to the off-diagonal upper triangle of
  both matrices.  Comparing ranks (not magnitudes) is what makes this a fair
  geometric prior rather than distilling DTW's metric.
- lambda (dtw_loss_weight) defaults to 0.0, exactly recovering the pure-triplet
  baseline, so the term is a no-op until explicitly enabled.
"""
from __future__ import annotations

import numpy as np
import torch


def _rank_normalize_offdiag(mat: torch.Tensor) -> torch.Tensor:
    """Rank-normalize the strict upper-triangle of a square distance matrix to
    [0,1] (dense ranks, min->0 max->1). NON-differentiable (uses argsort) —
    only use for the FROZEN DTW matrix at precompute time, never on a tensor
    that must carry gradient.
    """
    n = mat.shape[0]
    iu = torch.triu_indices(n, n, offset=1, device=mat.device)
    vals = mat[iu[0], iu[1]]
    if vals.numel() <= 1:
        return torch.zeros_like(mat)
    order = torch.argsort(vals)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(vals.numel(), dtype=torch.float32, device=mat.device)
    ranks = ranks / max(vals.numel() - 1, 1)
    out = torch.zeros_like(mat)
    out[iu[0], iu[1]] = ranks
    out[iu[1], iu[0]] = ranks
    return out


def _minmax_normalize_offdiag(mat: torch.Tensor) -> torch.Tensor:
    """Differentiable min-max scaling of the strict upper-triangle to [0,1].
    Safe for the embedding-distance side (gradient flows through). Returns a
    symmetric matrix with zero diagonal.
    """
    n = mat.shape[0]
    iu = torch.triu_indices(n, n, offset=1, device=mat.device)
    vals = mat[iu[0], iu[1]]
    if vals.numel() <= 1:
        return torch.zeros_like(mat)
    vmin = vals.min()
    vmax = vals.max()
    scaled = (vals - vmin) / (vmax - vmin + 1e-8)
    out = torch.zeros_like(mat)
    out[iu[0], iu[1]] = scaled
    out[iu[1], iu[0]] = scaled
    return out


def precompute_dtw_matrix(module, raw_features_dict, anim_name=None):
    """Build and attach a frozen, rank-normalized DTW distance matrix to a
    TripletMining module, aligned to its class index order.

    Args:
        module: TripletMining instance (its dict_similarity_classes_exemplars
                defines the class order used everywhere in the loss).
        raw_features_dict: dict mapping effort_tuple -> raw motion array
                [T, J, C] or [T, D].  Keys must be the bare effort tuples
                (no action prefix).
        anim_name: optional, only for logging.

    Side effect:
        Sets module.dtw_distance_matrix to a [N, N] torch.float32 tensor of
        rank-normalized DTW distances (symmetric, zero diagonal), or to None
        if it could not be built (missing features).
    """
    from dtaidistance import dtw_ndim

    class_keys = list(module.dict_similarity_classes_exemplars.keys())
    # Drop the neutral if the module excludes it from the distance matrix.
    # The triplet matrices are sized num_states_drives; align to that by
    # excluding (0,0,0,0) when bool_drop_neutral_exemplar is True.
    if getattr(module, "bool_drop_neutral_exemplar", False) and (0, 0, 0, 0) in class_keys:
        class_keys = [k for k in class_keys if k != (0, 0, 0, 0)]

    n = len(class_keys)
    missing = [k for k in class_keys if k not in raw_features_dict]
    if missing:
        print(f"[dtw_fusion] {anim_name or ''}: {len(missing)} classes missing raw "
              f"features; skipping DTW matrix.")
        module.dtw_distance_matrix = None
        return

    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        mi = raw_features_dict[class_keys[i]]
        mi = mi.reshape(mi.shape[0], -1).astype(np.double) if mi.ndim > 2 else mi.astype(np.double)
        for j in range(i + 1, n):
            mj = raw_features_dict[class_keys[j]]
            mj = mj.reshape(mj.shape[0], -1).astype(np.double) if mj.ndim > 2 else mj.astype(np.double)
            d = dtw_ndim.distance(mi, mj)
            mat[i, j] = d
            mat[j, i] = d

    t = torch.tensor(mat, dtype=torch.float32)
    module.dtw_distance_matrix = _rank_normalize_offdiag(t)
    print(f"[dtw_fusion] {anim_name or ''}: attached rank-normalized DTW matrix "
          f"[{n}x{n}] to module.")


def dtw_alignment_loss(classes_distances: torch.Tensor,
                       dtw_matrix: torch.Tensor) -> torch.Tensor:
    """MSE between rank-normalized embedding distances and the frozen
    rank-normalized DTW matrix, over the strict upper triangle.

    classes_distances flows gradient; dtw_matrix is constant.
    """
    if dtw_matrix is None:
        return classes_distances.new_zeros(())
    n = classes_distances.shape[0]
    if dtw_matrix.shape[0] != n:
        # Shape mismatch (e.g. partial batch) — skip safely.
        return classes_distances.new_zeros(())
    # Embedding side: differentiable min-max (gradient flows).
    # DTW side: frozen rank-normalized target (constant).
    emb_n = _minmax_normalize_offdiag(classes_distances)
    iu = torch.triu_indices(n, n, offset=1, device=classes_distances.device)
    diff = emb_n[iu[0], iu[1]] - dtw_matrix.to(classes_distances.device)[iu[0], iu[1]]
    return (diff * diff).mean()
