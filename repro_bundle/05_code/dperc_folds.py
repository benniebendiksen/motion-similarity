#!/usr/bin/env python3
"""
Dependency-light extraction of the d_perc lookup + fold construction from cv_preliminary.py.

WHY THIS EXISTS: cv_preliminary imports run_enhanced_embedding_triplet_training -> src.* (a bare
`src` namespace package). TMR's repo ALSO uses a bare `src` package. Importing both in one process
collides on the `src` name (ModuleNotFoundError: No module named 'src.load'). The TMR perceptual
fine-tune (tmr_perc_finetune.py) needs only these four helpers, none of which touch the heavy
`src` chain -- so we lift them here with a minimal dependency set (pandas + Config + TripletMining),
letting the TMR fine-tune import THIS instead of cv_preliminary. Logic is byte-faithful to
cv_preliminary (verified equal on the same modules).
"""
import random
import pandas as pd
from Config import Config  # noqa: F401  (kept for callers that pass Config())
from networks.triplet_mining import TripletMining

ACTIONS = ["walking", "pointing", "picking"]
ACTION_CAP = {"walking": "Walking", "pointing": "Pointing", "picking": "Picking"}


def create_triplet_module(anim_name, bool_drop, bool_fixed, sq_lr, sq_cn, config,
                          valid_indices=None):
    """Inlined from cv_preliminary (which inlined it from embedding_inference_autoencoder to
    dodge that module's unguarded `import tensorflow`)."""
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


def get_directed_alpha(key_left, key_right, triplet_modules):
    if not isinstance(triplet_modules, list):
        triplet_modules = [triplet_modules]
    action1, effort1 = key_left
    action2, effort2 = key_right
    if effort1 == (0, 0, 0, 0) or effort2 == (0, 0, 0, 0) or action1 != action2:
        return None
    for module in triplet_modules:
        if module.anim_name != action1:
            continue
        df = getattr(module, "alpha_dataframes", None)
        if df is None or "alpha_0_2" not in df.columns:
            return None
        pair = df[df["efforts_tuples"].apply(
            lambda x: set(x) == set([effort1, effort2]) if isinstance(x, list) else False)]
        if pair.empty:
            return None
        row = pair.iloc[0]
        stored = list(row["efforts_tuples"])
        a02, a20 = float(row["alpha_0_2"]), float(row["alpha_2_0"])
        if stored[0] == effort1:
            return (a02, a20)
        return (a20, a02)
    return None


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


def _module_keys(module):
    if hasattr(module, "exemplar_dict") and module.exemplar_dict:
        return list(module.exemplar_dict.keys())
    if hasattr(module, "df_comparisons") and module.df_comparisons is not None:
        ks = set()
        for tup in module.df_comparisons["efforts_tuples"]:
            if isinstance(tup, list):
                ks.update(tup)
        return list(ks)
    return []


def load_action_keys(action, config):
    """All effort keys for an action, via a triplet module over the full set."""
    m = create_triplet_module(action, False, True, False, False, config, valid_indices=None)
    return m, sorted(k for k in _module_keys(m) if k != (0, 0, 0, 0))
