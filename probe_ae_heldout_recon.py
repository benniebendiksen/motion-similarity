#!/usr/bin/env python3
"""Reconstruct a handful of held-out LMA clips with the current AE checkpoint.

Picks one representative clip per (action, mirror, category) where
   action   ∈ {Walking, Picking, Pointing}
   mirror   ∈ {non-mirror, Mirror}
   category ∈ {neutral, state, drive}

→ 3 × 2 × 3 = 18 clips total. Per clip we:
   - load the source BVH
   - normalize features → encode → decode → denormalize
   - write `<stem>_source.bvh` and `<stem>_recon.bvh`
   - record per-clip reconstruction MPJPE

Outputs land in viz_ae_heldout_recon/. Pull locally and render 2-pane mp4s.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from autoencoders.AEMotionMLD import AEMotionMLD
from autoencoders.features_to_bvh import features_to_bvh
from autoencoders.fk import features_to_joint_positions
from common.MotionDataset import extract_root_and_rotations
from common.lma_holdout import is_lma_held_out
from common.norm_utils import denormalize, load_norm_stats
from bvhReader.bvh import BVH


ACTIONS  = ["Walking", "Picking", "Pointing"]
MIRRORS  = ["", "Mirror"]   # "" = non-mirror, "Mirror" = mirror variant


def category_of(stem: str) -> str:
    """neutral / state / drive based on LMA non-zero count."""
    parts = stem.split("_")
    efforts = [int(x) for x in parts[-4:]]
    n = sum(1 for e in efforts if e != 0)
    return {0: "neutral", 2: "state", 3: "drive"}.get(n, "other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_ae_mld/v0_ae_seed0/best.pt")
    ap.add_argument("--lma-dir", default="datasets/lma_perform_reorganized_cmu")
    ap.add_argument("--out-dir", default="viz_ae_heldout_recon")
    ap.add_argument("--rng-seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load model ----
    state = torch.load(_ROOT / args.ckpt, map_location=device, weights_only=False)
    cfg = state["config"]
    model = AEMotionMLD(
        feature_dim=204, window_size=cfg["window_size"],
        hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"],
        num_latent_tokens=cfg["num_latent_tokens"],
        num_heads=cfg["num_heads"],
        num_encoder_layers=cfg["num_encoder_layers"],
        num_decoder_layers=cfg["num_decoder_layers"],
        dropout=cfg["dropout"],
    ).to(device).eval()
    saved_epoch = state.get("epoch", -1) + 1
    print(f"Loaded ckpt @ epoch {saved_epoch}, best_val = {state.get('best_val', float('nan')):.5f}")

    norm_stats = load_norm_stats(_ROOT / "datasets" / "norm_stats.npz")
    mean = norm_stats["mean"].astype(np.float32)
    std  = norm_stats["std"].astype(np.float32)
    WINDOW = cfg["window_size"]

    # ---- Enumerate held-out files, group by (action, mirror, category) ----
    lma = _ROOT / args.lma_dir
    buckets: dict = defaultdict(list)
    for bvh in sorted(lma.glob("*.bvh")):
        stem = bvh.stem
        if not is_lma_held_out(stem):
            continue
        action_with_mirror = stem.rsplit("_", 4)[0]   # e.g., "WalkingMirror"
        # Split action / mirror
        if action_with_mirror.endswith("Mirror"):
            action = action_with_mirror[:-len("Mirror")]
            mirror = "Mirror"
        else:
            action = action_with_mirror
            mirror = ""
        if action not in ACTIONS:
            continue
        cat = category_of(stem)
        buckets[(action, mirror, cat)].append(bvh)

    # ---- Deterministic representative pick per bucket ----
    rng = np.random.default_rng(args.rng_seed)
    picks: list[tuple[str, str, str, Path]] = []
    for action in ACTIONS:
        for mirror in MIRRORS:
            for cat in ("neutral", "state", "drive"):
                bucket = buckets.get((action, mirror, cat), [])
                if not bucket:
                    print(f"  [skip] no clips for ({action}, {mirror or 'noMirror'}, {cat})")
                    continue
                # Deterministic pick — index = rng.integers but reproducible
                idx = int(rng.integers(0, len(bucket)))
                picks.append((action, mirror, cat, bucket[idx]))

    print(f"\nPicked {len(picks)} held-out clips for reconstruction.")

    # ---- Reconstruct each + write source/recon BVHs + MPJPE ----
    manifest = [
        f"# AE held-out reconstruction QC",
        f"# ckpt = {args.ckpt}  saved_epoch = {saved_epoch}",
        f"# best_val = {state.get('best_val', float('nan')):.5f}",
        "",
        f"{'tag':40s}  {'category':9s}  {'mpjpe_cm':>9s}  source_clip",
        "-" * 100,
    ]
    mpjpe_list = []
    for i, (action, mirror, cat, src_path) in enumerate(picks):
        # Load + center on a deterministic 120-frame window (first WINDOW frames,
        # padding with last-frame if shorter).
        b = BVH(); b.load(str(src_path))
        raw = extract_root_and_rotations(b).astype(np.float32)
        if len(raw) >= WINDOW:
            raw_w = raw[:WINDOW]
        else:
            pad = np.tile(raw[-1:], (WINDOW - len(raw), 1, 1))
            raw_w = np.concatenate([raw, pad], axis=0)

        # Normalize + ±10 clip (mirrors MotionDataset.__getitem__)
        x_norm = (raw_w - mean) / std
        np.clip(x_norm, -10.0, 10.0, out=x_norm)
        x_t = torch.from_numpy(x_norm).unsqueeze(0).to(device)

        with torch.no_grad():
            x_hat, z = model(x_t)
            x_hat_dn = denormalize(x_hat, norm_stats)

        # The source we save for comparison should be the same WINDOW-frame
        # slice (so source.bvh and recon.bvh are aligned in time).
        src_dn = torch.from_numpy(raw_w).to(device)
        with torch.no_grad():
            pos_src   = features_to_joint_positions(src_dn)
            pos_recon = features_to_joint_positions(x_hat_dn[0])
        mpjpe_cm = (pos_src - pos_recon).norm(dim=-1).mean().item()
        mpjpe_list.append(mpjpe_cm)

        tag = f"{i:02d}_{action}{mirror}_{cat}"
        # Save source BVH (the windowed raw features re-written)
        features_to_bvh(src_dn, out_path=out_dir / f"{tag}_source.bvh")
        features_to_bvh(x_hat_dn[0], out_path=out_dir / f"{tag}_recon.bvh")

        rel = src_path.relative_to(_ROOT)
        manifest.append(f"{tag:40s}  {cat:9s}  {mpjpe_cm:9.3f}  {rel}")
        print(f"  done {tag}  mpjpe={mpjpe_cm:.3f} cm  ({src_path.name})")

    arr = np.array(mpjpe_list)
    manifest += [
        "",
        f"# MPJPE summary (cm):",
        f"#   min/median/mean/max = "
        f"{arr.min():.3f} / {np.median(arr):.3f} / {arr.mean():.3f} / {arr.max():.3f}",
    ]
    (out_dir / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")
    print(f"\nWrote {2 * len(picks)} BVHs + MANIFEST.txt to {out_dir}/")
    print(f"MPJPE median: {np.median(arr):.3f} cm   max: {arr.max():.3f} cm")


if __name__ == "__main__":
    main()
