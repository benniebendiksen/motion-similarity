#!/usr/bin/env python3
"""Compare the AE and the VAE on TRULY out-of-sample motion.

The 13 walks in `viz_ae_vae_ood/sources_cmu33/` were retargeted from the
LMA-perform skeleton to CMU-33 expressly because they have no equivalent in
either model's training corpus. This probe runs both checkpoints on the same
13 sources and writes source + AE_recon + VAE_recon BVH triplets, plus a
manifest with per-model per-clip MPJPE.

Run on chimera (both checkpoints already there); pull viz_ae_vae_ood/ back
locally to render 3-pane comparison mp4s.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from autoencoders.AEMotionMLD import AEMotionMLD
from autoencoders.VAEMotionMLD import VAEMotionMLD
from autoencoders.features_to_bvh import features_to_bvh
from autoencoders.fk import features_to_joint_positions
from common.MotionDataset import extract_root_and_rotations
from common.norm_utils import denormalize, load_norm_stats
from bvhReader.bvh import BVH


def _load_ae(ckpt_path: Path, device: torch.device) -> AEMotionMLD:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    model.load_state_dict(state["model"])
    return model, state.get("epoch", -1) + 1


def _load_vae(ckpt_path: Path, device: torch.device) -> VAEMotionMLD:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["config"]
    model = VAEMotionMLD(
        feature_dim=204, window_size=cfg["window_size"],
        hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"],
        num_latent_tokens=cfg["num_latent_tokens"],
        num_heads=cfg["num_heads"],
        num_encoder_layers=cfg["num_encoder_layers"],
        num_decoder_layers=cfg["num_decoder_layers"],
        dropout=cfg["dropout"],
    ).to(device).eval()
    model.load_state_dict(state["model"])
    return model, state.get("epoch", -1) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae-ckpt",  default="checkpoints_ae_mld/v0_ae_seed0/best.pt")
    ap.add_argument("--vae-ckpt", default="checkpoints_vae_mld/v0_mld_seed2/best.pt")
    ap.add_argument("--src-dir",  default="viz_ae_vae_ood/sources_cmu33")
    ap.add_argument("--out-dir",  default="viz_ae_vae_ood")
    args = ap.parse_args()

    out_dir = _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load both models ----
    ae,  ae_epoch  = _load_ae (_ROOT / args.ae_ckpt,  device)
    vae, vae_epoch = _load_vae(_ROOT / args.vae_ckpt, device)
    print(f"AE  @ epoch {ae_epoch}")
    print(f"VAE @ epoch {vae_epoch}")

    norm_stats = load_norm_stats(_ROOT / "datasets" / "norm_stats.npz")
    mean = norm_stats["mean"].astype(np.float32)
    std  = norm_stats["std"].astype(np.float32)
    WINDOW = 120

    # ---- Find OOD sources ----
    src_dir = _ROOT / args.src_dir
    sources = sorted(src_dir.glob("Walking_*.bvh"))
    if not sources:
        sys.exit(f"[err] no source BVHs at {src_dir}")

    print(f"\n{len(sources)} OOD sources from {src_dir.relative_to(_ROOT)}/")

    manifest = [
        "# AE vs VAE — held-out OOD reconstruction comparison",
        f"# AE ckpt:  {args.ae_ckpt}  (epoch {ae_epoch})",
        f"# VAE ckpt: {args.vae_ckpt} (epoch {vae_epoch})",
        "",
        f"{'tag':30s}  {'AE_mpjpe':>8s}  {'VAE_mpjpe':>9s}  source",
        "-" * 90,
    ]
    ae_mpjpes  = []
    vae_mpjpes = []

    for src_path in sources:
        # 8 = len('Walking_'), -4 = len('.bvh')
        effort = src_path.stem[len("Walking_"):]
        tag = f"Walking_{effort}"

        # Load + window + normalize
        b = BVH(); b.load(str(src_path))
        raw = extract_root_and_rotations(b).astype(np.float32)
        if len(raw) >= WINDOW:
            raw_w = raw[:WINDOW]
        else:
            pad = np.tile(raw[-1:], (WINDOW - len(raw), 1, 1))
            raw_w = np.concatenate([raw, pad], axis=0)
        x_norm = (raw_w - mean) / std
        np.clip(x_norm, -10.0, 10.0, out=x_norm)
        x_t = torch.from_numpy(x_norm).unsqueeze(0).to(device)

        # AE forward
        with torch.no_grad():
            x_hat_ae, _ = ae(x_t)
            x_hat_ae_dn = denormalize(x_hat_ae, norm_stats)

        # VAE forward (uses mu deterministically in eval)
        with torch.no_grad():
            x_hat_vae, mu_v, _, _ = vae(x_t)
            x_hat_vae_dn = denormalize(x_hat_vae, norm_stats)

        # Save source (windowed raw), AE recon, VAE recon
        src_dn = torch.from_numpy(raw_w).to(device)
        features_to_bvh(src_dn,           out_path=out_dir / f"{tag}_source.bvh")
        features_to_bvh(x_hat_ae_dn[0],   out_path=out_dir / f"{tag}_AE_recon.bvh")
        features_to_bvh(x_hat_vae_dn[0],  out_path=out_dir / f"{tag}_VAE_recon.bvh")

        # MPJPE per model
        with torch.no_grad():
            pos_src = features_to_joint_positions(src_dn)
            pos_ae  = features_to_joint_positions(x_hat_ae_dn[0])
            pos_vae = features_to_joint_positions(x_hat_vae_dn[0])
        ae_mpjpe  = (pos_src - pos_ae ).norm(dim=-1).mean().item()
        vae_mpjpe = (pos_src - pos_vae).norm(dim=-1).mean().item()
        ae_mpjpes.append(ae_mpjpe)
        vae_mpjpes.append(vae_mpjpe)

        rel = src_path.relative_to(_ROOT)
        manifest.append(f"{tag:30s}  {ae_mpjpe:>8.3f}  {vae_mpjpe:>9.3f}  {rel}")
        print(f"  {tag:30s}  AE={ae_mpjpe:.3f}  VAE={vae_mpjpe:.3f}")

    ae_arr  = np.array(ae_mpjpes)
    vae_arr = np.array(vae_mpjpes)
    manifest += [
        "",
        f"# AE  MPJPE  cm  min/median/mean/max = "
        f"{ae_arr.min():.3f} / {np.median(ae_arr):.3f} / "
        f"{ae_arr.mean():.3f} / {ae_arr.max():.3f}",
        f"# VAE MPJPE  cm  min/median/mean/max = "
        f"{vae_arr.min():.3f} / {np.median(vae_arr):.3f} / "
        f"{vae_arr.mean():.3f} / {vae_arr.max():.3f}",
        f"# AE - VAE  cm  mean delta (negative = AE better) = "
        f"{(ae_arr - vae_arr).mean():+.3f}",
    ]
    (out_dir / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")
    print(f"\nAE  median MPJPE: {np.median(ae_arr):.3f} cm")
    print(f"VAE median MPJPE: {np.median(vae_arr):.3f} cm")
    print(f"Wrote {3 * len(sources)} BVHs + MANIFEST.txt to {out_dir}/")


if __name__ == "__main__":
    main()
