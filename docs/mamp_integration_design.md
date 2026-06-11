# MAMP Integration & Encoder Training Slate — Design

## Goal
A single **held-out** encoder whose embeddings beat DTW (and geodesic) on BOTH
walking and picking perceptual-similarity Spearman. Current held-out state:
- AE-holdout (plain-MSE AEMotionMLD): picking **+0.476** ✓ DTW, walking +0.435 (−0.043 short)
- PROV (AETransformer, rotMSE20+vel100): walking **+0.499** ✓ DTW, picking +0.289
Mirror-image specialists. No single held-out encoder wins both.

## What we have RULED OUT (don't repeat)
- **Velocity-as-loss does not close walking.** velFT = AEMotionMLD + velocity_weight=1.0,
  warm-started from AE-holdout. It scored worse than plain AE-holdout on BOTH actions
  (walk 0.409 vs 0.435, pick 0.394 vs 0.476). Velocity diluted pose fidelity without
  buying enough dynamics. PROV's walking win comes from velocity-100x + rotMSE-20x +
  AETransformer + from-scratch — NOT velocity alone.
- **DTW-fusion lambda is inert** (proven on fixed split; earlier λ effects were split noise).
- **Windowing** helps walking for some encoders, hurts picking; pooled best for AE-holdout.

## Why MAMP is mechanistically different from velocity loss
Velocity loss penalizes deltas of a model that SEES the full clip → deltas come "for free"
from accurate pose copying → little new learning pressure (why velFT was inert).
MAMP MASKS 80% of patches and PREDICTS the MOTION of masked, high-dynamics regions from
sparse visible context → the encoder MUST learn dynamics to succeed. Motion-guided masking
(Gumbel-softmax over per-patch motion magnitude) concentrates difficulty on the
semantically rich temporal regions. Primary target is motion (x[t+s]-x[t]), not pose.

## MAMP source facts (model_mamp/transformer.py)
- Input `[N,C,T,V,M]`; patchify with patch_size=1 (joints), t_patch_size=4 (frames).
- `motion_aware_random_masking` (L276): motion=|x[t]-x[t-1]| mean over coords → Gumbel
  prior → mask high-motion patches. mask_ratio=0.80.
- `extract_motion` (L426): target = x[t+stride]-x[t].
- `forward_loss` (L437): MSE(pred, motion) on MASKED patches only.
- Downstream head = mean-pool encoder patches → linear classifier (action recognition,
  NTU-60/120). For us: replace classifier with our similarity pipeline; the linprobe head
  IS already `feat.mean(dim=[1,2,3])` → so mean-pool is MAMP-native.

## Data-format adaptation (NTU → our LMA)
| MAMP | Ours | Change |
|------|------|--------|
| C=3 (xyz), M persons | 6-D rotations, M=1 | dim_in=6, collapse M |
| V=25 joints | V=34 | num_joints=34 |
| T=120, t_patch=4 → 30 temporal patches | T=120 | unchanged ✓ |
| NTU feeder | extract_root_and_rotations + norm_stats.npz, clip ±10 | new feeder wrapping MotionDataset |
| trains on full NTU | HELD-OUT: exclude 342 LMA states/drives | reuse collect_lma_held_out_paths blacklist |
motion/patchify logic is coordinate-agnostic differencing → works on 6-D as-is.

## Per-clip embedding extraction
Pretrained MAMP encoder outputs `[N, TP*VP, dim]` (TP=30, VP=34). For our similarity
pipeline we need ONE vector/clip:
- **mean-pool** over patches (MAMP-native linprobe pooling) — DEFAULT.
- **attention-pool** (AETransformer-style learned query) — VARIANT, more expressive.

## Training slate (mechanistically distinct, informed by observations)
Each pretrained HELD-OUT (342 excluded), then embeddings → fixed-split (seed 42) eval.

1. **MAMP-meanpool** — masked motion prediction + mean-pool. The principled walking play.
2. **MAMP-attnpool** — same pretrain, attention-pool head. Tests whether richer pooling
   helps (AETransformer's pooling was part of PROV's strength).
3. **(reference) AE-holdout** — already have it; the picking champion, plain-MSE.
   No retrain; include as the baseline both MAMP variants must beat on picking while
   closing walking.

NOTE: We are NOT adding another velocity-loss AE — velFT already ran that and it failed.
If a "hybrid" is wanted, the right one is MAMP (motion prediction) + an auxiliary
pose-reconstruction term (to retain the picking-critical pose fidelity) — a candidate
4th run ONLY if MAMP-meanpool shows walking promise but regresses picking:

4. **MAMP+pose-recon (conditional)** — add small MSE pose-reconstruction head alongside
   the motion-prediction loss, to keep picking's pose fidelity while gaining walking's
   dynamics. Run only if (1)/(2) trade walking-for-picking.

## Open design choices / risks
- MAMP shapes representations for action RECOGNITION; transfer to perceptual-similarity
  REGRESSION (Spearman vs human ratings) is the empirical bet — plausible (dynamics
  sensitivity) but unproven in our regime.
- mask_ratio (0.80), motion_aware_tau (0.75), t_patch_size (4) are NTU-tuned; may need
  adjustment for 120-frame LMA clips with long static settle regions.
- Encoder depth/dim: start near MAMP defaults (dim_feat 256, depth 5) to match our
  latent-256 family for comparability.
