# Reproducibility Bundle — "Motion Encoding for Human Perceptual Similarity"

Phase-grouped copy of the artifacts and load-bearing code behind every reported finding.
Assembled 2026-07-02 from the working trees (`motion-similarity/` local + `triplets/` on the
chimera cluster). All paths/values here are **live-verified** against source (see
`../docs/provenance_ledger.md`, Section 0). Large encoder checkpoints are **referenced by path,
not copied** (GB-scale); everything needed to reproduce the *numbers* from embeddings/CSVs is here.

Provenance flows top-down through the numbered folders: raw human data → baselines/encoders →
results → perceptual fine-tune.

---

## 00_ground_truth/ — the root of d_perc (paper §2.4)
The uttermost source: raw human study responses. Everything else is deterministic aggregation.
- `validAnswers{Walking,Pointing,Picking}.csv` — ~52k rows total (walk 17,381 / point 17,691 /
  pick 16,960), **one row per participant-per-trial**. Cols: `timestamp, id (MTurk worker),
  hitId, qInd, effortsLeft, effortsRight, selected0, selected1`. `selected0/selected1` = the
  human's chosen most-similar pair. **Nothing is computed above this file.**
- `ratios{Walk,Point,Pick}.csv` — per-triplet `count_normalized` (choice frequency per pairing),
  produced by `05_code/readRatiosUserStudy.py: computeRatios()` (pure Python; **no R in the path**).
- **d_perc = 1 − count_normalized[(selected0=0, selected1=2)]** (Left–Right pairing), computed by
  `05_code/cv_preliminary.py: get_inverse_direct_comparison_value`. Neutral-involving pairs → None
  (excluded). This is the single ground-truth target all methods are correlated against.

## 01_baselines/ — geometric baselines (paper §5.2, §6.1)
- `baselines_nested_results.json` — {rows{dtw,geo}, aggregate, bars}, n=25 nested-CV. **These are
  the paper's exact baseline numbers**: DTW 0.490/0.273/0.303, geodesic 0.371/0.234/0.264.
- Code: `05_code/baselines_nested.py` → imports `05_code/infer_emb_vec.py`:
  - **DTW** = `calculate_real_variable_length_dtw` → `dtw_ndim.distance` (**Euclidean** local cost,
    variable-length).
  - **Geodesic** = `compute_geodesic_distances_dtw` → **quaternion geodesic**
    `2·arccos(|⟨q₁,q₂⟩|)` per joint (28 joints × 4 = 112 ch), DTW-aligned, length-normalized.
  - Features are **unpadded** (harness calls loader with `balance_class_frame_counts=False` →
    real lengths median walk 73 / point 53 / pick 71). NOTE: the loader default `True` pads to 137.

## 02_encoders/ — self-supervised encoders (paper §4)
- `configs/` — training recipes: `lma_mamp_pretrain.yaml` (motion-only MAMP),
  `lma_mamp_pose_pretrain.yaml` (MAMP+pose, headline), `lma_mamp_uencfull_pretrain.yaml`
  (MLD-style U-Net encoder), `lma_mamp_uencfullpose_pretrain.yaml` (the 2×2 cell, pending).
- `encode_mamp_lma.py` — MAMP encode (mask_ratio=0, mean-pool 1020 patches → 256-D embedding).
- **Checkpoints NOT copied** (GB-scale). On chimera: `triplets/MAMP{,_uencfull}/output_dir/<run>/
  checkpoint-<epoch>.pth`. Encoder registry + per-fold checkpoint selection: `05_code/
  cv_preliminary.py: ENCODERS` + `05_code/cv_nested.py`.

## 03_results/ — canonical nested-CV results (paper §6)
- `cvn_*.json` (24 files) — **the canonical result tree** (n=25, raw_test per fold). Every §6
  Spearman traces here. Key rows: `cvn_mamppose.json` (headline 0.552/0.434/0.543),
  `cvn_mamppose_cut1.json` (TMR comparator 0.462/0.302/0.481), `cvn_mampuencfull.json`
  (0.470/0.275/0.566), plus vanilla/MLD/MAMP sweep and the perceptual `perc_cut1_*` /
  `rankdiag_*` runs. (Superseded June-16 `motion-similarity/cvn_*_results.json` tree is
  ARCHIVED — do not use.)
- Fold protocol: 5×5, magnitude-stratified, seed **42+1000·repeat**, neutral excluded then
  re-added; per-fold checkpoint selection on the **validation fold** only, reported on the
  disjoint **test fold** (`05_code/cv_nested.py`).

## 04_perceptual/ — ranking fine-tune + TMR comparison (paper §6.3, §6.4)
- `tmr_encoded/*.npy` (171) — TMR (HumanML3D) embeddings of the rated clips, evaluated on the
  identical 25 folds by `05_code/tmr_nested.py` (prints TMR 0.452/0.425/0.408).
- Ranking fine-tune: `05_code/perc_finetune.py` (`_triplet_rank`, anchored, margin = d_perc gap;
  `--no-early-stop` single-selection, fixed 200 epochs). Headline result +plateau trajectory:
  `03_results/rankdiag_headline.json` (0.699/0.463/0.605) and `perc_cut1_rank_noES.json`.

## 05_code/ — load-bearing analysis code
`readRatiosUserStudy.py` (ratios), `cv_preliminary.py` (d_perc, ENCODERS, encode dispatch,
raw_spearman), `cv_nested.py` (nested-CV harness), `baselines_nested.py` + `infer_emb_vec.py`
(baselines), `perc_finetune.py` (rank fine-tune), `tmr_nested.py` (TMR eval), `encode_mamp_lma.py`
(under 02_encoders/).

---

**Not included (reference on chimera):** encoder checkpoints (`.pth`), sweep/scratch dirs, slurm
logs. **Ethics:** ground-truth CSVs contain MTurk worker IDs → anonymize + add IRB/human-subjects
statement for submission. **Canonical-run note:** headline rank appears as .463 (run 929964) vs
.456 (perc_cut1_rank_noES) — pick one canonical run before final camera-ready.
