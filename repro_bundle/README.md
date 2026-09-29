# Reproducibility Bundle — "Motion Encoding for Human Perceptual Similarity"

Phase-grouped copy of the artifacts and load-bearing code behind every reported finding.
Assembled 2026-07-02, **corrected + de-identified 2026-07-23** from the working trees
(`motion-similarity/` local + `triplets/` on the chimera cluster). All paths/values are
**live-verified** against source (see `../docs/provenance_ledger.md` §0 and `../docs/REPRO_TRUTH.md`).
Large encoder checkpoints are **referenced by path, not copied** (GB-scale); everything needed to
reproduce the reported *numbers* from embeddings/CSVs is here. Reproduction is exact up to the
stochastic bounds documented per stage (fold seeds, fine-tune seeds).

Provenance flows top-down through the numbered folders: raw human data → aggregation → baselines /
encoders → results → perceptual fine-tune → significance tests.

---

## 00_ground_truth/ — the root of d_perc (paper §2.4)

The uttermost source: human study responses (MTurk), and the deterministic aggregation to d_perc.

**Raw responses (DE-IDENTIFIED):** `validAnswers{Walking,Pointing,Picking}_deidentified.csv`
— one row per participant-per-trial (~52k rows total: walk 17,380 / point 17,690 / pick 16,959).
Columns: `qInd, motionType, effortsLeft, effortsRight, selected0, selected1`. The original files
carried `timestamp, id, hitId` (MTurk worker/HIT identifiers); **these three columns are dropped
here** for release. Dropping them is provably inert to every downstream number — see the
round-trip verification below. `selected0/selected1` = the human's chosen most-similar pair.

**Aggregation to d_perc (`aggregation_R/`) — THE actual path:**
- `lmaPilot.R` — the transform that produced the aggregate the paper's code consumes. It reads the
  raw responses, combines effort-pair permutations (`combineRowPermutations`) and normalizes choice
  frequency per pairing → writes `similarity_comparisons_ratios.csv`.
- `{walking,pointing,picking}_similarity_comparisons_ratios.csv` — the aggregate actually read by
  the evaluation code (columns include **`count`, `count_normalized`**). Byte-identical to the
  repo's `aux/*_similarity_comparisons_ratios.csv` (verified 4620/4620 rows).
- **d_perc = 1 − count_normalized[(selected0=0, selected1=2)]** (Left–Right pairing), computed by
  `05_code/cv_preliminary.py: get_inverse_direct_comparison_value` (reads `count_normalized`).
  Neutral-involving pairs → None (excluded). Single ground-truth target for all methods.

**De-identification round-trip (verified 2026-07-23):** running `lmaPilot.R` on the de-identified
raw reproduces the aggregate **IDENTICALLY** (4620/4620 rows, 0 `count_normalized` mismatches).
So the dropped identifier columns do not enter any reported number.

> Note: `ratios{Walk,Point,Pick}.csv` (columns `r01,r02,r12`) are a **secondary Python view** of the
> same choice frequencies (`05_code/readRatiosUserStudy.py`). They are provided for inspection but
> are **not** on the live d_perc path — the evaluation code reads `count_normalized` from the
> `aggregation_R/` aggregate above.

## 01_baselines/ — geometric baselines (paper §5.2, §6.1)
- `baselines_nested_results.json` — {rows{dtw,geo}, aggregate, bars}, n=25 nested-CV. Paper's exact
  baseline numbers: DTW 0.490/0.273/0.303, geodesic 0.371/0.234/0.264.
- Code: `05_code/baselines_nested.py` → `05_code/infer_emb_vec.py`:
  - **DTW** = `calculate_real_variable_length_dtw` → `dtw_ndim.distance` (Euclidean local cost, var-length).
  - **Geodesic** = quaternion geodesic `2·arccos(|⟨q₁,q₂⟩|)` per joint (28 joints), DTW-aligned.
  - Features **unpadded** (`balance_class_frame_counts=False`; real median lengths walk 73 / point 53 / pick 71).

## 02_encoders/ — self-supervised encoders (paper §4)
- `configs/` — training recipes (MAMP motion-only, MAMP+pose headline, MLD-style U-Net, 2×2 cell).
- `encode_mamp_lma.py` — MAMP encode (mask_ratio=0, mean-pool patches → 256-D embedding).
- **Checkpoints NOT copied** (GB-scale). On chimera: `triplets/MAMP{,_uencfull}/output_dir/<run>/
  checkpoint-<epoch>.pth`. Registry + per-fold selection: `05_code/cv_preliminary.py: ENCODERS` + `cv_nested.py`.

## 03_results/ — canonical nested-CV results (paper §6)
- `cvn_*.json` — canonical result tree (n=25, raw_test per fold). Headline `cvn_mamppose.json`
  (0.552/0.434/0.543). Every §6 Spearman traces here. (June-16 tree ARCHIVED — do not use.)
- `wilcoxon_paired_results.json` — paired significance tests (see 05_code/wilcoxon_paired.py below).
- Fold protocol: 5×5, magnitude-stratified, seed **42+1000·repeat**, neutral excluded then re-added;
  per-fold checkpoint selection on the validation fold, reported on the disjoint test fold.

## 04_perceptual/ — ranking fine-tune + TMR comparison (paper §6.3, §6.4)
- `tmr_encoded/*.npy` (171) — TMR (HumanML3D) 256-D embeddings of rated clips, evaluated on the
  identical 25 folds by `05_code/tmr_nested.py`. TMR sees our motion only through a lossy
  BVH→SMPL fit (~4cm MPJPE), so the TMR comparison is conservative (a lower bound on our advantage).
- Ranking fine-tune: `05_code/perc_finetune.py` (anchored triplet, margin = d_perc gap).
- **Symmetric TMR fine-tune** (the fine-tuned-vs-fine-tuned control): `05_code/tmr_perc_finetune.py`
  (+ `tmr_perc_finetune.sbatch`, + `05_code/dperc_folds.py` = dependency-light d_perc/fold helpers
  extracted from cv_preliminary to avoid a `src` namespace clash with the TMR repo). Fine-tunes TMR's
  ACTOR motion encoder with the IDENTICAL rank objective + nested-CV, from the fixed 263-feat guofeats
  (SMPL fit frozen upstream). Results: `03_results/tmr_perc_finetune_results.json` (n=25;
  W/P/K = .649/.573/.611). Comparison to MAMP+pose: raw MAMP+pose wins 2/3; after BOTH receive the
  same fine-tune, MAMP+pose wins walking, ties picking, and TMR wins pointing (.573 vs .463). This is
  the honest matched comparison reported in the paper's TMR subsection (see REPRO_TRUTH §12).
  Runs in a separate `tmr` conda env (hydra/pytorch-lightning/transformers); checkpoints referenced on
  chimera (`learned_baselines/tmr/models/.../last_weights/motion_encoder.pt`), not copied.

## 05_code/ — load-bearing analysis code
`cv_preliminary.py` (d_perc, ENCODERS, raw_spearman/raw_both), `cv_nested.py` (nested-CV harness),
`baselines_nested.py` + `infer_emb_vec.py` (baselines), `perc_finetune.py` (rank fine-tune),
`tmr_nested.py` (TMR eval), `kinematic_descriptors.py`, `readRatiosUserStudy.py` (secondary ratios),
`encode_mamp_lma.py` (under 02_encoders/), and:
- **`wilcoxon_paired.py`** — one-sided paired Wilcoxon signed-rank over the n=25 matched folds
  (pairing by (repeat,fold)); tests MAMP+pose vs each baseline/encoder per action. Pure-stdlib
  implementation (validated on tie/constant-shift/null cases); writes `03_results/wilcoxon_paired_results.json`.
- **`wilcoxon_mamp_vs_tmr.py`** — TWO-sided paired Wilcoxon, MAMP+pose vs TMR in both regimes (raw,
  and after identical perceptual fine-tune), per action. Establishes which cells are significant wins
  vs statistical ties. Sources: `cvn_mamppose_cut1.json` (raw MAMP+pose), `tmr_raw_perfold.json` (raw
  TMR, per-fold), `rankdiag_headline.json['B']` (fine-tuned MAMP+pose), `tmr_perc_finetune_results.json`
  (fine-tuned TMR). Writes `03_results/wilcoxon_mamp_vs_tmr_results.json`. Result: RAW = picking win
  (p=.011), walking tie (p=.85), pointing loss (p=.007); FINE-TUNED = walking win (p=.026), picking tie
  (p=.98), pointing loss (p=.003). This backs the paper's "matches or exceeds on the two kinematically
  rich actions" phrasing.

---

**Not included (reference on chimera):** encoder checkpoints (`.pth`), sweep/scratch dirs, slurm logs.
**Ethics / human subjects:** released response CSVs are **de-identified** (MTurk `id`/`hitId` and
`timestamp` removed; removal verified inert). No participant-identifying information is shipped.
