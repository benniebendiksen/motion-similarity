# Provenance Ledger — paper_draft.md

**Purpose.** Every quantitative result and load-bearing factual claim in the paper, mapped to its
source artifact and a verification status, so authors/reviewers scrutinize *conclusions* on a
verified *source* foundation.

**Status legend.**
- ✅ **VERIFIED** — traced to a source file this session and value confirmed.
- 🔷 **TRACEABLE** — a source artifact/script exists on disk that produces it, but the value was
  not independently re-derived this session (re-runnable to verify).
- ⚠️ **INHERITED** — carried from the prior draft / memory; source not yet located or re-checked.
- 📚 **CITATION** — external claim about a published work; verify against the primary source.
- ⏳ **PENDING** — result of a run still in progress.

Canonical result tree: `triplets/cvn_results/*.json` (Jun 23+). Old `motion-similarity/cvn_*_results.json`
(Jun 16) is ARCHIVED/superseded — never cite. (See memory `reference_canonical_results_tree`.)

Paths are on chimera under `/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/`.
Short roots: `MS/ = motion-similarity/`, `TR/ = triplets/`.

---

## 0. METHODOLOGY → CODE → ARTIFACT (the pipeline behind every finding)

Each row: the paper's *methodological claim* → the exact *code path* that implements it → the
*artifact* it produces → status. Live-verified against chimera code 2026-07-02 unless noted.

### 0.0 GROUND-TRUTH ROOT — human study data behind d_perc (traced 2026-07-02)

The uttermost source of d_perc is the raw human study responses. Full chain, all live-verified:

| Stage | File / code | What it holds/does | Status |
|---|---|---|---|
| **ROOT: raw human trials** | `TR/userStudyResults/validAnswers{Walking,Pointing,Picking}.csv` | ~52k rows total (walk 17,381 / point 17,691 / pick 16,960), ONE row per participant-per-trial. Cols: `timestamp, id (MTurk worker), hitId, qInd, effortsLeft, effortsRight, selected0, selected1`. `selected0/selected1` = the human's chosen most-similar pair. NOTHING computed above this. | ✅ verified |
| aggregate → ratios | `TR/simpleTriplet/readRatiosUserStudy.py: computeRatios(inFile,outFile)` | groups trials by triplet, counts pairing choices (r01/r02/r12), normalizes each by row sum → `count_normalized`. **Pure Python — NO R file in the path.** | ✅ verified |
| ratios file | `TR/userStudyResults/ratios{Walk,Point,Pick}.csv` (+ allRatios.csv) | per-triplet `count_normalized` (choice frequency per pairing) | ✅ verified |
| load | `MS/networks/triplet_mining.py: TripletMining` (pickle/preloaded dict w/ count_normalized col) | reads count_normalized into the module | ✅ verified |
| → d_perc | `MS/cv_preliminary.py:66-69 get_inverse_direct_comparison_value` | `1 - count_normalized[(selected0=0, selected1=2)]`; None if either effort neutral | ✅ verified |

- **d_perc is a deterministic aggregation of ~52k raw human choices** — no learned/stochastic step in the ground truth.
- **NO R FILE** is in the chimera computational path (user recalled one; find turned up none near the data — any R step was a LOCAL pre-filter that produced validAnswers*.csv from raw MTurk export; validAnswers = the defensible root).
- "median 11 raters/trial, 1,540 trials/action" (§2.4) should be re-derived from these CSVs to close that ledger item (§D).
- MTurk worker IDs present → note human-subjects/IRB + anonymization for the reproducibility/ethics statement.

### 0.1 Core evaluation primitives (every number depends on these)

| Paper claim (§) | Code path (file : function / lines) | Verifies | Status |
|---|---|---|---|
| Folds: 5×5, magnitude-stratified, seed **42+1000·repeat**, neutral excluded then re-added (§5.2, §8.1) | `MS/cv_nested.py:112` calls `cv.make_folds(all_keys[a], k, seed+1000*rep)`; `MS/cv_preliminary.py:242 make_folds` ("magnitude-stratified K-fold; neutral excluded; callers re-add") | fold seed + stratification + neutral handling | ✅ live-verified |
| Ground-truth **d_perc = 1 − count_normalized[(selected0=0, selected1=2)]** (§2.4.2) | `MS/cv_preliminary.py:50 get_inverse_direct_comparison_value`, lines 66–69: `1 - tgt["count_normalized"]` where `selected0==0 & selected1==2`; returns None if either effort neutral | d_perc formula + Left–Right(0,2) + neutral-excluded | ✅ live-verified |
| Eval Spearman over **non-neutral pairs only** (§2.4.2/§5.2) | `MS/cv_preliminary.py:~79 raw_spearman`: `continue` past neutral key / None-dp pairs, then `spearmanr` | neutral clip never enters a scored pair | ✅ live-verified |
| Per-fold checkpoint selection on validation fold only; TEST reported (§5.2) | `MS/cv_nested.py` outer loop (test=fold f, val=(f+1)%5, train=rest); selection by raw val-Spearman | 3-way disjoint split, leakage-clean | ✅ live-verified (structure) |
| Encoder registry / encode dispatch per family (mamp/mld/vanilla) (§6.7) | `MS/cv_preliminary.py:110+ ENCODERS` dict → `encode()` dispatch: mamp→`{code_dir}/encode_mamp_lma.py`; mld→`probes/encode_sweep.py`; vanilla→`probes/encode_aetransformer.py` | which ckpt/script produced each encoder's embeddings | ✅ live-verified |

### 0.2 Baselines (audited 2026-07-02 — see [[project_geodesic_baseline_provenance]])

| Paper claim (§5.2) | Code path | Verifies | Status |
|---|---|---|---|
| Geodesic = **quaternion geodesic** 2·arccos(\|⟨q₁,q₂⟩⟩\|), DTW-aligned, unpadded | `MS/pipelines/infer_emb_vec.py: compute_geodesic_distances_dtw` (reshape T×28×4 quats; `geo_frame_cost = mean 2*arccos(|dot|)`; DTW; /max(Ta,Tb)) | quaternion (not SO(3)-matrix) geodesic; DTW-aligned | ✅ live-verified |
| DTW = **Euclidean** local cost, variable-length | `MS/pipelines/infer_emb_vec.py: calculate_real_variable_length_dtw` → `dtw_ndim.distance` (Euclidean) on flattened feats | DTW local cost = Euclidean | ✅ live-verified |
| Baselines UNPADDED (variable length) | `MS/baselines_nested.py:73` calls loader with `balance_class_frame_counts=False` → real lengths (walk73/pt53/pk71 median, matches §3) | unpadded claim TRUE (default True pads to 137!) | ✅ live-verified |
| Baseline features = 28-joint **quaternions** (112 ch) vs encoder 6-D (§5.2 note) | empirical load: shape (T,112)=28×4 | representation split | ✅ live-verified |
| Baseline numbers DTW 0.490/0.273/0.303, geo 0.371/0.234/0.264 | `MS/baselines_nested.py` → `MS/baselines_nested_results.json` {rows,aggregate,bars} | draft = JSON exactly, n=25 | ✅ live-verified |

### 0.3 Perceptual fine-tune (rank) + plateau

| Paper claim (§6.3/6.4) | Code path | Artifact | Status |
|---|---|---|---|
| Anchored ranking loss, margin = \|p_k−p_j\| d_perc gap, dists min-max normed (§6.4) | `TR/../perc_finetune.py: _triplet_rank` | — | 🔷 verified earlier this session (not re-opened 07-02) |
| Single-SELECT / fixed-200-epoch (no early-stop) (§6.4) | `perc_finetune.py --no-early-stop`; reports final epoch | `TR/cvn_results/perc_cut1_rank_noES.json` AND job 929964 `rankdiag_headline.json` | ✅ (config verified) |
| Plateau: val-Spearman plateaus ~ep100–120, train loss ↓ (§6.4) | `perc_finetune.py` PERC_DIAG per-epoch log | job 929964 log (5000 lines, 25 folds) — see [[project_rank_plateau_justification]] | ✅ live-verified |
| Headline rank **0.699/0.463/0.605 ±SE (.022/.035/.031)** | `perc_finetune.py` cut1 base rank single-SELECT | **CANONICAL = job 929964 `rankdiag_headline.json`** (chosen because it ALSO produced the §6.4 plateau trajectory → one run for both number + plateau). Draft updated 0.456→0.463 (3 places). perc_cut1_rank_noES (.456) = superseded earlier run. | ✅ RESOLVED 2026-07-02 |

### 0.4 STILL TO AUDIT (live-verify before submission)

| Claim | Suspected code path | Status |
|---|---|---|
| §4.1.1 MPJPE 0.825 / 0.046 (recon fidelity) | `TR/recon_fidelity.py` (`ev.std()`/`em.std()` = **SD across held-out clips**, n=all non-mirror states/drives) | ✅ RESOLVED 2026-07-02: (a) ± is SD not SE (confirmed np.std); (b) **the MPJPE numbers are NOT in the LaTeX submission draft** — §4.1.1 was condensed out. So NO SD/SE inconsistency in the submission. OPEN DECISION: §4.1 asserts "skips preserve high-frequency detail" QUALITATIVELY now (unsupported) — optionally restore a compact MPJPE sentence (re-run recon_fidelity.py for current values, report mean±SD explicitly labeled). Old 0.825/0.046 came from the markdown draft, NOT re-verified live. |
| §3 kinematic descriptors (active joints 24.6/11.6/20.0, ROM, temporal std) | **NO surviving computing script found** (psp_profiler = PSP not kinematics; neutral_clustering_motion = different purpose). Re-derived from `similarity_exemplars/*_local.pickle` (28 joints × 4 quat). | ⚠️ PARTIAL 2026-07-02: qualitative claim VERIFIED (pointing by far least active on every interpretation: ~13 vs 25/20 joints, ~half temporal std). Exact decimals NOT reproduced — best match (standardize=False, thr=0.01): active 25.1/13.1/20.0 (paper 24.6/11.6/20.0, pick exact), tstd 0.107/0.053/0.100 (paper .089/.048/.090), ROM ~2× off (aggregation differs). Original recipe unrecovered. RESOLVED 2026-07-02: §3 REWRITTEN to threshold-robust approximate statements (active-joint fraction ~0.9/~0.55/~0.7 walk/point/pick; pointing temporal variability ~half; median clip 73/53/71 EXACT) — removed the fragile exact decimals (24.6/11.6/20.0 etc., unrecoverable recipe). Qualitative claim holds across all recipes. Reproducible script shipped: `repro_bundle/05_code/kinematic_descriptors.py`. |
| **1,540 trials/action + median 11 raters (range 10–18)** (§2.4) | derived from `00_ground_truth/validAnswers*.csv` | ✅ VERIFIED 2026-07-02: walk/point/pick all 1,540 distinct trials, median 11 raters (walk 10–17, point 10–18, pick 10–15); unique participants 424/353/453. Range "10–18" from pointing. |
| Corpus 5,795 total / **LMA 433 · Bandai 3,077 · CMU 2,285** / 5,453 train (§2.3, FULL-corpus models: ladder) | `TR/datasets/{lma_perform_reorganized_cmu, bandai_organized, cmu_all_perform}`; feeder `MAMP*/feeder/feeder_lma.py:24-25` | ✅ VERIFIED 2026-07-02: .bvh counts = 433/3,077/2,285 EXACT; +342 heldout = 5,795; −342 = 5,453 train. |
| **CUT1 corpus** (TMR-comparison base, §2.3/§6.3): ~14,143 clips, "size-matched to HumanML3D budget" | `TR/datasets/cut1_flat` (config `lma_mamp_pose_cut1_pretrain.yaml`: dataset_dirs=[cut1_flat], norm_stats_cut1.npz) | ✅ VERIFIED 2026-07-02: cut1_flat = **14,143 .bvh = AMASS (ACCAD etc., cmu33-retargeted) + CMU**; 14,140 train (~3 nan-skipped); **LEAKAGE-CLEAN (0 LMA eval clips)**. Budget match: 14,143 ≈ HumanML3D ~14,616 motions ✓. **§2.3 REWRITTEN + VERIFIED 2026-07-02:** cut1_flat true sources (filenames) = KIT 4230 / BML 4862 / CMU 2285 / ACCAD 252 / MPI/DFaust/TCD/SFU/etc — **ALL AMASS constituents (cut1 = AMASS-only subset, NO HumanAct12).** HumanML3D composition VERIFIED from TMR `datasets/annotations/humanml3d/annotations.json`: 29,228 entries (mirrored; ~14.6k unique) = **AMASS-path 26,846 (~92%) + HumanAct12-path 2,382 (~8%)**. TMR DATASETS.md confirms "almost all ... included in AMASS" except HumanAct12 part. So: matched budget (~14.1k vs ~14.6k), LARGELY SHARED SOURCE (AMASS ~92%), cut1 lacks the ~8% HumanAct12. Both include CMU (via AMASS). Eval clips (LMA) in NEITHER. This is a STRONGER, now-precise fairness claim. |
| encode() for mld/vanilla families (exact pooling, n_windows) | `probes/encode_sweep.py`, `probes/encode_aetransformer.py` | 🔷 dispatch verified; internals not opened |
| MAMP encode pooling (mask_ratio=0, mean-pool 1020 patches) | `MAMP/encode_mamp_lma.py:62` mask_ratio=0.0, mean(dim=1) | ✅ verified earlier session |

---

## A. Results — Spearman correlations (the results spine)

| Claim / table row | Draft loc | Source artifact | Status |
|---|---|---|---|
| Vanilla recon 0.476/0.213/0.491 | §4.1, §4.2, §6.6 | `cvn_vrot1vel0.json` | ✅ audited exact 2026-07-01 |
| Vanilla recon+vel 0.472/0.205/0.399 | §4.2, §6.6 | `cvn_vrot1vel1.json` | ✅ |
| Vanilla vel-only 0.321/0.362/0.471 | §4.2, §6.6 | `cvn_vrot0vel1.json` | ✅ |
| MLD-AE recon 0.532/0.331/0.498 | §4.1, §4.2, §6.6 | `cvn_mldae.json` | ✅ |
| MLD-AE +vel 0.537/0.339/0.493 | §4.2, §6.6 | `cvn_mldaevel.json` | ✅ |
| MLD vel-only 0.451/0.278/0.254 | §4.2, §6.6 | `cvn_aevelonly.json` (family=mld) | ✅ correctly labeled |
| MLD-VAE 0.597/0.324/0.501 | §4.1, §6.6 | `cvn_mldvae.json` | ✅ |
| MAMP 0.351/0.299/0.490 | §4.4, §6.4, §6.6, §6.7 | `cvn_mamp.json` | ✅ |
| MAMP+pose 0.552/0.434/0.543 | headline, §6.1/6.4/6.6/6.7 | `cvn_mamppose.json` | ✅ |
| MAMP-uencfull 0.470/0.275/0.566 | §6.7 | `cvn_mampuencfull.json` | ✅ |
| cut1 MAMP+pose 0.462/0.302/0.481 | §6.3 (TMR) | `cvn_mamppose_cut1.json` | ✅ (professor anchor) |
| **TMR 0.452/0.425/0.408** | §6.3 | `tmr/encoded/` + `tmr/tmr_nested.py` (prints stdout) | ✅ regenerated LIVE 2026-07-01 |
| Rank fine-tune 0.699/0.456/0.605 | §6.3, §6.4 | `perc_cut1_rank_noES.json` (schema B/folds/test) | ✅ |
| Rank double-SELECT 0.630/0.404/0.560 (robustness sentence) | §6.4 | `perc_cut1_rank.json` | ✅ |
| DTW baseline 0.490/0.273/0.303 | §6.1, §6.6 | `baselines_nested_results.json` (verify) | 🔷 means match draft; SE recomputed, re-confirm fold source |
| Geodesic 0.371/0.234/0.264 | §6.1, §6.6 | `baselines_nested_results.json` | 🔷 same |
| Regression/alpha/fixed arms (memory only, NOT in paper) | — | `perc_cut1_{regression,alpha,fixed}.json` | ✅ (kept out of paper) |
| MAMP-uencfull+pose (2×2 cell) | §6.7 placeholder | job **929970** running | ⏳ PENDING |

All ± values are **SE = SD/√25**, recomputed from the JSONs (not ÷5 of rounded SD). CI-clears-bar
(mean−1.96·SE) re-verified: MAMP+pose all 3 ✓; MLD-AE/VAE + uencfull fail pointing ✓.

## B. Reconstruction-fidelity (MPJPE) — §4.1.1

| Claim | Draft loc | Source | Status |
|---|---|---|---|
| Plain-AE recon MPJPE 0.825 ± 0.039 | §4.1.1 (L378) | `recon_fidelity.py` / `probes/probe_ae_heldout_recon.py` exist | 🔷 TRACEABLE — re-run to confirm value + whether ± is SD or SE |
| MLD-AE recon MPJPE 0.046 ± 0.021 | §4.1.1 (L379) | same | 🔷 same — **flagged: SE-vs-SD of these two not yet confirmed** |

## C. Per-action kinematic structure — §3

| Claim | Draft loc | Source | Status |
|---|---|---|---|
| Active joints 24.6/11.6/20.0; frac 0.88/0.41/0.71; temporal-std 0.089/0.048/0.090; ROM 0.276/0.123/0.290 | §3 table | computation over motion data (script TBD — `psp_profiler.py`?) | ⚠️ INHERITED — locate the computing script; re-derive |
| Clip lengths: walk median 73, pick 71, point 53; max 137/100 | §2.2, §3 | motion data | ⚠️ INHERITED — traceable from clip files; not re-derived |

## D. Corpus / data facts — §2

| Claim | Draft loc | Source | Status |
|---|---|---|---|
| 5,795 pretraining BVH clips; ~5,453 training after holdout | §2.3 | feeder logs ("5453 training clips" seen in session smoke logs) | 🔷 5,453 seen in logs; 5,795 total needs the manifest |
| Source split: LMA 433 / Bandai-Namco 3,077 / CMU 2,285 | §2.3 table | pretraining manifest | ⚠️ INHERITED — confirm against manifest file |
| 342 held-out LMA clips (57×3×2) | §2.1, §2.3 | arithmetic 57×3×2=342 ✓; matches held-out construction | ✅ arithmetic; holdout logic in code |
| 1,540 triplet trials/action; median 11 raters (range 10–18) | §2.4 | human-rating dataset | ⚠️ INHERITED — confirm against ratings data files |
| 34 joints × 6-D; 120-frame window; 1020 patches (30×34) | §2.2, §5.1 | config/code (model_args) | 🔷 in configs; arithmetic 30×34=1020 ✓ |
| d_perc = 1 − c(0,2); canonical-57 = 1+24+32 | §2.4.2, §2.1 | `cv_preliminary.py` (get_inverse_direct_comparison_value, load_action_keys) | ✅ verified in code (memory reference_dperc_lma_effort_space) |

## E. Citations — verify against primary sources

| Citation | Draft loc | Claim made | Status |
|---|---|---|---|
| MLD [Chen et al., CVPR 2023] | §4.1 | SkipTransformer / Motion Latent Diffusion | 📚 verify venue + that MLD is the SkipTransformer source |
| MAMP [Mao et al., ICCV 2023] | §4.4 | masked motion prediction framework + hyperparams | 📚 verify venue + method attribution |
| TMR [Petrovich et al., ICCV 2023] | §6.3 | text↔motion retrieval, contrastive, HumanML3D-trained | 📚 verify venue + "trained on HumanML3D" |
| Bandai-Namco [Kobayashi et al., 2023] | §2.3 | 3,077 clips, styled everyday actions | 📚 verify dataset citation |
| CMU Graphics Lab MoCap DB | §2.3 | 2,285 clips | 📚 standard DB; confirm attribution |
| joints2smpl ~4cm MPJPE fit | §6.3, §8 | SMPLify fit fidelity | 🔷 memory `project_tmr_smpl_pipeline` (probe result); re-confirm if challenged |

---

## Priority to close before submission
1. **§4.1.1 MPJPE (B)** — re-run `recon_fidelity.py`; confirm values + SD-vs-SE. *(Flagged inconsistency risk.)*
2. **§3 kinematic table (C)** — locate/re-run the computing script; these are inherited.
3. **Corpus counts + rater numbers (D)** — confirm 5,795 / source split / 1,540 / 11-raters against manifest + ratings files.
4. **Baseline SE fold source (A)** — re-confirm DTW/geodesic from `baselines_nested_results.json`.
5. **Citations (E)** — verify all five venues + attributions against primary sources.
6. **929970** fills the one PENDING cell.
