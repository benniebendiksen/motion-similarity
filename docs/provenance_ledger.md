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
| **CUT1 corpus** (TMR-comparison base, §2.3/§6.3): ~14,143 clips, "size-matched to HumanML3D budget" | `TR/datasets/cut1_flat` (config `lma_mamp_pose_cut1_pretrain.yaml`: dataset_dirs=[cut1_flat], norm_stats_cut1.npz) | ✅ VERIFIED 2026-07-02: cut1_flat = **14,143 .bvh = AMASS (ACCAD etc., cmu33-retargeted) + CMU**; 14,140 train (~3 nan-skipped); **LEAKAGE-CLEAN (0 LMA eval clips)**. Budget match: 14,143 ≈ HumanML3D ~14,616 motions ✓. **§2.3 REWRITTEN + VERIFIED 2026-07-02:** cut1_flat true sources (filenames) = KIT 4230 / BML 4862 / CMU 2285 / ACCAD 252 / MPI/DFaust/TCD/SFU/etc — **ALL AMASS constituents (cut1 = AMASS-only subset, NO HumanAct12).** HumanML3D composition VERIFIED from TMR `datasets/annotations/humanml3d/annotations.json`: 29,228 entries (mirrored; ~14.6k unique) = **AMASS-path 26,846 (~92%) + HumanAct12-path 2,382 (~8%)**. TMR DATASETS.md confirms "almost all ... included in AMASS" except HumanAct12 part. So: matched budget (~14.1k vs ~14.6k), LARGELY SHARED SOURCE (AMASS ~92%), cut1 lacks the ~8% HumanAct12. Both include CMU (via AMASS). Eval clips (LMA) in NEITHER. This is a STRONGER, now-precise fairness claim. **CONSTRUCTION PROVENANCE VERIFIED 2026-08-27 (reviewer-defense):** cut1 was NOT derived from HumanML3D/TMR files — it is an independent AMASS→CMU-33 build. Pipeline = `amass/run_amass_all.sh` (2-stage: Stage A `amass_smplh_to_bvh_batch.py` SMPL-H .npz→SMPL-24 BVH via smpl2bvh, 30fps + Z-up→Y-up + strip hands; Stage B `retarget_amass_to_cmu_batch.py` SMPL-24→CMU-33 via bvhReader/retarget.py) over 22 AMASS subsets → `datasets/amass_cmu_flat`. **`cut1_flat` = a SYMLINK dir (created 2026-06-24 via `ln -s`, no committed build script) into `amass_cmu_flat` MINUS {CNRS, DanceDB, GRAB, HUMAN4D, SOMA} PLUS the separate `cmu` corpus (2285).** The 5 excluded subsets are exactly the AMASS subsets HumanML3D does NOT draw from ⇒ selection was **HumanML3D-aware at the SUBSET level** (drop subsets HML3D never uses) but **NOT at the CLIP level** (kept EVERY clip in retained subsets; HML3D captioned only some clips per subset). This precisely explains the ~92%/~8% split. **EXACT composition (16 prefixes, `ls cut1_flat`):** KIT 4230 · BioMotionLab 3061 · cmu 2285 · BMLmovi 1801 · Eyes 750 · BMLhandball 649 · EKUT 348 · MPI 327 · ACCAD 252 · DFaust 129 · Transitions 110 · TCD 62 · SFU 44 · TotalCapture 37 · SSM 30 · HumanEva 28 = 14,143. **Caption-coverage classification 2026-08-27 (`scratchpad/mechanism_jobs/cut1_final_classify.py` exhaustive clip-id match vs `tmr/…/humanml3d/annotations.json`): 12,972 (91.7%) have a HumanML3D caption; 1,171 (8.3%) GENUINE ABSENCES** = within-HML3D-subset clips HML3D didn't individually annotate (cmu 498 · BMLhandball 361 · BioMotionLab 239) + TCD_handMocap 62 (+ MPI 11, Eyes 1). NOTE the fuzzy trailing-token matcher (91.7%) may over-count; an EXACT AMASS-source-path match is pending to lock the airtight `cut1∩HML3D` subset for the hybrid corpus. NAMING: "corpus-matched" is a budget+shared-source(AMASS) fairness claim, NOT a subset claim (cut1 ⊄ HML3D and HML3D ⊄ cut1). |
| encode() for mld/vanilla families (exact pooling, n_windows) | `probes/encode_sweep.py`, `probes/encode_aetransformer.py` | 🔷 dispatch verified; internals not opened. **2026-07-05: both patched to infer `use_skips` from state_dict keys** (`encoder.linear_blocks.*` / `transformer_encoder.linear_blocks.*`) since use_skips is not serialized in the checkpoint config — required for the 4-corner skip ablation to load no-skip/skip variants. Backward-compat verified (canonical skip+no-skip ckpts encode unchanged). `.bak` saved. |
| MAMP encode pooling (mask_ratio=0, mean-pool 1020 patches) | `MAMP/encode_mamp_lma.py:62` mask_ratio=0.0, mean(dim=1) | ✅ verified earlier session |

---

### 0.5 MODEL → TRAINING-JOB → CONFIG PROVENANCE (traced 2026-07-05)

⚠️ **SCOPE: this section is TRAINING provenance only** — model → training-job → training-config →
checkpoint. A paper *number* = training ∘ **evaluation**, and the EVAL half (which `cv_nested.py`
invocation, with which `--epoch-grid`/`--loss-plateau`/`--ckpt-epochs` SELECTION protocol, produced
each `cvn_*.json`) is **NOT fully recorded** — see §0.6. Do not read "every model mapped" as "every
number reproducible end-to-end"; the training chain is solid, the eval-command chain has gaps.

Every paper encoder mapped to (a) the launching sbatch/command, (b) the full training
parameterization, (c) the checkpoint dir. Paths relative to
`TR = triplets/` on chimera. **Per-fold checkpoint SELECTION is validation-driven at
eval time (§0.1)** — but the candidate MENU (grid/plateau) is an eval-command choice, NOT captured
here; the "final epoch reached" below is the trajectory extent only. All account/partition recorded
because the impact billing cap sometimes forces pomplun (see [[reference_chimera_encode_recipe]]).

**MAMP family** (`TR/MAMP/`, entrypoint `main_pretrain.py --config <yaml> --output_dir <dir>`;
architecture+training FULLY captured by the yaml — verified: dim_feat 256, depth 8 / dec 5,
heads 8, patch t=4, mask_ratio 0.80, blr 1e-3, min_lr 5e-4, wd 0.05, warmup 20, bs 64,
window 120, 34 joints, 6D, holdout_lma True):

| Model (ENCODERS key) | sbatch | config yaml | corpus | epochs | ckpt dir | acct/part |
|---|---|---|---|---|---|---|
| MAMP (motion-only) | `training/mamp_mamp_1200_pomplun.sbatch` | `lma_mamp_pretrain.yaml` (pose_weight 0) | 3ds | 1200 | `MAMP/output_dir/lma_mamp_holdout_1200` | funda/pomplun |
| MAMP-skip | `training/mamp_skip_1200_pomplun.sbatch` | `lma_mamp_skip_pretrain.yaml` (skip 1, pose 0) | 3ds | 1200 | `…/lma_mamp_skip_holdout_1200` | funda/pomplun |
| MAMP-skip+pose | `training/mamp_skippose_1200_pomplun.sbatch` | `lma_mamp_skippose_pretrain.yaml` (skip 1, pose 1) | 3ds | 1200 | `…/lma_mamp_skippose_holdout_1200` | funda/pomplun |
| **MAMP+pose (HEADLINE)** | **2-phase**: `mamp_pose_pretrain_aicore.sbatch` → `mamp_pose_continue_aicore.sbatch` | `lma_mamp_pose_pretrain.yaml` (600ep) **then** `lma_mamp_pose_resume.yaml` (`--resume checkpoint-599.pth`, delta = epochs 600→1200 ONLY) | 3ds | 1200 (ckpt-1199) | `…/lma_mamp_pose_holdout` | impact/H200 |
| MAMP+pose cut1 (TMR comparator) | inline (jobs 915946 orig, **927917 canonical retrain**) | `lma_mamp_pose_cut1_pretrain.yaml` (dataset_dirs=[cut1_flat], norm_stats_cut1.npz) | cut1 (~14.1k) | 1200 | `…/lma_mamp_pose_holdout_cut1` | impact/H200 |
| MAMP+pose 6ds (side-expt) | `training/mamp_pose_6ds_aicore.sbatch` | `lma_mamp_pose_6ds_pretrain.yaml` | 6ds | — | `…/lma_mamp_pose_holdout_6ds` | impact/H200 |
| MAMP-uenc (2×2) | inline (no committed sbatch) | `lma_mamp_uenc_pretrain.yaml` | 3ds | 1200 (ckpt-1199) | `MAMP_uencoder/output_dir/lma_mamp_uenc_holdout_1200` | — |
| MAMP-uencfull (2×2) | inline (no committed sbatch) | `lma_mamp_uencfull_pretrain.yaml` | 3ds | 1200 (ckpt-1199) | `MAMP_uencfull/output_dir/lma_mamp_uencfull_holdout_1200` | — |
| MAMP-uencfull+pose (2×2) | inline (job **929970**) | `lma_mamp_uencfullpose_pretrain.yaml` | 3ds | 1200 (ckpt-1199) | `MAMP_uencfull/output_dir/lma_mamp_uencfullpose_holdout_1200` | — |

**MLD-AE family** (`TR/training/train_ae_mld.py`; **each run self-documents via `<dir>/config.json`**
— verified present. Common: hidden 384, latent 256, K=1, enc/dec depth 7, heads 6, dropout 0.1,
lr 1e-4, wd 0.01, bs 64, epochs 10000, lr_decay 5000, val 5%, mask_ratio 0.0, window 120, seed 0.
Name = `v0_ae[_variant]_3ds_seed0[_suffix]`):

| Model | sbatch | command flags | ckpt dir | acct/part |
|---|---|---|---|---|
| MLD-AE-holdout (recon rung) | `training/aeh_traj.sbatch` | `--run-name v0_ae --seed 0 --datasets-3ds --keep-all-checkpoints --checkpoint-every 100 --run-suffix _traj` | `checkpoints_ae_mld/v0_ae_3ds_seed0_traj` | impact/H200 |
| MLD-AE-vel | `training/aevel_traj.sbatch` | `--run-name v0_ae_vel …_traj` | `…/v0_ae_vel_3ds_seed0_traj` | impact/H200 |
| MLD-AE-prov | `training/prov_traj.sbatch` | `--run-name v0_ae_provloss …_traj` | `…/v0_ae_provloss_3ds_seed0_traj` | impact/H200 |
| **MLD-AE-no-skip (4-corner)** | `training/ae_mld_noskip.sbatch` | `--run-name v0_ae --datasets-3ds --seed 0 --no-skips --run-suffix _noskip_traj --keep-all-checkpoints --checkpoint-every 100 --epochs 10000` | `…/v0_ae_3ds_seed0_noskip_traj` | funda/pomplun (job **931697 COMPLETED** 10000ep, best val .01414) |
| **MLD-AE skipON EXTENSION 6700→10000** | `training/aeh_traj_extend.sbatch` | identical canonical cmd → AUTO-RESUME from epoch_06700.pt to 10000 (canonical run time-limited at 6700; extended for equal-extent skip pair) | `…/v0_ae_3ds_seed0_traj` (now to ep10000) | funda/pomplun (job **932386 COMPLETED** 2026-07-06, best val .01440) |
| **MLD-AE-attnpool (swap-pooling)** | `training/ae_mld_attnpool.sbatch` | `--run-name v0_ae --datasets-3ds --seed 0 --pool attn --run-suffix _attnpool_traj --keep-all-checkpoints --checkpoint-every 100 --epochs 10000` (SKIPS ON; only pooling swapped: latent-query→attention-pool) | `…/v0_ae_3ds_seed0_attnpool_traj` | funda/pomplun (job **932654 COMPLETED** 10000ep, best val .01202) |

**Vanilla-transformer family** (`TR/train_AE_ablation_with_velocity_curriculum.py --preset <p>`;
arch hard-coded in call: hidden/latent from args, heads 4, enc/dec depth 3. Loss preset = the
full parameterization, defined in the script's PRESETS dict. epochs 10000 nominal, but the
canonical runs hit the 3-day TIME LIMIT ~ep8400):

| Model (ENCODERS key) | sbatch | preset (loss) | ckpt dir | acct/part |
|---|---|---|---|---|
| **vanilla-rot1vel0 (recon, §4.1/§6.6)** | `training/van_rot1_vel0.sbatch` | `rot1_vel0` (rotMSE 1, vel 0, rest 0) | `checkpoints/v3_ablation_rot1_vel0` | impact/H200 (job 909334, TIME-LIMIT ep8458) |
| vanilla-rot1vel1 (recon+vel) | `training/van_rot1_vel1.sbatch` | `rot1_vel1` | `…/v3_ablation_rot1_vel1` | impact/H200 |
| vanilla-rot0vel1 (vel-only) | `training/van_rot0_vel1.sbatch` | `rot0_vel1` | `…/v3_ablation_rot0_vel1` | impact/H200 |
| AE-vel (legacy PROV, back-ref only) | `training/ae_velPROV_aicore.sbatch` | `PROV` (geo+vel curriculum) | `…/v3_ablation_PROV` | impact/H200 |
| AE-recon (legacy PROV_noVel, back-ref) | `training/ae_provNoVel_pomplun.sbatch` | `PROV_noVel` | `…/v3_ablation_PROV_noVel` | funda/pomplun |
| **vanilla+skips (4-corner)** | `training/van_rot1_vel0_skips.sbatch` | `rot1_vel0 --use-skips` (encoder-internal U-Net; +0.13M fusion linears = 9.05M) | `checkpoints/v3_ablation_rot1_vel0_skips` | funda/pomplun (job **931696 COMPLETED** 2026-07-06, 10000ep) |
| **vanilla+querypool (swap-pooling)** | `training/van_querypool.sbatch` | `rot1_vel0 --pool query` (attention-pool→MLD-style latent-query token; std decoder retained) | `checkpoints/v3_ablation_rot1_vel0_querypool` | funda/pomplun (job **932655 COMPLETED** 2026-07-09, 10000ep) |
| **vanilla-bigcap (capacity control)** | `training/van_bigcap.sbatch` | `rot1_vel0 --hidden-dim 384 --latent-dim 256 --num-heads 6 --enc-layers 7 --dec-layers 7 --capacity-tag _bigcap` (35.0M params ≈ MLD's 31.16M budget; tests if MLD edge = architecture or SIZE) | `checkpoints/v3_ablation_rot1_vel0_bigcap` | funda/pomplun (job **932900 RUNNING**, ep~2427) |

**Perceptual fine-tune** (already in §0.3): job **929964** `rankdiag_headline.json`, `perc_finetune.py`
cut1 base rank single-SELECT, fixed 200ep — see [[project_rank_plateau_justification]].

Only remaining `—`: uenc/uencfull acct/partition (launched inline, no committed sbatch;
epochs+dirs now pinned = 1200/ckpt-1199, matching the pose budget). cut1 also inline but
job IDs recovered (915946/927917). All headline+rung models have a committed sbatch.

---

### 0.6 CHECKPOINT → EVAL-INVOCATION → cvn_*.json PROVENANCE (opened 2026-07-05)

The missing half of §0.5. A paper Spearman number = a `cv_nested.py` run over a trained checkpoint
trajectory, and the **SELECTION protocol** (candidate epoch MENU: `--epoch-grid`/`--loss-plateau`
or explicit `--ckpt-epochs`) materially affects the number — a coarser/uncapped menu lets SELECT
reach later, possibly over-trained checkpoints. This menu is an **eval-command choice not captured
in §0.5**.

**Discovered gap (the trigger):** the paper's MLD-AE row (`cvn_mldae.json`, .532/.331/.498) has
selected epochs at 300/900/1500/1700/2600/3200 — **none on the coarse `100,1200,…,6700` grid**, max
3200. So it came from a **FINE-grid run whose exact command is NOT committed anywhere** (no sbatch/sh
outputs `cvn_mldae.json`; no committed cv_nested call uses `--epoch-grid`/`--loss-plateau`). The
committed `cvn_mldholdout.sbatch` uses the COARSE grid and its result JSON doesn't exist on disk →
superseded/never-completed. So `cvn_mldae.json` is effectively an **ad-hoc/interactive eval run**.

**Consequence for the skip ablation:** my skips-OFF run (`cvn_mldnoskip_results.json`, job 931716)
used the coarse `--ckpt-epochs 100…6700` menu and leaned to **6700 (10/25 folds)**. It is therefore
NOT a matched-protocol pair with the fine-grid paper skips-ON number. Pointing is robust anyway
(0.331 fine vs 0.334 coarse — pointing barely varies across the trajectory), but walking/picking
deltas are protocol-contaminated.

**RESOLUTION (in progress):** re-run BOTH skips-ON and skips-OFF **eval only** (checkpoints already
trained, untouched) under ONE recorded plateau-capped protocol I define, so the ablation pair is
provably matched AND fully provenanced. Bonus: if re-run skips-ON reproduces ≈.532/.331/.498, it
retroactively validates the paper MLD-AE number with a committed command.

| Paper row | cvn file | eval command recorded? | status |
|---|---|---|---|
| MLD-AE (skips-ON, .532/.331/.498) | `cvn_mldae.json` | ❌ orig ad-hoc → ✅ **REPRODUCED** by `cvn_mldae_capped.sbatch` (job 931723, `--epoch-grid 100 --loss-plateau 3200`): .5318/.3314/.4982 EXACT + matching epoch-histogram | ✅ RESOLVED 2026-07-05 (paper row now committed-reproducible) |
| MLD-AE-noskip (skips-OFF) | matched: `cvn_mldnoskip_capped_results.json` (job 931724, same protocol): **.5677/.3334/.5112** | ✅ `cvn_mldnoskip_capped.sbatch` | ✅ the canonical skip-ablation comparator (supersedes coarse 931716) |
| others (vanilla/MAMP/etc.) | `cvn_*.json` | 🔷 committed sbatch exist for most (cvn_aerecon/aevel/mldprov/mldvel…); grids vary | audit before camera-ready |

**⚠️ AUDIT 2026-07-09 — THE GAP IS BROADER THAN ONE ROW.** Checked selected-epoch histograms of all
core paper cvn files vs the committed sbatch commands:
- Committed sbatch use COARSE `--ckpt-epochs` lists: AE-recon `[100,1400,2700,4100,5400,6700,8000]`,
  AE-vel `[100,1700,3400,5100,6700,8300,10000]`, MAMP `[200..800]`, MLD-AE/prov/vel `[100,1200,2300,3400,4500,5600,6700]`.
- But the PAPER cvn_*.json selected epochs land on the **grid-100 fine menu** (e.g. cvn_vrot1vel0 picks
  500/700/900/1100/1800/2100/3100…; cvn_mamppose 300/400/…/1100; cvn_mldae 300/900/1500/1700/2600/3200).
  These are NOT on the committed coarse lists → **the paper numbers were generated by uncommitted
  FINE-grid (`--epoch-grid 100`) runs**, same ad-hoc pattern as cvn_mldae, across ESSENTIALLY ALL CORE MODELS
  (vrot1vel0/vrot1vel1/vrot0vel1/mamp/mamppose/mldae/mldaevel/mldvae).
- So: training provenance (§0.5) solid; EVAL provenance (§0.6) has a SYSTEMIC gap — committed sbatch ≠
  what produced the JSONs. Numbers themselves audited-exact (§A), but not command-reproducible.

**RESOLUTION OPTIONS (eval-only, NO retrain; ~2h/model on pomplun):**
(a) Re-run the WHOLE §A spine under ONE committed protocol `--epoch-grid 100 --loss-plateau <cap>`,
    commit every sbatch → entire results table becomes command-reproducible. Rigorous, ~8 core evals.
    Precondition already met: SELECT-Spearman plateaus (D2) + ceiling-invariance demonstrated (skip pair)
    so re-run numbers ≈ current (pointing especially).
(b) Minimal: re-run only the ablation-comparison rows that appear side-by-side in one table (e.g. vanilla
    row-1 baseline for the skip/pooling table) so within-table protocol is uniform; document the rest as
    "audited-exact, fine-grid, single ad-hoc run" with a repro sbatch shipped.
DECISION PENDING (user). Row-1 vanilla (cvn_vrot1vel0) is the first instance — it anchors the
skip/pooling table's "0.22" baseline, so at minimum re-run it under the matched protocol.

#### 0.6.1 MECHANISM-ABLATION EVALS (skip / pooling / capacity) — all committed, matched protocol

All under `cv_nested.py --k 5 --repeats 5 --n-triplet-seeds 5 --triplet-epochs 200 --epoch-grid 100
--loss-plateau 10000`. Raw Spearman n=25 (W/P/K). Committed sbatch each.

⚠️ **PROTOCOL CORRECTION (2026-07-09):** `--loss-plateau 10000` = FULL-TRAJECTORY, which is the
**REJECTED** policy #2 (trajectory length is an arbitrary training-budget artifact — see D2-OFFICIAL
in [[project_methodology_decisions]]). **Official policy = grid-100 up to each model's own
RECONSTRUCTION-loss plateau** (model-intrinsic, recon-side, not perceptual, not the arbitrary
trajectory end). These runs are ceiling-invariant so numbers ≈ correct, BUT must be RE-RUN /
re-verified at per-model recon-plateau caps before they are manifest-official. They remain the
DEFINITIVE matched pairs on the CURRENT (to-be-corrected) protocol, superseding the intermediate
capped-3200 runs 931723/931724.

| Model | eval sbatch | job | cvn file | W / **P** / K |
|---|---|---|---|---|
| vanilla (attn, no skips) | *(§A cvn_vrot1vel0; OLD protocol — RE-RUN pending, see 0.6.2)* | — | `cvn_vrot1vel0.json` | .476/**.213**/.491 |
| vanilla + skips | `cvn_vanskips_full.sbatch` | 932504 | `cvn_vanskips_full_results.json` | .471/**.221**/.456 |
| vanilla + querypool | `cvn_vanquerypool.sbatch` | **932934 RUNNING** | `cvn_vanquerypool_results.json` | pending |
| vanilla bigcap (capacity) | *(eval pending training 932900)* | — | — | pending |
| MLD native (query+skips) | `cvn_mldae_full.sbatch` | 932421 | `cvn_mldae_full_results.json` | .532/**.331**/.498 |
| MLD no-skips | `cvn_mldnoskip_full.sbatch` | 932422 | `cvn_mldnoskip_full_results.json` | .570/**.330**/.513 |
| MLD → attn-pool | `cvn_mldattnpool.sbatch` | 932759 | `cvn_mldattnpool_results.json` | .501/**.424**/.497 |

⚠️ **PAPER-ALIGNMENT ACTION (flagged 2026-07-09):** current main.tex §6.8 (lines ~693–758) was
written PRE-ablation and frames "MLD-style U-Net skips" as a meaningful lever (line ~695: pointing
lifts "because skip connections—the mechanism that lifts…"). Our 4-corner ablation now shows skips
PERCEPTUALLY INERT (not contradictory — paper's narrow claim is skips lift COARSE actions, which our
MLD data corroborates: skips-off HELPED walk/pick — but the FRAMING of skips as the operative
mechanism is undercut). Plus the MLD→attn-pool result (pooling ≠ MLD's advantage) and capacity
control are entirely absent from the paper. **RECONCILE before submission:** rewrite §6.8 around the
verified story (skips inert; pooling is selection-vs-integration; capacity TBD) — do NOT ship the
pre-ablation framing alongside the ablation table. NO unsupported claim is currently IN the paper
(findings just aren't there yet), so this is a pending-write, not a live error. See [[project_skip_ablation_4corner]] [[project_swap_pooling]].

**Findings locked (mechanism):** (1) SKIPS inert both archs (vanilla P +.008, MLD P −.001).
(2) POOLING: MLD→attn-pool RAISES pointing .331→.424 (contradicts "richer un-averaged z"; attention-pool
= per-frame SELECTION beats latent-query INTEGRATION on the localized action). (3) CAPACITY control
(bigcap 35M vs MLD 31M) in progress — tests if the residual MLD>vanilla gap is architecture or size.

**Code provenance for these (all `.bak*` saved on chimera):** `use_skips` flag (VAEMotionMLD/AEMotionMLD
SkipTransformer, AETransformer UNetEncoderStack); `use_pool` flag (AEMotionMLD attn_pool head +
AETransformer latent_query head); capacity args (`--hidden-dim/--num-heads/--enc-layers/--dec-layers/
--capacity-tag` in vanilla trainer). Encode paths INFER all three from weights (encode_sweep.py:
`attn_pool.*`→attn / `encoder.linear_blocks.*`→skips; encode_aetransformer.py: `latent_query`→query,
`transformer_encoder.linear_blocks.*`→skips, hidden/latent/depth/heads read from weight shapes +
heads=hidden//64). Backward-compat verified on canonical ckpts.

#### 0.6.2 §A SPINE RE-RUN (committed protocol) — DECIDED, NOT YET DONE

AUDIT (0.6 above) found MOST core paper cvn_*.json came from uncommitted fine-grid runs. DECISION
(user 2026-07-09): re-run whole §A spine eval-only under committed `--epoch-grid 100 --loss-plateau
<per-model recon-plateau>`; cap = each model's own **reconstruction-loss plateau** epoch (D2-OFFICIAL:
where val_recon/val_mpjpe stops improving — model-intrinsic, recon-side, NOT perceptual/PSP, NOT the
arbitrary full-trajectory end), per-model NOT global (MAMP ~pre-divergence, MLD-AE/vanilla ~their own
recon-flat epoch). NEED to derive each cap from its val curve (fix operational def once, apply
uniformly). Applies to the mechanism evals (0.6.1) too — they used full-trajectory and must be
re-verified at recon-plateau. Precondition met (ceiling-invariance) so numbers ≈ current. Batch
after mechanism experiments + capacity result.

**TODO before submission:** confirm each paper `cvn_*.json` traces to a committed cv_nested command
with a known selection menu; note any that are ad-hoc. See [[project_skip_ablation_4corner]],
[[project_swap_pooling]].

---

## A. Results — Spearman correlations (the results spine)

| Claim / table row | Draft loc | Source artifact | Status |
|---|---|---|---|
| Vanilla recon 0.476/0.213/0.491 | §4.1, §4.2, §6.6 | `cvn_vrot1vel0.json` | ✅ audited exact 2026-07-01 |
| Vanilla recon+vel 0.472/0.205/0.399 | §4.2, §6.6 | `cvn_vrot1vel1.json` | ✅ |
| Vanilla vel-only 0.321/0.362/0.471 | §4.2, §6.6 | `cvn_vrot0vel1.json` | ✅ |
| MLD-AE recon 0.532/0.331/0.498 | §4.1, §4.2, §6.6 | `cvn_mldae.json` | ⚠️ number audited-exact, but EVAL COMMAND ad-hoc/uncommitted (fine grid ≤3200) — see §0.6; re-run under recorded protocol pending |
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
