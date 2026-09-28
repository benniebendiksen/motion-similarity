# Unified Corpus Build — CMU-adapted HumanML3D (reproducibility record)

**Goal.** Retarget **every HumanML3D clip** to our **34-joint, 6-D-rotation, CMU-skeleton, 30 fps** format,
caption-aligned, banks + norm-stats, held-out-clean — one unified corpus for all Tier-1 models that **matches
published TMR's training set by construction** (so a Tier-1 TMR-objective vs published-TMR comparison isolates
representation/conversion, not corpus). Decided 2026-09-26: do it right, gate every route.

**Scope.** HumanML3D = **14,614 clips** (×2 with left/right mirrors = 29,228 annotation ids). Sources:
**13,423 AMASS** (KIT 4647, CMU 2913, BMLmovi 1839, Eyes_Japan 1465, MPI_HDM05 771, BioMotionLab 373, EKUT 350,
ACCAD 277, + smaller subsets) + **1,191 HumanAct12**. The keystone that defines the exact clip set + crops +
captions is TMR's `learned_baselines/tmr/repo/datasets/annotations/humanml3d/annotations.json` (`id → path,
duration, start/end, captions`) — *what published TMR trained on*.

**Status (2026-09-28).**

| Source | Clips | State |
|---|---|---|
| AMASS (non-CMU) | 10,510 src | ✅ aligned to captions (99.99%) |
| CMU | all HumanML3D CMU | ✅ complete — round-trip **byte-exact** |
| HumanAct12 | 1,191 | ✅ **source validated** (gate passed); positions→CMU rotations (joints2smpl) **pending** |
| shared finish | — | crop-per-annotation, caption-align, banks, norm-stats, held-out — **pending** |

---

## 1. Provenance ledger (per source — the required artifact)

| Source | Exact release / file | Retargeting route → 34-joint 6D CMU @30fps | Fidelity gate | Result |
|---|---|---|---|---|
| **AMASS non-CMU** (KIT, BMLmovi, Eyes_Japan, MPI_HDM05, BioMotionLab, EKUT, ACCAD, …) | AMASS SMPL-H (already retargeted → `datasets/amass_cmu_flat`, 13,647) | `SMPL-H → SMPL-24 BVH (smpl2bvh) → CMU-34, lerp/slerp resample →30fps` (`amass/*`) | `annotations.json` path → retargeted filename, exact per-source (`align_corpus_v2.py`) | **10,509/10,510 (99.99%)**, 1 edge miss |
| **CMU** | cgspeed **Motionbuilder-friendly** BVH (`cmuconvert-mb2-*`, standard CMU skeleton) + existing `cmu_all_perform` | `bvhConverterToPerform.prepare_files`: drop `Neck/RThumb/LThumb` → `adjustFrameRate(30)` (120→30, 4:1) → `fix_end_sites` | round-trip: convert mb `80_63`, compare to `cmu_all_perform/80_63.bvh` (`cmu_roundtrip.py`) | **MAX abs diff = 0** (28 joints, 568 frames) |
| **HumanAct12** | **HumanML3D `pose_data/humanact12.zip`** (24-joint SMPL positions, HumanML3D frame) — *not* raw action-to-motion | *(pending)* `positions → joints2smpl (fit_seq.py) → SMPL params → SMPL→BVH→CMU-34` | source gate: `joints_to_guofeats(clip)` vs `new_joint_vecs/000001` (`ha12_gate.py`) | **non-feet max 0.037 (float match)**; feet 2/35 frames (binarization) |

Output convention shared by every route: **ROOT Hips + 28 articulated joints** (round-trip-verified identical
hierarchy between `amass_cmu_flat` and `cmu_all_perform`), **Frame Time 0.033333 (30 fps)**.

---

## 2. Conversion recipe — every source, reproducibly

### 2.1 AMASS (non-CMU), 13,423 clips
Already retargeted in `datasets/amass_cmu_flat` (13,647 clips ⊇ HumanML3D's AMASS; the extra are non-HumanML3D
AMASS subsets like GRAB/DanceDB, harmless — we intersect with `annotations.json`). Pipeline: `amass/
amass_smplh_to_bvh_batch.py` (SMPL-H `.npz` → SMPL-24 BVH via smpl2bvh, R_x(−90°) root fix, per-clip gender/fps)
→ `retarget_amass_to_cmu_batch.py` → resample to 30 fps. **Caption alignment:** `align_corpus_v2.py` normalizes
`annotations.json` paths (`EKUT/265/SLP102_poses` → `EKUT_265_SLP102`) to our filenames
(`EKUT_265_SLP102_cmu33_30fps`); **10,509/10,510 resolve**. Note: 2,341 source clips map to multiple HumanML3D
ids (distinct start/end + captions) → the shared **crop stage (§3)** slices per `annotations.json`.

### 2.2 CMU, all HumanML3D CMU clips
2,500 already in `cmu_all_perform`; the 248 source ids it lacked (`datasets/_cmu_missing_ids.txt`, 16 subjects:
30,31,36,54,55,62,63,64,76,78,80,90,91,118,139,144) were reconstructed:
1. **Right release matters.** cgspeed ships a **Daz-friendly** (Poser skeleton: `hip/abdomen/chest/rShldr/rThumb1`)
   and a **Motionbuilder-friendly** (standard CMU: `Hips/LHipJoint/LowerBack/Neck/Neck1/LThumb/RThumb/
   LeftHandIndex1`) release. `bvhConverterToPerform` drops `Neck/RThumb/LThumb`, which exist **only** in the
   standard skeleton → the **Motionbuilder** release is required (the Daz one would raise "joint not found").
2. **Fetch:** `mediafire_dl_mb.py` (9 `cmuconvert-mb2-*.zip` → `datasets/_cmu_mb_raw/`, extract → 1,540 BVH; all
   248 ids present). *MediaFire note:* the browser path is blocked by ads/account walls; the script decodes the
   page's base64 `data-scrambled-url` → direct link → curl. **Reusable for any MediaFire fetch.**
3. **Convert:** `prepare_files` = load → `convert_to_target_format` (drop `Neck/RThumb/LThumb` + end-sites) →
   `adjustFrameRate(30)` → save; then `fix_end_sites`. Imported via **`bvhconv_lib.py`** = `head -377
   bvhConverterToPerform.py` (the original file has a broken executable tail on import).
4. **Gate (round-trip):** converting mb `80_63` reproduces `cmu_all_perform/80_63.bvh` with **MAX abs diff = 0** →
   pipeline proven the *same* one that built cmu_all_perform. `cmu_convert_248.py` → 248 outputs in
   `datasets/_cmu_248_out/`, all 28-joint/30 fps, 0 bad. Full CMU = 2,500 + 248.

### 2.3 HumanAct12, 1,191 clips — the key handling
**⚠ HumanML3D does NOT use raw action-to-motion HumanAct12.** From HumanML3D's `raw_pose_processing.ipynb`
(cell 13): *"The source data from **HumanAct12** is already included in `./pose_data` in this repository. You need
to **unzip** it."* — and its crop loop applies **no** transform to humanact12 (no start/end crop, no `x*=-1`,
no left/right swap). So HumanML3D **re-processes** action-to-motion's HumanAct12 into its own coordinate frame
and distributes the result as **`pose_data/humanact12.zip`** in the HumanML3D repo.
- **Wrong source (caught by gate):** the raw action-to-motion Drive folder (`1TBY2x…`, 24-joint positions) gave
  guofeats vs `new_joint_vecs/000001` **mean diff 0.30**, and all four flip variants (x-inv / L-R swap / both)
  gave the *identical* residual → not an orientation flip, a genuinely different source.
- **Right source:** `curl` HumanML3D `pose_data/humanact12.zip` → `datasets/_humanact12_h3d/extracted/humanact12/
  humanact12/*.npy` (1,191 clips, 24-joint SMPL positions). **Source gate PASSED:** `joints_to_guofeats(clip)` vs
  `new_joint_vecs/000001` → root/positions/rot6D/velocity all ≤ **0.037** (floating-point match); the only
  residual is **2/35 foot-contact frames** flipping (velocity-threshold binarization — negligible, and
  DATASETS.md explicitly notes this "small difference").
- The earlier `datasets/_humanact12_raw` (action-to-motion) is **SUPERSEDED — do not use it.**
- **Remaining (GPU):** these are SMPL *positions*; our corpus needs 34-joint 6-D *rotations*, so convert
  `positions → joints2smpl (learned_baselines/bvh2tmr_pipeline/joints2smpl/fit_seq.py) → SMPL params → SMPL→BVH→
  CMU-34`. Per-clip fidelity gates: reconstruction MPJPE (fitted SMPL joints vs input) + left/right-semantic
  check via HumanAct12 action labels (`A0101`=squat, etc.).

---

## 3. Shared finish (pending, each gated)
1. **Crop** every retargeted clip per `annotations.json` `start`/`end` → the 14,614 clip-for-clip HumanML3D set
   (handles the 2,341 multi-crop sources). GATE: count reconciles to 14,614.
2. **Caption-align** all clips via `annotations.json`.
3. **Banks + norm-stats:** rebuild DistilBERT-token + MPNet banks over the full captioned set (extend
   `probes/distilbert_token_precompute.py` / `text_precompute.py`); recompute `norm_stats` on the unified corpus.
4. **Held-out + reconciliation:** verify LMA eval clips (`lma_perform_reorganized_cmu`) are absent; final count =
   14,614 (± documented drops).
- **Naming:** `datasets/hml3d_cmu_flat` (+ `hml3d_cmu_captioned.json`, `hml3d_text_{distilbert_tokens,mpnet}.pt`,
  `norm_stats_hml3d.npz`). Do NOT overwrite cut1 artifacts. Update [[reference_canonical_results_tree]] when built.

---

## 4. Governing principle — why a multi-route corpus is NOT arbitrary
**Rule: use the highest-fidelity *available* source per dataset, retargeted to one common 34-joint 6-D output.**
- AMASS subsets → SMPL-H (their only distributed form; AMASS exists to unify disparate mocap into SMPL).
- CMU → raw CMU mocap (CMU-native → CMU target: no cross-skeleton SMPL step, no SMPL-fit error → *higher* fidelity
  for CMU, not merely convenient).
- HumanAct12 → HumanML3D's own re-processed release (matches the reference by construction).

Not per-dataset whim; it does **not** open a "raw everything" slope (the AMASS subsets' raw forms are incompatible
skeletons — raw would be *more* inconsistent). HumanML3D itself is multi-route (AMASS + HumanAct12), so this adds
no arbitrariness it lacks. **Consistency is guaranteed at the OUTPUT (34-joint 6-D, 30 fps) + a per-route fidelity
gate.** (Strict single-route alternative — AMASS-CMU credentialed re-download for byte-parity on CMU — was rejected:
it *adds* SMPL-fit error to CMU and needs AMASS credentials, for ~20% of the corpus.)

## 5. Lessons — two wrong-source traps (methods-appendix material)
Both were caught by fidelity gates *before* corrupting the corpus, and both are easy to fall into:
1. **CMU:** the obvious cgspeed "Daz-friendly" release has a Poser skeleton incompatible with the standard-CMU
   pipeline; the **Motionbuilder-friendly** release is the correct one.
2. **HumanAct12:** the obvious action-to-motion release is *not* what HumanML3D uses; HumanML3D ships its own
   re-processed `pose_data/humanact12.zip`. Matching shapes ≠ matching data — only the guofeats-vs-reference gate
   revealed it.
Takeaway: **byte/float-level agreement with the reference dataset's own artifacts is the only reliable source
check**; shape and count checks are necessary but not sufficient.

## 6. Scripts & key paths (all on chimera under `.../virtual_reality/triplets/`)
`align_corpus_v2.py` (AMASS caption coverage) · `mediafire_dl_mb.py` (CMU mb2 fetch) · `bvhconv_lib.py` (sanitized
converter) · `cmu_roundtrip.py`, `cmu_convert_248.py` (CMU gate + convert) · `ha12_gate.py` (HA12 source gate).
Data: `datasets/{amass_cmu_flat, cmu_all_perform, _cmu_mb_raw, _cmu_248_out, _humanact12_h3d, _cmu_missing_ids.txt}`.
Keystone: `learned_baselines/tmr/repo/datasets/annotations/humanml3d/annotations.json`. Reference:
`…/HumanML3D/new_joint_vecs/000001.npy`. joints2smpl: `learned_baselines/bvh2tmr_pipeline/joints2smpl/`.
*(For the repro bundle these scripts should be version-controlled alongside this doc.)*
