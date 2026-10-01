# Unified Corpus Build — CMU-adapted HumanML3D (reproducibility record)

**Goal.** Retarget **every HumanML3D clip** to our **34-joint, 6-D-rotation, CMU-skeleton, 30 fps** format,
caption-aligned, banks + norm-stats, held-out-clean — one unified corpus for all Tier-1 models that **matches
published TMR's training set by construction** (so a Tier-1 TMR-objective vs published-TMR comparison isolates
representation/conversion, not corpus). Decided 2026-09-26: do it right, gate every route.

**Scope.** We match the **full HumanML3D corpus** — i.e. **TMR's HumanML3D benchmark**, the larger of TMR's two
training sets. ⚠ This is **not** the smaller, KIT-only **KIT-ML** benchmark: HumanML3D re-captions the *entire*
AMASS collection **+ HumanAct12**, so every constituent below (CMU, BMLmovi, Eyes_Japan, … and KIT) is
text-labeled — KIT is just the single largest subset *within* HumanML3D, not the whole corpus. The keystone
defining the exact clip set + crops + captions is TMR's
`learned_baselines/tmr/repo/datasets/annotations/humanml3d/annotations.json` (`id → path, duration,
start/end, captions`; all 29,228 entries carry ≥1 caption — verified).

**14,614 clips** (×2 with L/R mirrors = 29,228 annotation ids), from **18 constituent source datasets** grouped
into **3 processing routes** by highest-fidelity available source (see §4):

| Constituent dataset | clips | Processing route |
|---|---:|---|
| KIT | 4,647 | **R1 — AMASS SMPL-H** → SMPL-24 BVH → CMU-34 |
| CMU | 2,913 | **R2 — CMU-native** cgspeed BVH → CMU-34 (byte-exact round-trip) |
| BMLmovi | 1,839 | R1 |
| Eyes_Japan_Dataset | 1,465 | R1 |
| humanact12 | 1,191 | **R3 — positions → SMPLify fit** → SMPL → BVH → CMU-34 |
| MPI_HDM05 | 771 | R1 |
| BioMotionLab_NTroje | 373 | R1 |
| EKUT | 350 | R1 |
| ACCAD | 277 | R1 |
| DFaust_67 | 135 | R1 |
| MPI_Limits | 132 | R1 |
| MPI_mosh | 122 | R1 |
| Transitions_mocap | 110 | R1 |
| TotalCapture | 74 | R1 |
| SFU | 68 | R1 |
| BMLhandball | 67 | R1 |
| HumanEva | 50 | R1 |
| SSM_synced | 30 | R1 |
| **TOTAL** | **14,614** | R1 = 10,510 · R2 = 2,913 · R3 = 1,191 |

By provenance 13,423 clips are **AMASS** (of which CMU's 2,913 take the higher-fidelity CMU-native route **R2**,
the other 10,510 take the SMPL-H route **R1**) and 1,191 are **HumanAct12** (route **R3**). Routes are detailed
in §2; all three land in the **same** 34-joint 6-D CMU / 30 fps output so clips are format-identical regardless of
constituent.

**Status (2026-09-30).**

| Source | Clips | State |
|---|---|---|
| AMASS (non-CMU) | 10,510 src | ✅ aligned to captions (99.99%) |
| CMU | all HumanML3D CMU | ✅ complete — round-trip **byte-exact** |
| HumanAct12 | 1,191 | ✅ source validated **+ SMPL fit COMPLETE** (1,191/1,191, MPJPE median 3.02 cm); SMPL→CMU-34 back-half **pending** |
| shared finish | — | crop-per-annotation, caption-align, banks, norm-stats, held-out — **pending** |

---

## 1. Provenance ledger (per source — the required artifact)

| Source | Exact release / file | Retargeting route → 34-joint 6D CMU @30fps | Fidelity gate | Result |
|---|---|---|---|---|
| **AMASS non-CMU** (KIT, BMLmovi, Eyes_Japan, MPI_HDM05, BioMotionLab, EKUT, ACCAD, …) | AMASS SMPL-H (already retargeted → `datasets/amass_cmu_flat`, 13,647) | `SMPL-H → SMPL-24 BVH (smpl2bvh) → CMU-34, lerp/slerp resample →30fps` (`amass/*`) | `annotations.json` path → retargeted filename, exact per-source (`align_corpus_v2.py`) | **10,509/10,510 (99.99%)**, 1 edge miss |
| **CMU** | cgspeed **Motionbuilder-friendly** BVH (`cmuconvert-mb2-*`, standard CMU skeleton) + existing `cmu_all_perform` | `bvhConverterToPerform.prepare_files`: drop `Neck/RThumb/LThumb` → `adjustFrameRate(30)` (120→30, 4:1) → `fix_end_sites` | round-trip: convert mb `80_63`, compare to `cmu_all_perform/80_63.bvh` (`cmu_roundtrip.py`) | **MAX abs diff = 0** (28 joints, 568 frames) |
| **HumanAct12** | **HumanML3D `pose_data/humanact12.zip`** (24-joint SMPL positions, HumanML3D frame) — *not* raw action-to-motion | `[:22] + scale→SMPL + ground → joints2smpl SMPLify3D (batched) → SMPL params → SMPL→BVH→CMU-34` (`ha12_batch_fit.py`) | (1) source gate `joints_to_guofeats` vs `new_joint_vecs/000001`; (2) per-clip reconstruction MPJPE | (1) non-feet max 0.037, feet 2/35 frames; (2) **MPJPE mean 3.09 / median 3.02 / p95 4.29 / max 6.69 cm** (n=1,191; >5 cm = 1%) |

Output convention shared by every route: **ROOT Hips + 28 articulated joints** (round-trip-verified identical
hierarchy between `amass_cmu_flat` and `cmu_all_perform`), **Frame Time 0.033333 (30 fps)**.

### 1.1 Dataset origins (upstream of the release we consumed)
- **HumanML3D (keystone — clip set, crops, captions).** EricGuo5513/HumanML3D — <https://github.com/EricGuo5513/HumanML3D>
  (Guo et al., CVPR 2022). The exact id→path/start/end/captions we align to is TMR's mirror of these annotations:
  Mathux/TMR — <https://github.com/Mathux/TMR> → `datasets/annotations/humanml3d/annotations.json` (*what published
  TMR trained on*). HumanML3D itself is derived from AMASS + HumanAct12, which is why our corpus is multi-route.
- **AMASS subsets** (KIT, BMLmovi, Eyes_Japan, MPI_HDM05, BioMotionLab/BMLrub, EKUT, ACCAD, TotalCapture, …).
  AMASS project, MPI-IS — <https://amass.is.tue.mpg.de/> (Mahmood et al., ICCV 2019). Distributed as per-subset
  **SMPL-H `.npz`**; free account/registration required. This is each subset's only unified distributed form.
- **CMU.** Original mocap: **CMU Graphics Lab Motion Capture Database** — <http://mocap.cs.cmu.edu/> (free, funded by
  NSF EIA-0196217). The BVH conversion we consume is the **cgspeed** re-release (B. Hahne), *Motionbuilder-friendly*
  variant (standard CMU skeleton) — <https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/cmu-bvh-conversion>;
  the `cmuconvert-mb2-*.zip` archives are hosted on **MediaFire** (fetched via `mediafire_dl_mb.py`). *(The co-hosted
  "Daz-friendly" variant is the wrong skeleton — see §5.)*
- **HumanAct12.** Original: **Action2Motion** (Guo et al., ACM MM 2020) — <https://ericguo5513.github.io/action-to-motion/>
  (raw 24-joint positions, Google Drive folder `1TBY2x…`). **The version we use is NOT this raw source** but
  HumanML3D's own re-processed **`pose_data/humanact12.zip`** shipped inside the HumanML3D repo above (see §2.3/§5).

---

## 2. Conversion recipe — every source, reproducibly

### 2.1 Route R1 — AMASS SMPL-H (16 subsets, 10,510 clips: KIT, BMLmovi, Eyes_Japan, MPI_HDM05, BioMotionLab_NTroje, EKUT, ACCAD, DFaust_67, MPI_Limits, MPI_mosh, Transitions_mocap, TotalCapture, SFU, BMLhandball, HumanEva, SSM_synced)
Already retargeted in `datasets/amass_cmu_flat` (13,647 clips ⊇ HumanML3D's AMASS; the extra are non-HumanML3D
AMASS subsets like GRAB/DanceDB, harmless — we intersect with `annotations.json`). Pipeline: `amass/
amass_smplh_to_bvh_batch.py` (SMPL-H `.npz` → SMPL-24 BVH via smpl2bvh, R_x(−90°) root fix, per-clip gender/fps)
→ `retarget_amass_to_cmu_batch.py` → resample to 30 fps. **Caption alignment:** `align_corpus_v2.py` normalizes
`annotations.json` paths (`EKUT/265/SLP102_poses` → `EKUT_265_SLP102`) to our filenames
(`EKUT_265_SLP102_cmu33_30fps`); **10,509/10,510 resolve**. Note: 2,341 source clips map to multiple HumanML3D
ids (distinct start/end + captions) → the shared **crop stage (§3)** slices per `annotations.json`.

### 2.2 Route R2 — CMU-native (CMU, 2,913 clips)
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

### 2.3 Route R3 — HumanAct12 positions→SMPL fit (humanact12, 1,191 clips) — the key handling
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

**Fit: positions → SMPL rotations (COMPLETE, 2026-09-30).** HumanAct12 is the only source distributed as
3-D joint *positions*, not rotations, so each clip is fit to the SMPL body model (neutral gender) via
**SMPLify3D** (joints2smpl), `joint_category="AMASS"` using the first 22 SMPL joints, 150 LBFGS iterations.
Output kept = per-frame `poses[F,72]` (axis-angle) + `trans[F,3]`; **betas are left neutral** because the
downstream `smpl2bvh` emits a fixed-shape SMPL skeleton and the retarget imposes the CMU skeleton — so only the
**rotations** propagate to the corpus. Pre-processing per clip: slice to 22 joints, **scale-normalize** (below),
then ground (subtract the constant `[root_x₀, min_y_over_clip, root_z₀]` → frame-0 root at XZ origin, lowest
point on the floor; preserves all relative motion and heading). Driver:
`learned_baselines/bvh2tmr_pipeline/joints2smpl/ha12_batch_fit.py` (batched — each clip fit in 128-frame chunks
in one optimization for GPU throughput; frames fit independently from the mean-pose init).

**Scale normalization — the key fidelity step.** HumanAct12 skeletons vary in overall size per actor
(pelvis→neck 0.43–0.55 m; stature ≈ 1.1–1.5 m) whereas SMPL-neutral is fixed (~1.7 m, pelvis→neck ≈ 0.51 m).
When a clip's skeleton is far from SMPL size, shape parameters cannot absorb the gap and the fit leaves a large
residual that *corrupts the recovered joint angles* — the fit, not the data, is at fault. We therefore rescale
each clip so its mean pelvis→neck length equals **0.51 m** before fitting (`scale = 0.51 / mean‖J₁₂ − J₀‖`,
stored per clip). This is lossless for our purpose (rotations are scale-invariant; the retarget sets absolute
size) and has the side benefit of placing HumanAct12 at the same metric scale as the SMPL-native AMASS clips.
Diagnosis that isolated scale as the cause (on the high-MPJPE tail): more iterations did not help
(150 ≡ 300, i.e. converged), and per-frame warm-start barely helped (7.65 → 7.38 cm at ~9× the cost), whereas
scaling halved the error (7.65 → 3.94 cm); clips already at SMPL scale were unchanged (3.00 → 3.02 cm).

**Result (gate 2 — reconstruction MPJPE, fitted SMPL joints vs. input, n = 1,191):**
mean **3.09 cm**, median **3.02 cm**, p95 **4.29 cm**, max **6.69 cm**; only **1 %** of clips exceed 5 cm
(vs. 24 % without scaling) and 9 % exceed 4 cm (vs. 38 %). Scale factors spanned 0.89–1.21 (median 1.02).
Outputs: `datasets/_humanact12_smpl_scaled/<clip>.npz` (`poses, trans, mpjpe, scale, fps=20`). Compute: pomplun
H200 + chimera24 H200-MIG array (`ha12_fit_array.sbatch` / `ha12_fit_mig.sbatch`); ~50 s/clip full-GPU,
~80–120 s/clip on a 35 GB MIG slice. (A transient shared-scratch write failure killed several MIG shards
mid-run; the driver skips already-written clips, so a resubmit resumed cleanly.)

- **Remaining (back half, CPU, pending):** `SMPL params → pack + resample 20→30 fps (reuse amass
  resample_rotvec/resample_trans) → smpl2bvh (--gender NEUTRAL --fps 30) → retarget_amass_to_cmu_batch.py →
  CMU-34`, identical to the AMASS route's second half so output lands format-identical to `amass_cmu_flat`.

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

## 5. Lessons — fidelity traps (methods-appendix material)
All caught by fidelity gates *before* corrupting the corpus, and all easy to fall into:
1. **CMU wrong release:** the obvious cgspeed "Daz-friendly" release has a Poser skeleton incompatible with the
   standard-CMU pipeline; the **Motionbuilder-friendly** release is the correct one.
2. **HumanAct12 wrong source:** the obvious action-to-motion release is *not* what HumanML3D uses; HumanML3D ships
   its own re-processed `pose_data/humanact12.zip`. Matching shapes ≠ matching data — only the
   guofeats-vs-reference gate revealed it.
3. **HumanAct12 fit scale:** fitting SMPL to raw-scale positions left a large residual on ~a quarter of clips
   (>5 cm MPJPE) that corrupted the recovered *rotations*. The cause was **skeleton-size mismatch**, not optimizer
   settings — confirmed by the MPJPE gate: more iterations were inert (converged) and per-frame warm-start barely
   moved it, but **per-clip scale-normalization to SMPL size halved the error** and cut the >5 cm tail from 24 % to
   1 %. Lesson: when a model-fitting step leaves a structured residual, suspect a *scale/units* mismatch before
   reaching for more compute; gate on reconstruction error, not on convergence.
Takeaway: **byte/float-level agreement with the reference dataset's own artifacts is the only reliable source
check**; shape and count checks are necessary but not sufficient.

## 6. Scripts & key paths
**Version-controlled** in `repro_bundle/06_corpus_build/` (see its README): `align_corpus_v2.py` (AMASS caption
coverage) · `mediafire_dl_mb.py` (CMU mb2 fetch) · `cmu_roundtrip.py`, `cmu_convert_248.py` (CMU gate + convert) ·
`ha12_gate.py` (HA12 source gate). Regenerated on chimera: `bvhconv_lib.py` = `head -377 bvhConverterToPerform.py`.
HA12 fit (on chimera, `learned_baselines/bvh2tmr_pipeline/joints2smpl/`): `ha12_batch_fit.py` (scale-norm +
batched SMPLify), `ha12_fit_array.sbatch` / `ha12_fit_mig.sbatch` (pomplun / H200-MIG arrays),
`ha12_fit_diag.py` + `ha12_scale_test.py` + `ha12_scale_test2.py` (the warm-start / iters / scale diagnostics);
SMPL→BVH tool re-cloned at `fit3d/third_party/smpl2bvh`. *(TODO: copy the HA12 fit scripts into
`repro_bundle/06_corpus_build/` for the release.)*
Chimera data (`.../virtual_reality/triplets/`): `datasets/{amass_cmu_flat, cmu_all_perform, _cmu_mb_raw,
_cmu_248_out, _humanact12_h3d, _humanact12_smpl_scaled, _cmu_missing_ids.txt}`. Keystone:
`learned_baselines/tmr/repo/datasets/annotations/humanml3d/annotations.json`. Reference:
`…/HumanML3D/new_joint_vecs/000001.npy`. joints2smpl: `learned_baselines/bvh2tmr_pipeline/joints2smpl/`.
