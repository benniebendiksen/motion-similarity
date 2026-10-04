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
| CMU | 2,913 | **R2 — AMASS-CMU (SMPL-H), same route as R1** (corrected from cgspeed for TMR-consistency; co-register 100%) |
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

**Status (2026-10-04) — CORPUS COMPLETE.**

| Source | Clips | State |
|---|---|---|
| AMASS (non-CMU) | 10,510 src | ✅ aligned to captions (99.99%) |
| CMU | 2,035 HumanML3D | ✅ **COMPLETE via AMASS-CMU** (SMPL+H G) — co-register 100%, MISSING=0 (cgspeed superseded) |
| HumanAct12 | 1,191 | ✅ **COMPLETE** — SMPL fit (median 3.02 cm) + back-half; 1,191/1,191 CMU BVH (hierarchy == `amass_cmu_flat`) |
| shared finish | — | ✅ **COMPLETE + VERIFIED** — 14,612-clip `hml3d_cmu_flat` (1 NaN dropped) + caption (40,384 caps) + DistilBERT/MPNet banks + `norm_stats_hml3d` + held-out PASS + format-identity |

---

## 1. Provenance ledger (per source — the required artifact)

| Source | Exact release / file | Retargeting route → 34-joint 6D CMU @30fps | Fidelity gate | Result |
|---|---|---|---|---|
| **AMASS non-CMU** (KIT, BMLmovi, Eyes_Japan, MPI_HDM05, BioMotionLab, EKUT, ACCAD, …) | AMASS SMPL-H (already retargeted → `datasets/amass_cmu_flat`, 13,647) | `SMPL-H → SMPL-24 BVH (smpl2bvh) → CMU-34, lerp/slerp resample →30fps` (`amass/*`) | `annotations.json` path → retargeted filename, exact per-source (`align_corpus_v2.py`) | **10,509/10,510 (99.99%)**, 1 edge miss |
| **CMU** (=AMASS-CMU) | **AMASS `CMU.tar.bz2` (SMPL+H G)** → `_amass_cmu_raw/CMU/<subj>/<clip>_poses.npz` (2,088 clips) — *corrected from cgspeed, which is a different conversion/extent* | **R1 route**: `amass_smplh_to_bvh_batch.py` → SMPL-24 BVH → `retarget_amass_to_cmu_batch.py` → CMU-33 | **co-registration**: annotation `duration` ≈ clip duration (`_verify_cmu_coreg.py`) | **100%** ≤0.1 s (`80_63` 37.883 vs 37.900 s); cgspeed was 74% |
| **HumanAct12** | **HumanML3D `pose_data/humanact12.zip`** (24-joint SMPL positions, HumanML3D frame) — *not* raw action-to-motion | `[:22] + scale→SMPL + ground → joints2smpl SMPLify3D (batched) → SMPL params → SMPL→BVH→CMU-34` (`ha12_batch_fit.py`) | (1) source gate `joints_to_guofeats` vs `new_joint_vecs/000001`; (2) per-clip reconstruction MPJPE | (1) non-feet max 0.037, feet 2/35 frames; (2) **MPJPE mean 3.09 / median 3.02 / p95 4.29 / max 6.69 cm** (n=1,191; >5 cm = 1%) |

Output convention shared by every route: **ROOT Hips + 28 articulated joints** (round-trip-verified identical
hierarchy between `amass_cmu_flat` and `cmu_all_perform`), **Frame Time 0.033333 (30 fps)**.

### 1.1 Dataset origins (upstream of the release we consumed)
- **HumanML3D (keystone — clip set, crops, captions).** EricGuo5513/HumanML3D — <https://github.com/EricGuo5513/HumanML3D>
  (Guo et al., CVPR 2022). The exact id→path/start/end/captions we align to is TMR's mirror of these annotations:
  Mathux/TMR — <https://github.com/Mathux/TMR> → `datasets/annotations/humanml3d/annotations.json` (*what published
  TMR trained on*). HumanML3D itself is derived from AMASS + HumanAct12, which is why our corpus is multi-route.
- **AMASS subsets — including CMU** (KIT, BMLmovi, Eyes_Japan, MPI_HDM05, BioMotionLab/BMLrub, EKUT, ACCAD,
  TotalCapture, **CMU**, …). AMASS project, MPI-IS — <https://amass.is.tue.mpg.de/> (Mahmood et al., ICCV 2019).
  Distributed per-subset as **SMPL+H `.npz`** (download the **"SMPL+H G"** file per dataset); account/registration
  required. This is the source HumanML3D/TMR actually use, so **CMU is obtained here too** (`CMU.tar.bz2`, SMPL+H G),
  *not* from a CMU-specific BVH release.
- **CMU — superseded origin (do not use):** the **cgspeed** BVH re-release (B. Hahne, *Motionbuilder-friendly*) of
  the CMU Graphics Lab database (<http://mocap.cs.cmu.edu/>) via MediaFire. It is a *different conversion/extent*
  than TMR's AMASS-CMU and was dropped for TMR-consistency (§2.2, §5 lesson 4). Scripts (`mediafire_dl_mb.py`,
  `cmu_roundtrip.py`, `cmu_convert_248.py`) retained for provenance only.
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

### 2.2 Route R2 — CMU via **AMASS-CMU (SMPL-H)** — same route as R1 (CMU, 2,913 clips)
**⚠ CORRECTED 2026-10-04 for TMR-consistency (supersedes the earlier cgspeed CMU-native build).**
HumanML3D is built from **AMASS + HumanAct12**, and AMASS *contains* CMU — so TMR's "CMU/…" clips are the
**AMASS-CMU subset (SMPL-H / MoSh++ fit)**, *not* the cgspeed BVH re-release. CMU therefore goes through the
**identical R1 pipeline** as every other AMASS subset: AMASS-CMU `CMU.tar.bz2` (SMPL+H G, from
amass.is.tue.mpg.de) → `datasets/_amass_cmu_raw/CMU/<subj>/<clip>_poses.npz` (2,088 clips, SMPL-H 156-dim) →
`amass_smplh_to_bvh_batch.py` → SMPL-24 BVH → `retarget_amass_to_cmu_batch.py` → CMU-33 @30fps
(`datasets/_amass_cmu_cmu/`). *(Gender models unavailable on-host: `SMPL_FEMALE/MALE.pkl` symlink →
`SMPL_NEUTRAL.pkl`; harmless, because the smpl2bvh npz carries no betas and Stage B transfers rotations + imposes
the CMU skeleton, so body proportions are normalized away — motion content/extent are gender-independent.)*

**Gate (co-registration):** annotation `duration` vs retargeted clip duration. **AMASS-CMU = 100%** within 0.1 s
(e.g. `80_63`: annotation 37.883 s vs 37.900 s). This is the decisive TMR-faithfulness check.

**⌧ SUPERSEDED — the cgspeed CMU-native build (do NOT use for the corpus).** We first built CMU from the cgspeed
**Motionbuilder-friendly** BVH release (fetched via `mediafire_dl_mb.py`; the Daz-friendly release has the wrong
Poser skeleton), validated by a **byte-exact round-trip** (`cmu_roundtrip.py`, MAX abs diff 0 vs `cmu_all_perform`;
248 gap-fills via `cmu_convert_248.py`). That work was *fidelity-correct but TMR-wrong*: cgspeed is a **different
conversion of the same performances** with a **different extent** — `80_63` is 18.9 s (cgspeed) vs 37.9 s
(AMASS/HumanML3D), and only **74%** of cgspeed CMU co-registered. See §5 lesson 4.

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

**Back half: SMPL params → CMU-34 BVH (COMPLETE, 2026-10-02).** Two stages, identical to the AMASS route's
second half so output lands format-identical to `amass_cmu_flat`:
- **Stage A** (`ha12_params_to_bvh.py`): `poses[F,72] → [F,24,3]` (already 24-joint SMPL axis-angle — no
  palm-pad; **no Z-up→Y-up basis change**, since the fit was done in Y-up, unlike AMASS) → resample 20→30 fps
  (slerp rotations / lerp trans) → `smpl2bvh --gender NEUTRAL --fps 30` → SMPL-24 BVH. ⚠ the driver sets
  `MKL_THREADING_LAYER=GNU` or the smpl2bvh subprocess aborts (MKL/libgomp clash when the numpy-importing parent
  spawns it). → 1,191/1,191, 24 joints.
- **Stage B** (`retarget_amass_to_cmu_batch.py`, the R1 retargeter verbatim): SMPL-24 BVH → CMU-33 →
  `datasets/_humanact12_cmu/`. **1,191/1,191** (`ok=1191 skipped=0 failed=0`); aggregate gate: all **28 joints +
  30 fps**, joint hierarchy **identical to `amass_cmu_flat`**, and **Y-up/upright** (orientation gate: HA12 and
  AMASS both have tallest-axis=y, head above root — no flip despite HA12 skipping the basis change).
  Compute: chimera24 H200-MIG (6-shard Stage A array + chained Stage B); env `j2s_gpu` (has glm+torch+smplx).

---

## 3. Shared finish (COMPLETE 2026-10-04) — `build_hml3d_stage1.py` + bank/norm copies
1. **Flatten + intersect** the 3 routes to HumanML3D's exact set (resolve each `annotations.json` source → its
   route BVH; drop non-HumanML3D AMASS extras). → **14,612 clips** (14,614 − 1 KIT edge-drop
   `KIT/9/WalkInCounterClockwiseCircle07` − 1 NaN clip `KIT_1226_Trial_62` dropped post-verification).
2. **Crop to exact annotation windows** (now safe — all routes co-register): whole-clip windows symlinked (8,393),
   genuine sub-span windows cropped (6,220). GATE: 0 clips with ≤0 frames (the earlier garbage was CMU
   misregistration, cured by the AMASS-CMU route); lengths min 3 / median 225 / max 704 frames.
3. **Caption-align** → `datasets/hml3d_captioned.json` (**14,612** entries after the NaN drop; 40,384 unique captions).
4. **Banks** (`probes/{distilbert_token_precompute,text_precompute}_hml3d.py`, tmr env, GPU): DistilBERT-token bank
   = 40,384 captions (full coverage); MPNet bank = 14,613 clips (full coverage).
5. **Norm-stats** (`compute_norm_hml3d.py`, canonical builder + DEGENERATE_EPS=0.05): `norm_stats_hml3d.npz`,
   204 channels, 14,612 clips (1 NaN-clip skipped). ⚠ **92/204 degenerate channels (std<0.05)** vs cut1's 45 —
   ~14 fully-constant joints align with the retarget's structural stubs (clavicles absorbed, palm stubs,
   near-identity LowerBack); ties to the normalization-lever note ([[project_input_readout_findings]]).
6. **Held-out: PASS** — 0 overlap between the corpus clips and the 486 LMA eval clips (`lma_perform*`).
- **Artifacts:** `datasets/hml3d_cmu_flat/` (14,612) + `hml3d_captioned.json` + `hml3d_text_{distilbert_tokens,mpnet}.pt`
  + `norm_stats_hml3d.npz`. cut1 artifacts untouched. Update [[reference_canonical_results_tree]] when models train on it.
  *(Banks were computed pre-drop (14,613); the dropped NaN clip leaves 1 inert stale entry in the MPNet bank — the
  dataloader lists the BVH dir so it is never loaded.)*

### 3.1 Verification (`_corpus_verify.py`, 2026-10-04)
- **Format identity — perfect:** all 14,612 clips share one 33-joint hierarchy (0 mismatches), 30 fps, 0 missing,
  0 empty-caption.
- **Value sanity:** rotation channels (joints 1–33) all within [−1, 1]; the ±16 extremes are root (joint 0)
  translation (world units) — expected.
- **NaN:** exactly 1 (`KIT_1226_Trial_62`) → dropped.
- **Per-route degeneracy is STRUCTURAL, not HA12:** R1 91 / R2 92 / R3 99 of 204 — uniform across routes, so the
  degenerate channels are the retarget's structural stubs, not HA12 fit stiffness. The 45→92 rise vs cut1 is from
  *correcting* the corpus (cut1 wrongly carried extra non-HumanML3D subsets whose diversity lifted channels above
  the floor); the leaner HumanML3D-only set is more homogeneous. Not a regression.

---

## 4. Governing principle — **match TMR's actual sources** (TMR-consistency over source-fidelity)
**Rule (revised 2026-10-04): use the SAME source HumanML3D/TMR used for each dataset, so every clip co-registers
with the annotations and all models (incl. our Tier-1 TMR) train on the *same* motions TMR did.**
- AMASS subsets (incl. **CMU**) → **AMASS SMPL-H** (what HumanML3D is built from). CMU is an AMASS subset, so it
  takes the identical SMPL-H route — *not* a separate CMU-native source.
- HumanAct12 → HumanML3D's own re-processed `pose_data` release (matches the reference by construction).
- Output consistency (34-joint 6-D, 30 fps) is guaranteed by the shared retarget; **TMR-consistency is verified by
  a per-clip co-registration gate** (annotation `duration` ≈ our clip duration): AMASS-CMU 100%, HA12 100%.

**Why this supersedes the earlier "highest-fidelity available source" rule.** That rule optimized fidelity to
*some* faithful rendering of each performance, and so chose the cgspeed CMU-native BVH for CMU (byte-exact
round-trip). But fidelity-to-a-source is the wrong objective: cgspeed is a *different conversion with a different
extent* than TMR's AMASS-CMU (18.9 s vs 37.9 s on `80_63`; 26% of CMU failed co-registration), so it would have
trained our models on motions TMR never saw. **Consistency-with-TMR wins over source-fidelity.** The AMASS-CMU
credentialed re-download — *rejected* under the old rule (adds SMPL-fit error, needs credentials) — is **adopted**
here: it is the only TMR-faithful CMU source, and the user supplied the AMASS credentials (SMPL+H G). The residual
SMPL-fit/retarget error is the price of using TMR's actual data, and it is uniform across all AMASS subsets.

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
4. **CMU: byte-exact to the *wrong* source.** The cgspeed CMU build round-tripped **byte-exact** — maximally
   *faithful*, yet TMR-**wrong**, because TMR's CMU is the **AMASS-CMU** subset, a different conversion with a
   different extent (`80_63` 18.9 s vs 37.9 s; 26% of cgspeed CMU failed co-registration). A byte-perfect gate on
   the wrong target passes while the corpus silently diverges from TMR. Lesson: **fidelity and consistency are
   different objectives** — gate on **co-registration with the reference's own timeline** (annotation `duration`
   ≈ clip duration), not just internal fidelity; "faithful conversion" is meaningless until the *source* is the
   one the reference actually used.
Takeaway: **agreement with the reference dataset's own artifacts — both its data (float-level) AND its timeline
(co-registration) — is the only reliable source check**; shape, count, and even byte-exactness to *a* source are
necessary but not sufficient.

## 6. Scripts & key paths
**All three routes are version-controlled** in `repro_bundle/06_corpus_build/` (see its README for the per-route
table + deps):
- **R1 (AMASS):** `amass_smplh_to_bvh_batch.py` (SMPL-H→SMPL-24 BVH), `retarget_amass_to_cmu_batch.py`
  (BVH→CMU-33), `align_corpus_v2.py` (caption coverage).
- **R2 (CMU via AMASS-CMU):** `run_cmu_stageA.sbatch` (AMASS-CMU SMPL-H → SMPL-24 BVH, sharded) +
  `run_cmu_stageB.sbatch` (retarget → CMU-33), reusing R1's `amass_smplh_to_bvh_batch.py` +
  `retarget_amass_to_cmu_batch.py`; `_verify_cmu_coreg.py` (co-registration gate). *Superseded (provenance only):*
  `mediafire_dl_mb.py`, `cmu_roundtrip.py`, `cmu_convert_248.py` (+ `bvhReader/bvhConverterToPerform.py`).
- **R3 (HumanAct12):** `ha12_gate.py`, `ha12_batch_fit.py` (scale-norm + batched SMPLify),
  `ha12_fit_array.sbatch` / `ha12_fit_mig.sbatch`, the diagnostics
  `ha12_fit_diag.py` / `ha12_scale_test.py` / `ha12_scale_test2.py`, and the **back half**
  `ha12_params_to_bvh.py` (+ `ha12_stageA.sbatch` / `ha12_stageB.sbatch`).
- **Shared finish:** `build_hml3d_stage1.py` (flatten+intersect+window-crop+caption → `hml3d_cmu_flat` +
  `hml3d_captioned.json`), `probes/{distilbert_token_precompute,text_precompute}_hml3d.py` (banks),
  `compute_norm_hml3d.py` (norm-stats), `_heldout_check.py`.
- **Shared core:** `bvhReader/` — third-party BVH library (alinen/bvh-python, **GPL-v3**, LICENSE included),
  extended for retargeting; `retarget.py` is called by every route. External deps: PyGLM, torch, numpy, matplotlib.
- Still **chimera-only** (environment, not code): the `smpl2bvh` clone at `fit3d/third_party/smpl2bvh` and the
  joints2smpl assets (SMPL-neutral model, `neutral_smpl_mean_params.h5`, `gmm_08.pkl`).
Chimera data (`.../virtual_reality/triplets/`): `datasets/{amass_cmu_flat, cmu_all_perform, _cmu_mb_raw,
_cmu_248_out, _humanact12_h3d, _humanact12_smpl_scaled, _cmu_missing_ids.txt}`. Keystone:
`learned_baselines/tmr/repo/datasets/annotations/humanml3d/annotations.json`. Reference:
`…/HumanML3D/new_joint_vecs/000001.npy`. joints2smpl: `learned_baselines/bvh2tmr_pipeline/joints2smpl/`.
