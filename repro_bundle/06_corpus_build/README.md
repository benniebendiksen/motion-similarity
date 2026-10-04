# 06 — Unified corpus build (CMU-adapted HumanML3D)

Conversion + fidelity-gate scripts for retargeting **every HumanML3D clip** (the full corpus TMR trains
on — 14,614 clips across 18 source datasets, **not** the smaller KIT-only KIT-ML set) to our common
**34-joint 6-D CMU / 30 fps** format. Full recipe, constituent-dataset table, provenance ledger, dataset
origins, and lessons: [`docs/context/TMLR/06_CORPUS_BUILD.md`](../../docs/context/TMLR/06_CORPUS_BUILD.md).

The 18 constituents are converted by **three routes** that all emit the identical output skeleton:
- **R1 — AMASS SMPL-H** (10,510 clips: KIT, BMLmovi, Eyes_Japan, MPI_HDM05, BioMotionLab, EKUT, ACCAD,
  DFaust, MPI_Limits, MPI_mosh, Transitions, TotalCapture, SFU, BMLhandball, HumanEva, SSM).
- **R2 — CMU-native** cgspeed BVH (2,913 clips) — higher fidelity than SMPL-H for CMU.
- **R3 — HumanAct12** positions → SMPLify fit (1,191 clips).

**Paths are chimera-specific** (`/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets`); edit the
`TR`/path constants at the top of each script for another host. The AMASS/retarget scripts were invoked
**from the `triplets/` project root** (so `import bvhReader.*` resolves); `bvhReader/` here is that package.

## Scripts by route

| Script | Route | Role | Gate / result |
|---|---|---|---|
| `amass_smplh_to_bvh_batch.py` | R1-A | AMASS SMPL-H `.npz` → **SMPL-24 BVH** (smpl2bvh; `R_x(−90°)·R_y(180°)` Z-up→Y-up root fix; per-clip gender/fps; slerp/lerp resample →30 fps) | emits the SMPL-24 layout R2's CMU route is checked against |
| `retarget_amass_to_cmu_batch.py` | R1-B | SMPL-24 BVH → **CMU-33** via `bvhReader.retarget.retarget_bvh` (T-pose `rest_delta` offset; clavicles/palms dropped) | mirrors AMASS tree → `datasets/amass_cmu_flat` |
| `align_corpus_v2.py` | R1 | Map `annotations.json` paths → retargeted AMASS filenames (caption coverage) | 10,509/10,510 (99.99%) |
| `mediafire_dl_mb.py` | R2 | Server-side fetch of the 9 `cmuconvert-mb2-*.zip` (Motionbuilder-friendly CMU BVH) — decodes `data-scrambled-url`, verifies ZIP magic | 9/9 OK, all 248 ids |
| `cmu_roundtrip.py` | R2 | CMU fidelity gate: convert mb `80_63`, require it to reproduce `cmu_all_perform/80_63.bvh` | MAX abs diff = 0 |
| `cmu_convert_248.py` | R2 | Convert the 248 CMU source ids `cmu_all_perform` lacked, through the round-trip-proven pipeline | 248/248, 28 joints, 30 fps |
| `ha12_gate.py` | R3 | HumanAct12 source-correctness gate: `joints_to_guofeats(clip)` vs HumanML3D `new_joint_vecs/000001` | see note below |
| `ha12_batch_fit.py` | R3 | HumanAct12 positions → SMPL params via SMPLify3D: `[:22]` + **scale→SMPL (pelvis→neck=0.51 m)** + ground → batched fit (B=128 chunks); saves `poses,trans,mpjpe,scale,fps=20` per clip | **n=1,191: MPJPE mean 3.09 / median 3.02 / max 6.69 cm; >5 cm = 1%** |
| `ha12_fit_array.sbatch` / `ha12_fit_mig.sbatch` | R3 | SLURM arrays running `ha12_batch_fit.py` (pomplun H200 / chimera24 H200-MIG), resumable (skips existing npz) | 1,191/1,191 |
| `ha12_fit_diag.py`, `ha12_scale_test.py`, `ha12_scale_test2.py` | R3 | Diagnostics that isolated **scale** (not iters/warm-start) as the tail cause: batched@150≡@300 (converged), warm-start 7.65→7.38 (inert), scale 7.65→3.94 (halved); single-vs-multi-bone scale | — |
| `ha12_params_to_bvh.py` | R3 back-half A | SMPL params `[F,72]`→`[F,24,3]` (no palm-pad, no basis change — fit was Y-up), resample 20→30 (slerp/lerp), → **smpl2bvh** → SMPL-24 BVH@30fps. **Sets `MKL_THREADING_LAYER=GNU`** or the smpl2bvh subprocess dies ("incompatible with libgomp") | 1,191/1,191, 24 joints |
| `ha12_stageA.sbatch` / `ha12_stageB.sbatch` | R3 back-half | A: array running `ha12_params_to_bvh.py` (resumable). B (chained `afterok`): `retarget_amass_to_cmu_batch.py` on the SMPL-24 tree → CMU-33 | **1,191/1,191 CMU BVH; 28 joints + 30 fps; joint names == `amass_cmu_flat`; Y-up upright (orientation gate)** |

The R3 back half reuses R1's `retarget_amass_to_cmu_batch.py` + `bvhReader/` verbatim, so HA12 lands
format-identical to the AMASS corpus. Output: `datasets/_humanact12_cmu/` (final CMU-33 BVH @30fps).

## Shared retarget core — `bvhReader/`

Third-party BVH library adapted from **[alinen/bvh-python](https://github.com/alinen/bvh-python)** and extended
for our retargeting/VAE workflow; **GPL-v3** (full text in `bvhReader/LICENSE`, kept for redistribution + attribution).
Used by **every** route (and by the pending R3 back half). External deps: **PyGLM** (`glm`), **torch**, **numpy**,
**matplotlib** (`pip install PyGLM numpy matplotlib torch`).

| File | Purpose |
|---|---|
| `bvh.py` | `BVH`/`Joint` classes: load/save BVH, per-joint local/global rotation + position, Euler↔quat↔matrix↔6-D |
| `retarget.py` | Skeleton→skeleton retargeting (`retarget_bvh`, `align_limbs`, `apply_scaled_root_motion`) — the core all routes call |
| `rotation_conversions.py` | Pure-tensor (torch) rotation utilities |
| `bvhConverterToPerform.py` | The CMU converter used by **R2** (`bvhconv_lib.py` = `head -377` of this, dropping its broken executable tail on import) |
| `bvhvisualize.py` | Matplotlib BVH viewer (imported by `retarget.py`) |

**⚠ `rest_delta` gotcha (reproducibility-critical).** `align_limbs` computes each joint's `rest_delta` — the
structural T-pose offset between source and target skeletons. An earlier version read it from *frame 0 of the
animation* instead of the rest pose, baking the clip's frame-0 posture into every output (the "arms wide open"
artifact on KIT clips). The current code uses `find_rotation_tpose` over the HIERARCHY offsets (purely
structural). If retargets are regenerated, re-QC per subset — see `amass/qc_nonkit_samples.py` on chimera.

## Notes

- **`ha12_gate.py`:** as checked in, `clip` points at the *raw action-to-motion* source (`_humanact12_raw/…`) —
  the **failing diagnostic** (mean diff 0.30; all four flip variants identical → not an orientation flip, a
  different source). The **passing** run swaps `clip` to the HumanML3D `pose_data/humanact12.zip` source
  (`_humanact12_h3d/…/*.npy`) → non-feet max 0.037, residual 2/35 foot-contact frames. Both kept in one file so
  the wrong-source trap is reproducible.
- **R3 fit env:** `j2s_gpu` with a **CUDA-12 torch** (`torch 2.4.1+cu121`) for H200s, plus `smplx`, `h5py`,
  `chumpy`, the SMPL-neutral model, and joints2smpl assets (`neutral_smpl_mean_params.h5`, `gmm_08.pkl`); the
  `smpl2bvh` tool (Fukazawa) lives at `fit3d/third_party/smpl2bvh` on chimera.
- **Correct HumanAct12 fetch:**
  `curl -L https://raw.githubusercontent.com/EricGuo5513/HumanML3D/main/pose_data/humanact12.zip`.

## CMU correction (R2) + shared finish

**R2 CMU is AMASS-CMU, not cgspeed** (corrected 2026-10-04 for TMR-consistency — TMR's CMU is the AMASS subset;
cgspeed is a different conversion/extent, `80_63` 18.9 s vs 37.9 s, 74% co-reg). CMU runs the **R1 pipeline**:
`run_cmu_stageA.sbatch` (AMASS-CMU SMPL+H G → SMPL-24 BVH) → `run_cmu_stageB.sbatch` (retarget → CMU-33);
`_verify_cmu_coreg.py` gates on annotation-`duration` ≈ clip-duration (→ 100%). The cgspeed scripts
(`mediafire_dl_mb`, `cmu_roundtrip`, `cmu_convert_248`) are kept for provenance only.

**Shared finish** (all CPU/GPU-light, reuse validated code):
| Script | Stage | Output |
|---|---|---|
| `build_hml3d_stage1.py` | flatten 3 routes + intersect to HumanML3D + crop to annotation windows + caption-align | `hml3d_cmu_flat/` (14,613 clips: 8,393 whole symlinks + 6,220 crops) + `hml3d_captioned.json` (40,384 caps) |
| `distilbert_token_precompute_hml3d.py` / `text_precompute_hml3d.py` | DistilBERT-token + MPNet banks (tmr env, GPU) | `hml3d_text_{distilbert_tokens,mpnet}.pt` (full coverage) |
| `compute_norm_hml3d.py` | per-channel norm (canonical builder, EPS=0.05) | `norm_stats_hml3d.npz` (204 ch; 92 degenerate — mostly retarget structural stubs) |
| `_heldout_check.py` | eval-clip overlap gate | PASS (0 overlap with 486 LMA eval clips) |

Corpus = full HumanML3D (TMR benchmark), **nonmirror-only**; cut1 artifacts untouched.
