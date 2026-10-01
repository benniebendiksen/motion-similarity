# 06 — Unified corpus build (CMU-adapted HumanML3D)

Conversion + fidelity-gate scripts for retargeting every HumanML3D clip to our
34-joint 6-D CMU / 30 fps format. Full recipe, provenance ledger, dataset
origins, and lessons: [`docs/context/TMLR/06_CORPUS_BUILD.md`](../../docs/context/TMLR/06_CORPUS_BUILD.md).

**Paths are chimera-specific** (`/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets`);
edit the `TR` constant at the top of each script for another host. Python 3, numpy;
`ha12_gate.py` needs the TMR repo on `sys.path` (`src.guofeats`).

| Script | Role | Gate result |
|---|---|---|
| `align_corpus_v2.py` | Map `annotations.json` paths → retargeted AMASS filenames (caption coverage) | 10,509/10,510 (99.99%) |
| `mediafire_dl_mb.py` | Server-side fetch of the 9 `cmuconvert-mb2-*.zip` (Motionbuilder-friendly CMU BVH) — decodes `data-scrambled-url`, verifies ZIP magic | 9/9 OK, all 248 ids present |
| `cmu_roundtrip.py` | CMU fidelity gate: convert mb `80_63`, require it to reproduce `cmu_all_perform/80_63.bvh` | MAX abs diff = 0 |
| `cmu_convert_248.py` | Convert the 248 CMU source ids `cmu_all_perform` lacked, through the round-trip-proven pipeline | 248/248, 28 joints, 30 fps |
| `ha12_gate.py` | HumanAct12 source-correctness gate: `joints_to_guofeats(clip)` vs HumanML3D `new_joint_vecs/000001` | see note |
| `ha12_batch_fit.py` | HumanAct12 positions → SMPL params via SMPLify3D: `[:22]` + **scale→SMPL (pelvis→neck=0.51 m)** + ground → batched fit (B=128 chunks); saves `poses,trans,mpjpe,scale,fps=20` per clip | **n=1,191: MPJPE mean 3.09 / median 3.02 / max 6.69 cm; >5 cm = 1%** |
| `ha12_fit_array.sbatch` / `ha12_fit_mig.sbatch` | SLURM arrays running `ha12_batch_fit.py` (pomplun H200 / chimera24 H200-MIG), resumable (skips existing npz) | 1,191/1,191 |
| `ha12_fit_diag.py`, `ha12_scale_test.py`, `ha12_scale_test2.py` | Diagnostics that isolated **scale** (not iters/warm-start) as the tail cause: batched@150≡@300 (converged), warm-start 7.65→7.38 (inert), scale 7.65→3.94 (halved); and single-bone vs multi-bone scale comparison | — |

Needs the j2s_gpu env with a **CUDA-12 torch** (`torch 2.4.1+cu121`) for H200s, plus `smplx`, `h5py`,
`chumpy` and the SMPL-neutral model + joints2smpl assets (`neutral_smpl_mean_params.h5`, `gmm_08.pkl`).
The SMPL→BVH tool `smpl2bvh` (Fukazawa) lives at `fit3d/third_party/smpl2bvh` on chimera.

**`ha12_gate.py` note.** As checked in, the `clip` path points at the *raw action-to-motion*
source (`_humanact12_raw/…`) — this is the **failing diagnostic** run (mean diff 0.30, all four
flip variants identical → not an orientation flip, a different source). The **passing** run swaps
`clip` to the HumanML3D `pose_data/humanact12.zip` source
(`_humanact12_h3d/extracted/humanact12/humanact12/*.npy`) → non-feet max 0.037, residual 2/35
foot-contact frames. Both are kept in one file so the wrong-source trap is reproducible.

**Not included here:** `bvhconv_lib.py` = `head -377 bvhConverterToPerform.py` (the CMU converter with
its broken executable tail stripped); regenerate on chimera from the converter in `triplets/`.
The correct HumanAct12 fetch is a one-liner:
`curl -L https://raw.githubusercontent.com/EricGuo5513/HumanML3D/main/pose_data/humanact12.zip`.
