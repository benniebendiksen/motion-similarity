#!/usr/bin/env python3
"""
retarget_amass_to_cmu_batch.py
================================
Batch-retargets AMASS-derived SMPL-24 BVH files (produced by
amass_smplh_to_bvh_batch.py) to the CMU-33 skeleton used by the training
pipeline.

Source:  ACCAD/bvh_smpl_30fps/**/*.bvh     (30 fps, ZYX rotation order)
         — nested tree mirrors the AMASS sub-dataset layout
           (category_c3d/ folders + s00x/ folders)
Target:  datasets/cmuTPose.bvh              (CMU 33-joint T-pose)
Output:  datasets/amass_accad_cmu/**/*_cmu33_30fps.bvh  (mirrors source tree)

Usage (run from project root):
    # Full ACCAD
    python amass/retarget_amass_to_cmu_batch.py \\
        --src_dir ACCAD/bvh_smpl_30fps \\
        --dst_dir datasets/amass_accad_cmu

    # Single subfolder for validation
    python amass/retarget_amass_to_cmu_batch.py \\
        --src_dir ACCAD/bvh_smpl_30fps/s001 \\
        --dst_dir datasets/amass_accad_cmu/s001 \\
        --overwrite

Joint mapping and curved-neck fix
---------------------------------
Uses the identical SMPL -> CMU mapping as fit3d/retarget_fit3d_to_cmu_batch.py,
because Stage A emits the same SMPL-24 BVH layout for both pipelines.  The
head/neck structural offset is absorbed by align_limbs' rest_delta correction
in bvhReader/retarget.py (the same code path that fixed Fit3D and H36M).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure bvhReader package is importable regardless of cwd.
_ROOT    = Path(__file__).parent.resolve()   # .../amass/
_PROJECT = _ROOT.parent                       # .../triplets/
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from bvhReader.bvh import BVH
from bvhReader.retarget import retarget_bvh

# ---------------------------------------------------------------------------
# Joint mapping:  source (SMPL-24 name)  ->  target (CMU-33 name)
# Identical to fit3d/retarget_fit3d_to_cmu_batch.py — keep in sync if changed.
# ---------------------------------------------------------------------------
SMPL_TO_CMU: dict[str, str] = {
    "Pelvis":         "Hips",

    "Left_hip":       "LeftUpLeg",
    "Left_knee":      "LeftLeg",
    "Left_ankle":     "LeftFoot",
    "Left_foot":      "LeftToeBase",

    "Right_hip":      "RightUpLeg",
    "Right_knee":     "RightLeg",
    "Right_ankle":    "RightFoot",
    "Right_foot":     "RightToeBase",

    "Spine1":         "LowerBack",
    "Spine2":         "Spine",
    "Spine3":         "Spine1",
    "Neck":           "Neck1",
    "Head":           "Head",

    "Left_shoulder":  "LeftArm",
    "Left_elbow":     "LeftForeArm",
    "Left_wrist":     "LeftHand",

    "Right_shoulder": "RightArm",
    "Right_elbow":    "RightForeArm",
    "Right_wrist":    "RightHand",
}

CMU_TPOSE = _PROJECT / "datasets" / "cmuTPose.bvh"


def process_tree(src_dir: Path, dst_dir: Path, overwrite: bool) -> tuple[int, int, int]:
    bvh_files = sorted(src_dir.rglob("*.bvh"))
    if not bvh_files:
        print(f"[WARN] no .bvh files found under {src_dir}")
        return 0, 0, 0

    ok = 0
    skipped = 0
    failed = 0

    for src_path in bvh_files:
        rel = src_path.relative_to(src_dir)
        stem = src_path.stem.replace("_smpl24_bodyonly_30fps", "")
        out_path = dst_dir / rel.parent / f"{stem}_cmu33_30fps.bvh"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            print(f"  [skip] {rel.parent}/{out_path.name}")
            skipped += 1
            continue

        try:
            source = BVH()
            source.load(str(src_path))

            # Fresh T-pose per clip so retarget_bvh can modify it freely.
            target = BVH()
            target.load(str(CMU_TPOSE))

            retarget_bvh(source, target, SMPL_TO_CMU, str(out_path))
            print(f"  [ok]   {rel.parent}/{out_path.name}  ({source.numFrames()} frames)")
            ok += 1

        except Exception as exc:
            print(f"  [ERR]  {rel}: {exc}")
            failed += 1

    return ok, skipped, failed


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Retarget AMASS-derived SMPL-24 BVH files to the CMU-33 skeleton."
    )
    ap.add_argument(
        "--src_dir", type=str, required=True,
        help="Source directory containing SMPL-24 BVH files (walked recursively)."
    )
    ap.add_argument(
        "--dst_dir", type=str, required=True,
        help="Destination directory for CMU-33 BVH files (mirrors src tree)."
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Re-process files even if the output already exists."
    )
    args = ap.parse_args()

    if not CMU_TPOSE.exists():
        sys.exit(f"[ERROR] CMU T-pose skeleton not found: {CMU_TPOSE}")

    src_dir = Path(args.src_dir).resolve()
    dst_dir = Path(args.dst_dir).resolve()

    if not src_dir.exists():
        sys.exit(f"[ERROR] src_dir does not exist: {src_dir}")

    print(f"[amass-retarget] src: {src_dir}")
    print(f"[amass-retarget] dst: {dst_dir}")

    ok, skipped, failed = process_tree(src_dir, dst_dir, args.overwrite)

    total = ok + skipped + failed
    print(f"\n[done] ok={ok}  skipped={skipped}  failed={failed}  total={total}")


if __name__ == "__main__":
    main()
