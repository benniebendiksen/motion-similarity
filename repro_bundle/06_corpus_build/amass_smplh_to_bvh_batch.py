#!/usr/bin/env python3
"""
amass_smplh_to_bvh_batch.py
===========================

Batch-convert AMASS SMPL-H .npz files (body-only) to SMPL-24 BVH via smpl2bvh,
resampling from the clip's native FPS (typically 120) to your training FPS (30).

Pipeline per clip:
  1) Load AMASS .npz (keys: trans, poses, gender, mocap_framerate, betas, dmpls)
  2) Slice poses[:, :66] -> body-only (root + 21 body joints, 22*3 axis-angle).
     Drop the 90 hand dims (15 L-hand + 15 R-hand joints).
  3) Pack to smpl2bvh's SMPL-24 layout: [root] + [21 body] + [2 palm identity+epsilon].
  4) Resample trans + rotations from clip_fps -> target_fps (lerp + slerp).
     No basis change: AMASS is already in SMPL's native Y-up frame.
  5) Save .npz with keys: poses [1,F,24,3] and trans [1,F,3]  (smpl2bvh layout)
  6) Call smpl2bvh.py to write a BVH at target_fps, using the per-clip gender.

Differences vs fit3d_smplx_to_bvh_batch.py:
  * Input is AMASS .npz (axis-angle 52-joint SMPL-H), not Fit3D .json (rotmat SMPL-X).
  * Basis change: AMASS ships in a Z-up world frame (standard for mocap), while the
    SMPL canonical rest pose and BVH are Y-up. We apply R_x(-90°) to root orientation
    and translation. The matrix is different from Fit3D's because AMASS's world-axis
    labelling differs from Fit3D's.
  * Gender is read per-clip from npz['gender'], not passed at the command line.
  * Native FPS is read per-clip from npz['mocap_framerate'] (typically 120 for ACCAD;
    varies across AMASS sub-datasets).
  * Walks the input directory recursively and mirrors its structure in the output,
    so ACCAD's nested layout (category_c3d/*.npz and s0??/*.npz) is preserved.

Neck/head curvature fix:
  * This Stage A script produces clean SMPL-24 BVH files. The curved-neck/head
    correction is applied in Stage B by bvhReader/retarget.py's align_limbs via its
    rest_delta rotation offset (same code path used for Fit3D and H36M). No special
    handling is needed here.

Stage B (retargeting to CMU-33):
  Run fit3d/retarget_fit3d_to_cmu_batch.py against the output BVH tree, or clone it
  as amass/retarget_amass_to_cmu_batch.py with path arguments updated. The joint
  mapping (SMPL arm chain -> CMU arm chain, palms dropped) is identical.

smpl2bvh dependency notes:
  * Uses the same fit3d/third_party/smpl2bvh clone you already have set up for Fit3D.
  * Model files live at fit3d/third_party/smpl2bvh/data/smpl/smpl/
      SMPL_MALE.pkl, SMPL_FEMALE.pkl, SMPL_NEUTRAL.pkl
  * We use SMPL (not SMPL-H) model pkls: the first 22 joints of SMPL-H are structurally
    identical to SMPL's body joints, and we pack the rest as identity palms exactly as
    Fit3D does, so SMPL_{GENDER}.pkl handles the BVH emission correctly.

Example (ACCAD):
  python amass/amass_smplh_to_bvh_batch.py \\
    --amass_dir     ACCAD \\
    --out_npz_dir   ACCAD/smpl_npz_30fps \\
    --out_bvh_dir   ACCAD/bvh_smpl_30fps \\
    --smpl2bvh_repo fit3d/third_party/smpl2bvh \\
    --target_fps 30

Output layout (mirrors source):
  ACCAD/bvh_smpl_30fps/Female1General_c3d/A1 - Stand_smpl24_bodyonly_30fps.bvh
  ACCAD/bvh_smpl_30fps/s001/EricCamper04_smpl24_bodyonly_30fps.bvh
  ...
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


# --------- Axis conversion: AMASS Z-up -> BVH/SMPL Y-up --------------------
# AMASS world frame is Z-up (standard for mocap). The SMPL model's canonical
# rest pose and BVH convention are Y-up. Rotate -90° around X to align the
# vertical axis, then 180° around Y so the character faces the same direction
# as Fit3D-derived BVH (which is +Z in the Y-up frame).
# Applied only to root orientation (global_orient) and world translation;
# local body joint rotations are coordinate-frame-independent.
_M_ZUP_TO_YUP = (
    R.from_euler("y", 180, degrees=True)
    * R.from_euler("x", -90, degrees=True)
).as_matrix().astype(np.float32)


# ---------------------------- Resampling ------------------------------------

def _uniform_times(num_frames: int, fps: float) -> np.ndarray:
    if num_frames < 2:
        return np.array([0.0], dtype=np.float64)
    return np.arange(num_frames, dtype=np.float64) / float(fps)


def resample_trans(trans: np.ndarray, src_fps: float, target_fps: float) -> np.ndarray:
    """Linear resample translation. trans: [F,3]"""
    F = trans.shape[0]
    t_old = _uniform_times(F, src_fps)
    duration = t_old[-1]
    F_new = int(np.floor(duration * target_fps)) + 1
    t_new = np.arange(F_new, dtype=np.float64) / float(target_fps)

    out = np.zeros((F_new, 3), dtype=np.float32)
    for k in range(3):
        out[:, k] = np.interp(t_new, t_old, trans[:, k]).astype(np.float32)
    return out


def resample_rotvec(rotvec: np.ndarray, src_fps: float, target_fps: float) -> np.ndarray:
    """Slerp-resample axis-angle rotations. rotvec: [F,J,3]"""
    F, J, _ = rotvec.shape
    t_old = _uniform_times(F, src_fps)
    duration = t_old[-1]
    F_new = int(np.floor(duration * target_fps)) + 1
    t_new = np.arange(F_new, dtype=np.float64) / float(target_fps)

    out = np.zeros((F_new, J, 3), dtype=np.float32)
    for j in range(J):
        r_old = R.from_rotvec(rotvec[:, j, :])
        slerp = Slerp(t_old, r_old)
        r_new = slerp(t_new)
        out[:, j, :] = r_new.as_rotvec().astype(np.float32)
    return out


# -------------------------- AMASS .npz parsing ------------------------------

def load_amass_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, str, float]:
    """
    Returns:
      trans:  [F,3]                   root translation (meters)
      poses:  [F,22,3]                body-only axis-angle (root + 21 body joints)
      gender: 'MALE' | 'FEMALE' | 'NEUTRAL'
      fps:    float                   clip's native capture framerate
    """
    d = np.load(npz_path, allow_pickle=True)

    # Older AMASS files use 'mocap_framerate'; newer AMASS-X / stageii files
    # (CNRS, GRAB, SOMA) use 'mocap_frame_rate'. Accept either.
    if "mocap_framerate" in d.files:
        fps_key = "mocap_framerate"
    elif "mocap_frame_rate" in d.files:
        fps_key = "mocap_frame_rate"
    else:
        raise ValueError(f"{npz_path}: missing fps key (tried 'mocap_framerate' and 'mocap_frame_rate')")

    for k in ("trans", "poses", "gender"):
        if k not in d.files:
            raise ValueError(f"{npz_path}: missing required key '{k}'")

    trans = np.asarray(d["trans"], dtype=np.float32)
    # CNRS is SMPL-X (poses has 165 dims = 55 joints); all others are SMPL-H
    # (156 dims = 52 joints). In both cases the first 66 dims are the same
    # 22-joint body (root + 21), so body-only slicing works uniformly.
    poses_full = np.asarray(d["poses"], dtype=np.float32)

    if poses_full.ndim != 2 or poses_full.shape[1] < 66:
        raise ValueError(
            f"{npz_path}: unexpected poses shape {poses_full.shape}; "
            f"expected [F, >=66]"
        )

    # Slice to body-only: 1 root + 21 body = 22 joints = 66 dims
    poses_body = poses_full[:, :66].reshape(-1, 22, 3)

    # Basis change (Z-up -> Y-up): premultiply the root rotation by _M_ZUP_TO_YUP.
    # Local body joint rotations are defined relative to their parents in the
    # canonical SMPL frame and need no transformation. World translation is
    # handled separately below.
    root_mat_zup = R.from_rotvec(poses_body[:, 0, :]).as_matrix()          # [F,3,3]
    root_mat_yup = _M_ZUP_TO_YUP[None, :, :] @ root_mat_zup                # [F,3,3]
    poses_body[:, 0, :] = R.from_matrix(root_mat_yup).as_rotvec().astype(np.float32)

    # Rotate world translation into the Y-up frame.
    trans = (_M_ZUP_TO_YUP @ trans.T).T.astype(np.float32)

    # AMASS ships exact-zero axis-angle rows for motionless joints. smpl2bvh's
    # quat.py divides by rotation angle, which produces NaN for a [0,0,0] row.
    # Fit3D avoids this incidentally because R.from_matrix(I).as_rotvec() returns
    # floating-point noise instead of exact zeros. Nudge any exact-zero row off
    # the pole by the same 1e-8 epsilon we use for the palm joints.
    zero_mask = np.linalg.norm(poses_body, axis=-1) < 1e-12
    poses_body[zero_mask, 0] += 1e-8

    gender_raw = d["gender"]
    if isinstance(gender_raw, np.ndarray):
        gender_raw = gender_raw.item()
    if isinstance(gender_raw, bytes):
        gender_raw = gender_raw.decode("utf-8")
    gender = str(gender_raw).strip().upper()
    if gender not in ("MALE", "FEMALE", "NEUTRAL"):
        raise ValueError(f"{npz_path}: unexpected gender '{gender_raw}'")

    fps = float(d[fps_key])

    return trans, poses_body, gender, fps


def pack_smpl24_bodyonly_rotvec(body_rotvec: np.ndarray) -> np.ndarray:
    """
    body_rotvec: [F,22,3]  (root + 21 body joints)
    returns:     [F,24,3]  (smpl2bvh SMPL-24 layout; joints 22,23 are identity palms)
    """
    F = body_rotvec.shape[0]
    rotations = np.zeros((F, 24, 3), dtype=np.float32)
    rotations[:, :22, :] = body_rotvec
    # 22,23 (palms): tiny epsilon avoids 0/0 -> NaN in axis-angle -> quaternion
    rotations[:, 22:24, 0] = 1e-8
    return rotations


def build_smpl2bvh_npz_from_amass(
    npz_path: Path,
    out_npz_path: Path,
    target_fps: float,
    center_trans: bool,
    scale_trans: float,
) -> Tuple[int, int, str, float]:
    """
    Build (and resample) .npz compatible with smpl2bvh.
    Returns: (F_old, F_new, gender, src_fps)
    """
    trans, poses_body, gender, src_fps = load_amass_npz(npz_path)

    if scale_trans != 1.0:
        trans = trans * float(scale_trans)

    rotations = pack_smpl24_bodyonly_rotvec(poses_body)  # [F,24,3]

    if center_trans:
        trans = trans - trans[0:1]

    F_old = trans.shape[0]

    if abs(target_fps - src_fps) > 1e-6:
        trans = resample_trans(trans, src_fps=src_fps, target_fps=target_fps)
        rotations = resample_rotvec(rotations, src_fps=src_fps, target_fps=target_fps)

    F_new = trans.shape[0]

    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz_path, poses=rotations[np.newaxis], trans=trans[np.newaxis])

    return F_old, F_new, gender, src_fps


# ------------------------ smpl2bvh invocation --------------------------------

def run_smpl2bvh(
    smpl2bvh_repo: Path,
    poses_npz: Path,
    out_bvh: Path,
    gender: str,
    fps: float,
    mirror: bool,
    python_exe: str,
) -> None:
    smpl2bvh_py = Path(smpl2bvh_repo) / "smpl2bvh.py"
    if not smpl2bvh_py.exists():
        raise FileNotFoundError(f"smpl2bvh.py not found at: {smpl2bvh_py}")

    out_bvh.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exe,
        smpl2bvh_py.name,
        "--gender", gender,
        "--model_path", "data/smpl",
        "--poses", str(poses_npz.resolve()),
        "--fps", str(int(round(fps))),
        "--output", str(out_bvh.resolve()),
    ]
    if mirror:
        cmd.append("--mirror")

    subprocess.run(cmd, cwd=str(smpl2bvh_repo), check=True)


# --------------------------------- Main -------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amass_dir", type=str, required=True,
                    help="Root directory of an AMASS sub-dataset (e.g. ACCAD). "
                         "Walked recursively for *.npz files.")
    ap.add_argument("--out_npz_dir", type=str, required=True,
                    help="Where to write intermediate smpl2bvh npz files (mirrors input tree)")
    ap.add_argument("--out_bvh_dir", type=str, required=True,
                    help="Where to write output SMPL-24 BVH files (mirrors input tree)")
    ap.add_argument("--smpl2bvh_repo", type=str, required=True,
                    help="Path to cloned KosukeFukazawa/smpl2bvh repo "
                         "(e.g. fit3d/third_party/smpl2bvh)")
    ap.add_argument("--target_fps", type=float, default=30.0)
    ap.add_argument("--center_trans", action="store_true",
                    help="Subtract first-frame translation so trans[0]=0 (recommended)")
    ap.add_argument("--no_center_trans", dest="center_trans", action="store_false")
    ap.set_defaults(center_trans=True)
    ap.add_argument("--scale_trans", type=float, default=1.0,
                    help="Multiply translations by this factor (default 1.0; "
                         "smpl2bvh handles m->cm internally)")
    ap.add_argument("--mirror", action="store_true",
                    help="Also save mirrored motion (smpl2bvh --mirror)")
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--dry_run", action="store_true",
                    help="Only build npz, do not call smpl2bvh")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing outputs")

    args = ap.parse_args()

    amass_dir = Path(args.amass_dir)
    out_npz_dir = Path(args.out_npz_dir)
    out_bvh_dir = Path(args.out_bvh_dir)
    smpl2bvh_repo = Path(args.smpl2bvh_repo)

    npz_paths = sorted(amass_dir.rglob("*.npz"))
    # Skip any outputs from a previous run, in case out_npz_dir is nested under amass_dir
    npz_paths = [p for p in npz_paths if out_npz_dir not in p.parents]

    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found under {amass_dir}")

    print(f"[amass] Found {len(npz_paths)} .npz files under {amass_dir}")
    print(f"[amass] target_fps={args.target_fps} | center_trans={args.center_trans} | scale_trans={args.scale_trans}")
    print(f"[amass] smpl2bvh_repo={smpl2bvh_repo}")

    ok = 0
    skipped = 0
    failed = 0

    for np_path in npz_paths:
        rel = np_path.relative_to(amass_dir)  # e.g. Female1General_c3d/A1 - Stand_poses.npz
        # Drop the "_poses" suffix if present (AMASS convention) to keep filenames tidy
        stem = np_path.stem
        if stem.endswith("_poses"):
            stem = stem[:-len("_poses")]

        out_name = f"{stem}_smpl24_bodyonly_{int(args.target_fps)}fps"
        out_npz = out_npz_dir / rel.parent / f"{out_name}.npz"
        out_bvh = out_bvh_dir / rel.parent / f"{out_name}.bvh"

        if not args.overwrite and out_bvh.exists() and (args.dry_run or out_npz.exists()):
            print(f"[skip] {rel} (already exists)")
            skipped += 1
            continue

        try:
            F_old, F_new, gender, src_fps = build_smpl2bvh_npz_from_amass(
                npz_path=np_path,
                out_npz_path=out_npz,
                target_fps=args.target_fps,
                center_trans=args.center_trans,
                scale_trans=args.scale_trans,
            )
            print(f"[npz] {rel} | {gender} | {src_fps:.0f}->{args.target_fps:.0f} fps | frames {F_old}->{F_new}")

            if not args.dry_run:
                run_smpl2bvh(
                    smpl2bvh_repo=smpl2bvh_repo,
                    poses_npz=out_npz,
                    out_bvh=out_bvh,
                    gender=gender,
                    fps=args.target_fps,
                    mirror=args.mirror,
                    python_exe=args.python,
                )
                print(f"[bvh] {rel} -> {out_bvh}")

            ok += 1

        except subprocess.CalledProcessError as e:
            print(f"[ERR] smpl2bvh failed for {np_path}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERR] failed for {np_path}: {e}")
            failed += 1

    print(f"[done] ok={ok}  skipped={skipped}  failed={failed}  total={len(npz_paths)}")


if __name__ == "__main__":
    main()
