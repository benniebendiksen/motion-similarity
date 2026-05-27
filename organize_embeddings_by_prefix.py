#!/usr/bin/env python3
"""
organize_embeddings_by_prefix.py

Moves embedding files from a single source directory into three new destination
directories based on filename prefix with an underscore:
  - "Walking_"   -> lma_perform_walking_encoded_2
  - "Pointing_"  -> lma_perform_pointing_encoded_2
  - "Picking_"   -> lma_perform_picking_encoded_2

This script is careful to only match the exact prefixes WITH the trailing underscore,
so names like "PointingMirror_..." are ignored.

Paths are set for your environment on macOS.
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict

# ---- Configure your paths here ----
SRC_DIR = Path("/Users/bendiksen/Desktop/research/vr_lab/triplets/datasets/lma_perform_encoded")

DATASETS_ROOT = Path("/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/datasets")
DEST_DIRS: Dict[str, Path] = {
    "Walking_":  DATASETS_ROOT / "lma_perform_walking_encoded_2",
    "Pointing_": DATASETS_ROOT / "lma_perform_pointing_encoded_2",
    "Picking_":  DATASETS_ROOT / "lma_perform_picking_encoded_2",
}

# Only move files that start with one of those exact prefixes:
PREFIXES = tuple(DEST_DIRS.keys())

# Optional: restrict to certain extensions (uncomment if desired)
# ALLOWED_EXTS = {".pt", ".pth", ".npy"}  # add/remove as you wish
ALLOWED_EXTS = None  # None = allow all file extensions

# Safety first: preview actions without moving anything.
DRY_RUN = False  # set to False to actually move files

# Recurse into subfolders under source? (True recommended for safety)
RECURSE = True


def ensure_dirs():
    for p in DEST_DIRS.values():
        p.mkdir(parents=True, exist_ok=True)


def matches_prefix(name: str) -> str | None:
    """
    Return the matching prefix (with underscore) if the filename starts exactly
    with one of the desired prefixes; else None.
    """
    for pref in PREFIXES:
        if name.startswith(pref):
            return pref
    return None


def allowed_ext(name: str) -> bool:
    if ALLOWED_EXTS is None:
        return True
    return Path(name).suffix in ALLOWED_EXTS


def main():
    if not SRC_DIR.exists():
        print(f"ERROR: Source directory not found: {SRC_DIR}")
        return

    ensure_dirs()

    count_total = 0
    count_matched = 0
    count_moved = 0

    # Build a quick regex that enforces prefix + underscore strictly
    # (This is redundant with startswith, but handy if you want to extend patterns)
    pattern = re.compile(r"^(Walking|Pointing|Picking)_")

    def handle_file(fpath: Path):
        nonlocal count_total, count_matched, count_moved
        if not fpath.is_file():
            return

        name = fpath.name
        count_total += 1

        # Optional extension filter
        if not allowed_ext(name):
            return

        # Ensure exact prefix with underscore
        if not pattern.match(name):
            return

        # Double-check with startswith for a precise key into DEST_DIRS
        pref = matches_prefix(name)
        if pref is None:
            return

        count_matched += 1
        dest_dir = DEST_DIRS[pref]
        dest_path = dest_dir / name

        if dest_path.exists():
            print(f"[SKIP] Already exists at destination: {dest_path}")
            return

        rel = fpath.relative_to(SRC_DIR) if fpath.is_relative_to(SRC_DIR) else fpath
        print(f"{'[DRY-RUN] ' if DRY_RUN else ''}MOVE: {rel} -> {dest_path}")

        if not DRY_RUN:
            shutil.move(str(fpath), str(dest_path))
            count_moved += 1

    if RECURSE:
        for root, _dirs, files in os.walk(SRC_DIR):
            root_path = Path(root)
            for nm in files:
                handle_file(root_path / nm)
    else:
        for nm in os.listdir(SRC_DIR):
            handle_file(SRC_DIR / nm)

    print("\nSummary")
    print("-------")
    print(f"  Scanned files : {count_total}")
    print(f"  Matched files : {count_matched}")
    if DRY_RUN:
        print(f"  (Dry-run) Moves planned: {count_matched}")
        print("  To execute moves, set DRY_RUN = False in the script and re-run.")
    else:
        print(f"  Files moved   : {count_moved}")


if __name__ == "__main__":
    main()

