#!/usr/bin/env python3
"""
Single-Embedding Dataset (one .pt per motion)

- Matches files that begin with one of:
    "Walking_", "Pointing_", "Picking_"   (underscore included)
- Parses the 4 effort values from the filename tail
    e.g., Walking_-1_1_0_-1.pt  -> (-1, 1, 0, -1)
- Loads a single embedding tensor per file (torch.save -> torch.load)
- Produces a similarity dict: { effort_tuple: [embedding_np, ...] }
- Optional filtering to "valid" efforts: drives (1 zero), states (2 zeros), neutral (4 zeros)
"""

from __future__ import annotations
import os
import re
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import torch


_PREFIXES = ("Walking_", "Pointing_", "Picking_")
# Regex: strict prefix, four ints in {-1,0,1}, separated by underscores, then extension
# Captures: prefix, e1, e2, e3, e4
_FILENAME_RE = re.compile(
    r"^(Walking|Pointing|Picking)_"                # group 1: prefix tag
    r"(-1|0|1)_"                                   # group 2
    r"(-1|0|1)_"                                   # group 3
    r"(-1|0|1)_"                                   # group 4
    r"(-1|0|1)"                                    # group 5
    r"(?:\.[A-Za-z0-9]+)$"                         # extension
)


def _is_valid_effort_tuple(t: Tuple[int, int, int, int]) -> bool:
    """
    Keep Drives (1 zero), States (2 zeros), and Neutral (4 zeros),
    matching your previous filtering policy.
    """
    zeros = t.count(0)
    return zeros in (1, 2, 4)


class SingleEmbeddingDataset:
    """
    Dataset for directories containing single embeddings (.pt) named:
      Walking_-1_1_0_-1.pt, Pointing_0_1_1_-1.pt, Picking_1_0_-1_1.pt, ...
    """

    def __init__(
        self,
        embedding_dir: str | Path,
        allowed_exts: Optional[Iterable[str]] = (".pt", ".pth", ".npy"),
        filter_valid_efforts: bool = True,
        device: Optional[str] = "cpu",
    ):
        self.dir = Path(embedding_dir)
        self.allowed_exts = tuple(allowed_exts) if allowed_exts else None
        self.filter_valid_efforts = filter_valid_efforts
        self.device = device

        if not self.dir.exists():
            raise FileNotFoundError(f"Embedding directory not found: {self.dir}")

        self.files = self._scan()
        print(f"[SingleEmbeddingDataset] Found {len(self.files)} candidate files in {self.dir}")

        self.index: List[Tuple[str, Tuple[int, int, int, int], Path]] = []
        for f in self.files:
            parsed = self._parse_filename(f.name)
            if parsed is None:
                continue
            prefix, effort = parsed
            if self.filter_valid_efforts and not _is_valid_effort_tuple(effort):
                # keep the same policy you used previously
                continue
            self.index.append((prefix, effort, f))

        print(f"[SingleEmbeddingDataset] Kept {len(self.index)} files after filtering")

    def _scan(self) -> List[Path]:
        if self.allowed_exts:
            out: List[Path] = []
            for ext in self.allowed_exts:
                out.extend(self.dir.glob(f"*{ext}"))
            return out
        else:
            return list(self.dir.iterdir())

    @staticmethod
    def _parse_filename(name: str) -> Optional[Tuple[str, Tuple[int, int, int, int]]]:
        """
        Returns (prefix, effort_tuple) or None if name does not match the strict pattern.
        """
        m = _FILENAME_RE.match(name)
        if not m:
            return None
        prefix = m.group(1)  # Walking | Pointing | Picking
        e1 = int(m.group(2))
        e2 = int(m.group(3))
        e3 = int(m.group(4))
        e4 = int(m.group(5))
        return prefix, (e1, e2, e3, e4)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[str, Tuple[int, int, int, int], np.ndarray, Path]:
        """
        Returns (prefix, effort_tuple, embedding_np, file_path)
        """
        prefix, effort, path = self.index[idx]
        emb = self._load_embedding(path)
        return prefix, effort, emb, path

    def _load_embedding(self, path: Path) -> np.ndarray:
        """
        Loads a single embedding as numpy array.
        - .pt/.pth: torch.load(Tensor or dict containing Tensor)
        - .npy:     np.load
        """
        if path.suffix in (".pt", ".pth"):
            try:
                # Preferred in PyTorch 2.4+ (prevents arbitrary pickled code paths)
                obj = torch.load(str(path), map_location=self.device, weights_only=True)
            except TypeError:
                # Older PyTorch: no weights_only kwarg — fall back to normal load
                obj = torch.load(str(path), map_location=self.device)
            if isinstance(obj, torch.Tensor):
                t = obj
            elif isinstance(obj, dict):
                # try common keys, else find the first Tensor value
                if "embedding" in obj and isinstance(obj["embedding"], torch.Tensor):
                    t = obj["embedding"]
                else:
                    tensors = [v for v in obj.values() if isinstance(v, torch.Tensor)]
                    if not tensors:
                        raise ValueError(f"No tensor found inside: {path}")
                    t = tensors[0]
            else:
                raise ValueError(f"Unsupported torch object type in {path}: {type(obj)}")
            t = t.detach().to("cpu").squeeze()
            return t.numpy()
        elif path.suffix == ".npy":
            arr = np.load(str(path))
            return np.squeeze(arr)
        else:
            raise ValueError(f"Unsupported extension for {path.name}")

    # ---- Public APIs ----

    def to_similarity_dict(self) -> Dict[Tuple[int, int, int, int], List[np.ndarray]]:
        """
        Build similarity dict: { effort_tuple: [embedding_np, ...] }
        """
        sim: Dict[Tuple[int, int, int, int], List[np.ndarray]] = {}
        for _, effort, emb, _ in (self[i] for i in range(len(self))):
            sim.setdefault(effort, []).append(emb)
        print(f"[SingleEmbeddingDataset] Built similarity dict with {len(sim)} classes")
        return sim

    def to_partition(self, which: str = "train") -> Dict[str, Dict]:
        """
        Return a partition dict compatible with your previous loaders:
            { "train": {effort_tuple: [emb]}, "test": {} }
        """
        if which not in ("train", "test"):
            raise ValueError("which must be 'train' or 'test'")
        sim = self.to_similarity_dict()
        return {"train": sim, "test": {}}

    def save_similarity_dict(self, out_path: str | Path) -> Path:
        sim = self.to_similarity_dict()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(sim, f)
        print(f"[SingleEmbeddingDataset] Saved similarity dict to: {out_path}")
        return out_path


# ---------- Convenience CLI / helpers ----------

def build_and_save_for_anim(anim_dir: str | Path, out_pickle: str | Path) -> Path:
    """
    Scan a directory, build similarity dict, and save as pickle.
    """
    ds = SingleEmbeddingDataset(anim_dir)
    return ds.save_similarity_dict(out_pickle)


def example_bulk_build():
    """
    Example bulk build for your three new dirs.
    Adjust paths if desired.
    """
    root = Path("/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/datasets")

    mapping = {
        "walking":  root / "lma_perform_walking_encoded_2",
        "pointing": root / "lma_perform_pointing_encoded_2",
        "picking":  root / "lma_perform_picking_encoded_2",
    }

    out_dir = root / "similarity_pickles_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    for anim, indir in mapping.items():
        out_pkl = out_dir / f"{anim}_embedding_similarity_exemplars.pickle"
        print(f"\n[{anim}] Building similarity pickle from: {indir}")
        build_and_save_for_anim(indir, out_pkl)


if __name__ == "__main__":
    # Run a quick demo build (comment out if calling from elsewhere)
    example_bulk_build()

