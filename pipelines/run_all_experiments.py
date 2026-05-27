#!/usr/bin/env python3
"""
Experiment orchestrator for the motion-similarity pipeline comparison.

Runs every training pipeline (or a chosen subset), then runs the matching
inference script for each, captures the stdout, parses the per-animation
correlation metrics, and writes structured JSON results under
  experiments/<name>/results.json

Afterwards it delegates to compare_results.py for the summary table.

Usage
-----
# Train + infer all variants:
    python pipelines/run_all_experiments.py

# Skip training (re-evaluate existing checkpoints):
    python pipelines/run_all_experiments.py --skip-training

# Run only specific experiments (comma-separated):
    python pipelines/run_all_experiments.py --only raw_motion,emb_vec_rots_only

# List available experiments without running:
    python pipelines/run_all_experiments.py --list

Isolation
---------
Each experiment writes checkpoints to
    <project_root>/experiments/<name>/checkpoints/
via the MOTION_CHECKPOINT_DIR env var read by Config.
The MOTION_CHECKPOINT_FILE env var tells each inference script which
.pt file to load (set automatically by this script after training).

Adding experiments
------------------
Append a dict to EXPERIMENTS below. Required keys:
  name, description, train_script, train_flags, infer_script, network_class
Optional:
  extra_infer_flags  – extra CLI args forwarded to the infer script (list)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # pipelines/
_ROOT = _HERE.parent                             # project root
_DATASETS = _ROOT.parent / "datasets"            # ../datasets relative to project root
_EXPERIMENTS_DIR = _ROOT / "experiments"
_PYTHON = sys.executable                         # same interpreter that's running this script


# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------
# Each entry is one distinct training-and-evaluation run.
# train_script / infer_script are filenames relative to pipelines/.
# train_flags / extra_infer_flags are lists of CLI tokens forwarded verbatim.
# network_class controls how the "best checkpoint" is identified after training.
#
# To add a new variant, append a dict – no other changes needed.

EXPERIMENTS: List[Dict] = [
    # ── 1. Raw BVH → CNN, standard neutral ───────────────────────────────
    {
        "name": "raw_motion",
        "description": "Raw BVH → CNN  |  standard neutral  |  base triplet loss",
        "train_script": "train_raw_motion.py",
        "train_flags": ["--animation", "all"],
        "infer_script": "infer_raw_motion.py",
        "network_class": "SimilarityNetwork",
    },
    # ── 2. Raw BVH → CNN, clustering-based neutral ────────────────────────
    {
        "name": "raw_motion_clustered",
        "description": "Raw BVH → CNN  |  k-means neutral  |  base triplet loss",
        "train_script": "train_raw_motion_clustered.py",
        "train_flags": ["--animation", "all"],
        "infer_script": "infer_raw_motion.py",
        "network_class": "SimilarityNetwork",
    },
    # ── 3. Raw BVH → CNN, perception-aligned loss ─────────────────────────
    {
        "name": "raw_motion_percloss",
        "description": "Raw BVH → CNN  |  standard neutral  |  perception-aligned loss",
        "train_script": "train_raw_motion.py",
        "train_flags": ["--animation", "all", "--use-perception-loss"],
        "infer_script": "infer_raw_motion.py",
        "network_class": "SimilarityNetwork_BestCorr",  # picks best-correlation checkpoint
    },
    # ── 4. AE embedding vectors → MLP, rots_only combination ─────────────
    {
        "name": "emb_vec_rots_only",
        "description": "AE emb → MLP  |  rots_only  |  base triplet loss",
        "train_script": "train_emb_vec.py",
        "train_flags": [
            "--combination-method", "rots_only",
            "--embedding-dir",   str(_DATASETS / "lma_perform_walking_encoded"),
            "--embedding-dir-2", str(_DATASETS / "lma_perform_pointing_encoded"),
            "--embedding-dir-3", str(_DATASETS / "lma_perform_picking_encoded"),
        ],
        "infer_script": "infer_emb_vec.py",
        "network_class": "EmbeddingRefiningSimilarityNetwork",
    },
    # ── 5. AE embedding vectors → MLP, concat combination ────────────────
    {
        "name": "emb_vec_concat",
        "description": "AE emb → MLP  |  concat  |  base triplet loss",
        "train_script": "train_emb_vec.py",
        "train_flags": [
            "--combination-method", "concat",
            "--embedding-dir",   str(_DATASETS / "lma_perform_walking_encoded"),
            "--embedding-dir-2", str(_DATASETS / "lma_perform_pointing_encoded"),
            "--embedding-dir-3", str(_DATASETS / "lma_perform_picking_encoded"),
        ],
        "infer_script": "infer_emb_vec.py",
        "network_class": "EmbeddingRefiningSimilarityNetwork",
    },
    # ── 6. AE embedding vectors → MLP, k-means neutral, rots_only ─────────
    {
        "name": "emb_vec_clustered",
        "description": "AE emb → MLP  |  rots_only  |  k-means neutral",
        "train_script": "train_emb_vec_clustered.py",
        "train_flags": [
            "--combination-method", "rots_only",
            "--embedding-dir",   str(_DATASETS / "lma_perform_walking_encoded"),
            "--embedding-dir-2", str(_DATASETS / "lma_perform_pointing_encoded"),
            "--embedding-dir-3", str(_DATASETS / "lma_perform_picking_encoded"),
        ],
        "infer_script": "infer_emb_vec.py",
        "network_class": "EmbeddingRefiningSimilarityNetwork",
    },
    # ── 7. AE embedding vectors → MLP via full SimilarityNetwork trainer ──
    {
        "name": "emb_vec_percloss",
        "description": "AE emb → MLP (SimilarityNetwork trainer)  |  perception loss",
        "train_script": "train_emb_vec_percloss.py",
        "train_flags": [
            "--animation", "walking",
            "--use-perception-loss",
            "--embedding-dir", str(_DATASETS / "lma_perform_walking_encoded"),
        ],
        "infer_script": "infer_emb_vec_percloss.py",
        "network_class": "SimilarityNetwork_BestCorr",
    },
    # ── 8. Baseline: raw AE distances, no triplet fine-tuning ────────────
    {
        "name": "emb_baseline",
        "description": "AE embedding distances only  |  no trained model (floor baseline)",
        "train_script": None,     # no training step
        "train_flags": [],
        "infer_script": "infer_emb_baseline.py",
        "network_class": None,
    },
]


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_best_checkpoint(checkpoint_dir: Path, network_class: Optional[str]) -> Optional[str]:
    """
    Return the filename (not full path) of the checkpoint to evaluate.

    Strategy per network_class:
      SimilarityNetwork            → highest-epoch  0_similarity_model_weights_epoch_*.pt
      SimilarityNetwork_BestCorr   → highest-epoch  0_best_correlation_epoch_*.pt
      EmbeddingRefiningSimilarityNetwork → 0_final_embedding_model.pt
      None                         → None (baseline, no model needed)
    """
    if network_class is None:
        return None

    if network_class == "EmbeddingRefiningSimilarityNetwork":
        final = checkpoint_dir / "0_final_embedding_model.pt"
        if final.exists():
            return final.name
        # Fallback: highest-epoch embedding model
        candidates = sorted(checkpoint_dir.glob("0_embedding_model_epoch_*.pt"))
        return candidates[-1].name if candidates else None

    if network_class == "SimilarityNetwork_BestCorr":
        candidates = sorted(checkpoint_dir.glob("0_best_correlation_epoch_*.pt"))
        if candidates:
            return candidates[-1].name
        # Fall through to plain similarity weights if no best-corr checkpoint
        candidates = sorted(checkpoint_dir.glob("0_similarity_model_weights_epoch_*.pt"))
        return candidates[-1].name if candidates else None

    # "SimilarityNetwork" (base triplet loss, uses final epoch file)
    candidates = sorted(checkpoint_dir.glob("0_similarity_model_weights_epoch_*.pt"))
    return candidates[-1].name if candidates else None


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

# Regex patterns for the inference report blocks.
_RE_SECTION = re.compile(
    r"={30,}\n"
    r"(?P<anim>[A-Z]+)\s+\((?P<subset>[^)]+)\):\s*EMBEDDING L2 VS RAW FEATURE (?P<metric>GEODESIC|DTW) DISTANCE\n"
    r"={30,}",
)
_RE_PEARSON  = re.compile(r"Embedding L2 distances\s+-\s+Pearson:\s+r=(?P<r>-?\d+\.\d+)")
_RE_SPEARMAN = re.compile(r"Embedding L2 distances\s+-\s+Spearman:\s+r=(?P<r>-?\d+\.\d+)")
_RE_R2       = re.compile(r"Embedding L2\s+-\s+slope:.*?R²:\s*(?P<r2>-?\d+\.\d+)")
_RE_WINNER   = re.compile(r"Overall,\s+the\s+(?P<winner>\S+)\s+method\s+shows")


def parse_inference_output(text: str) -> Dict:
    """
    Parse the structured text produced by any of the infer_*.py scripts.

    Returns a dict keyed by (animation, metric_type) → metric values, e.g.
    {
      "walking": {
        "geodesic": {"emb_pearson": 0.47, "emb_spearman": 0.43, "emb_r2": 0.22, "winner": "Embedding_L2"},
        "dtw":      {...},
      },
      ...
    }
    """
    results: Dict = {}

    # Split the text on section headers
    parts = _RE_SECTION.split(text)
    # parts = [preamble, anim1, subset1, metric1, block1, anim2, ...]
    # groups are: anim, subset, metric, then the text block follows
    # Actually split() with groups gives: [before, g1, g2, g3, after, g1, g2, g3, after ...]

    # Walk through 4-tuples after the preamble
    idx = 1  # skip preamble
    while idx + 3 <= len(parts):
        anim   = parts[idx].lower()
        _subset = parts[idx + 1]
        metric = parts[idx + 2].lower()   # "geodesic" or "dtw"
        block  = parts[idx + 3]
        idx += 4

        m_p = _RE_PEARSON.search(block)
        m_s = _RE_SPEARMAN.search(block)
        m_r = _RE_R2.search(block)
        m_w = _RE_WINNER.search(block)

        entry = {
            "emb_pearson":  float(m_p.group("r"))  if m_p else None,
            "emb_spearman": float(m_s.group("r"))  if m_s else None,
            "emb_r2":       float(m_r.group("r2")) if m_r else None,
            "winner":       m_w.group("winner")    if m_w else None,
        }

        results.setdefault(anim, {})[metric] = entry

    return results


# ---------------------------------------------------------------------------
# Running a single experiment
# ---------------------------------------------------------------------------

def run_subprocess(cmd: List[str], env: Dict[str, str], log_path: Path) -> Tuple[int, str]:
    """Run cmd as a subprocess, tee stdout+stderr to log_path, return (returncode, combined_output)."""
    print(f"    $ {' '.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    combined = []
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        for line in proc.stdout:
            sys.stdout.write("      " + line)
            logf.write(line)
            combined.append(line)
        proc.wait()

    return proc.returncode, "".join(combined)


def run_experiment(exp: Dict, skip_training: bool = False) -> Dict:
    """
    Train (unless skip_training) then infer for one experiment.
    Returns the parsed results dict (also written to results.json).
    """
    name = exp["name"]
    exp_dir = _EXPERIMENTS_DIR / name
    chk_dir = exp_dir / "checkpoints"
    chk_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT: {name}")
    print(f"  {exp['description']}")
    print(f"{'=' * 70}")

    env = {**os.environ, "MOTION_CHECKPOINT_DIR": str(chk_dir) + "/"}

    # ------------------------------------------------------------------
    # 1. Training
    # ------------------------------------------------------------------
    if exp["train_script"] is None:
        print("  [train]  No training required (baseline).")
    elif skip_training:
        print("  [train]  Skipped (--skip-training).")
    else:
        t0 = time.time()
        train_cmd = [
            _PYTHON,
            str(_HERE / exp["train_script"]),
            *exp["train_flags"],
        ]
        log_path = exp_dir / "train.log"
        print(f"  [train]  Writing log → {log_path.relative_to(_ROOT)}")
        rc, _ = run_subprocess(train_cmd, env, log_path)
        elapsed = time.time() - t0
        if rc != 0:
            print(f"  [train]  ⚠️  Exited with code {rc} after {elapsed:.0f}s — continuing to inference anyway.")
        else:
            print(f"  [train]  ✓  Completed in {elapsed:.0f}s.")

    # ------------------------------------------------------------------
    # 2. Identify best checkpoint
    # ------------------------------------------------------------------
    chk_file = find_best_checkpoint(chk_dir, exp.get("network_class"))
    if exp.get("network_class") is not None and chk_file is None:
        print(f"  [infer]  ✗  No checkpoint found in {chk_dir} — skipping inference.")
        return {}

    if chk_file:
        print(f"  [infer]  Checkpoint selected: {chk_file}")
        env["MOTION_CHECKPOINT_FILE"] = chk_file

    # ------------------------------------------------------------------
    # 3. Inference
    # ------------------------------------------------------------------
    infer_cmd = [
        _PYTHON,
        str(_HERE / exp["infer_script"]),
        *(exp.get("extra_infer_flags") or []),
    ]
    log_path = exp_dir / "infer.log"
    print(f"  [infer]  Writing log → {log_path.relative_to(_ROOT)}")
    t0 = time.time()
    rc, output = run_subprocess(infer_cmd, env, log_path)
    elapsed = time.time() - t0
    if rc != 0:
        print(f"  [infer]  ⚠️  Exited with code {rc} after {elapsed:.0f}s.")
    else:
        print(f"  [infer]  ✓  Completed in {elapsed:.0f}s.")

    # ------------------------------------------------------------------
    # 4. Parse and save results
    # ------------------------------------------------------------------
    parsed = parse_inference_output(output)
    results = {
        "experiment": name,
        "description": exp["description"],
        "checkpoint": chk_file,
        "metrics": parsed,
    }

    results_path = exp_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [save]   Results → {results_path.relative_to(_ROOT)}")

    return results


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(all_results: List[Dict]) -> None:
    """
    Print a side-by-side table of Embedding-L2 Pearson / Spearman / R²
    for each (experiment × animation × distance_metric) combination.
    Also delegates to compare_results.py if it exists.
    """
    # Collect animation names and metric types from all results
    anim_set:   set = set()
    metric_set: set = set()
    for r in all_results:
        for anim, mdict in r.get("metrics", {}).items():
            anim_set.add(anim)
            metric_set.update(mdict.keys())

    anims   = sorted(anim_set)
    metrics = sorted(metric_set)   # e.g. ["dtw", "geodesic"]

    for anim in anims:
        for dist_metric in metrics:
            header = f"\n  {anim.upper()}  ·  Embedding L2 vs {dist_metric.upper()}"
            print(header)
            print("  " + "─" * (len(header) - 2))

            col_w = 20
            header_row = (
                f"  {'Experiment':<28}"
                f"{'Pearson r':>{col_w}}"
                f"{'Spearman r':>{col_w}}"
                f"{'R²':>{col_w}}"
                f"{'Winner':>{col_w}}"
            )
            print(header_row)
            print("  " + "─" * (len(header_row) - 2))

            for r in all_results:
                entry = r.get("metrics", {}).get(anim, {}).get(dist_metric)
                if entry is None:
                    continue
                def fmt(v):
                    return f"{v:.4f}" if v is not None else "  n/a "
                print(
                    f"  {r['experiment']:<28}"
                    f"{fmt(entry.get('emb_pearson')):>{col_w}}"
                    f"{fmt(entry.get('emb_spearman')):>{col_w}}"
                    f"{fmt(entry.get('emb_r2')):>{col_w}}"
                    f"{(entry.get('winner') or ''):>{col_w}}"
                )


def save_comparison_csv(all_results: List[Dict]) -> None:
    """Flatten all results into a tidy CSV at experiments/comparison.csv."""
    rows = []
    for r in all_results:
        for anim, mdict in r.get("metrics", {}).items():
            for dist_metric, entry in mdict.items():
                rows.append({
                    "experiment":   r["experiment"],
                    "description":  r["description"],
                    "checkpoint":   r.get("checkpoint", ""),
                    "animation":    anim,
                    "distance_type": dist_metric,
                    "emb_pearson":  entry.get("emb_pearson"),
                    "emb_spearman": entry.get("emb_spearman"),
                    "emb_r2":       entry.get("emb_r2"),
                    "winner":       entry.get("winner"),
                })
    if not rows:
        print("  No metrics to write to CSV.")
        return

    csv_path = _EXPERIMENTS_DIR / "comparison.csv"
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Comparison CSV → {csv_path.relative_to(_ROOT)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate all motion-similarity pipeline variants."
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip training and go straight to inference (uses existing checkpoints).",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        metavar="NAME[,NAME...]",
        help="Comma-separated list of experiment names to run (default: all).",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print available experiment names and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable experiments:\n")
        for exp in EXPERIMENTS:
            train_info = exp["train_script"] or "(no training)"
            print(f"  {exp['name']:<30}  {exp['description']}")
        return

    # Filter experiments
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        selected = [e for e in EXPERIMENTS if e["name"] in wanted]
        missing = wanted - {e["name"] for e in selected}
        if missing:
            print(f"Warning: unknown experiment name(s): {', '.join(sorted(missing))}")
    else:
        selected = EXPERIMENTS

    print(f"\nRunning {len(selected)} experiment(s):")
    for e in selected:
        flag = "(skip training)" if args.skip_training else ""
        print(f"  • {e['name']}  {flag}")

    _EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for exp in selected:
        result = run_experiment(exp, skip_training=args.skip_training)
        if result:
            all_results.append(result)

    # If previously-run experiments exist, merge them for the comparison table
    if not all_results:
        # Load whatever is already on disk
        for p in sorted(_EXPERIMENTS_DIR.glob("*/results.json")):
            with open(p) as f:
                all_results.append(json.load(f))

    if all_results:
        print("\n\n" + "=" * 70)
        print("  COMPARISON SUMMARY  (Embedding L2 metrics on validation subset)")
        print("=" * 70)
        print_comparison_table(all_results)
        save_comparison_csv(all_results)

        # Delegate to compare_results.py if it exists (for richer display)
        cmp_script = _HERE / "compare_results.py"
        if cmp_script.exists():
            print(f"\n  Running {cmp_script.name} for extended report…")
            subprocess.run([_PYTHON, str(cmp_script)], check=False)
    else:
        print("\nNo results to display yet.")


if __name__ == "__main__":
    main()
