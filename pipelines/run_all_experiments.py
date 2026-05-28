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
_RESULTS_DIR = _HERE / "results"                 # pipelines/results/ — committed output
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
            "--embedding-dir",   str(_DATASETS / "lma_perform_walking_ae_paired"),
            "--embedding-dir-2", str(_DATASETS / "lma_perform_pointing_ae_paired"),
            "--embedding-dir-3", str(_DATASETS / "lma_perform_picking_ae_paired"),
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
            "--embedding-dir",   str(_DATASETS / "lma_perform_walking_ae_paired"),
            "--embedding-dir-2", str(_DATASETS / "lma_perform_pointing_ae_paired"),
            "--embedding-dir-3", str(_DATASETS / "lma_perform_picking_ae_paired"),
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
            "--walking-dir", str(_DATASETS / "lma_perform_walking_ae_paired"),
            "--pointing-dir", str(_DATASETS / "lma_perform_pointing_ae_paired"),
            "--picking-dir",  str(_DATASETS / "lma_perform_picking_ae_paired"),
        ],
        "infer_script": "infer_emb_vec.py",
        "network_class": "EmbeddingRefiningSimilarityNetwork",
    },
    # ── 7. AE embedding vectors → MLP via EmbeddingRefiningSimilarityNetwork, perception loss ──
    {
        "name": "emb_vec_percloss",
        "description": "AE emb → MLP (EmbeddingRefiningSimilarityNetwork)  |  perception loss",
        "train_script": "train_emb_vec_percloss.py",
        "train_flags": [
            "--animation", "walking",
            "--use-perception-loss",
            "--embedding-dir", str(_DATASETS / "lma_perform_walking_ae_paired"),
        ],
        "infer_script": "infer_emb_vec.py",
        "network_class": "EmbeddingRefiningSimilarityNetwork",
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
_RE_PEARSON      = re.compile(r"Embedding L2 distances\s+-\s+Pearson:\s+r=(?P<r>-?\d+\.\d+)")
_RE_SPEARMAN     = re.compile(r"Embedding L2 distances\s+-\s+Spearman:\s+r=(?P<r>-?\d+\.\d+)")
_RE_R2           = re.compile(r"Embedding L2\s+-\s+slope:.*?R²:\s*(?P<r2>-?\d+\.\d+)")
_RE_RAW_PEARSON  = re.compile(r"Raw feature \w+ distances\s+-\s+Pearson:\s+r=(?P<r>-?\d+\.\d+)")
_RE_RAW_SPEARMAN = re.compile(r"Raw feature \w+ distances\s+-\s+Spearman:\s+r=(?P<r>-?\d+\.\d+)")
_RE_RAW_R2       = re.compile(r"Raw Feature \w+\s+-\s+slope:.*?R²:\s*(?P<r2>-?\d+\.\d+)")
_RE_WINNER       = re.compile(r"Overall,\s+the\s+(?P<winner>\S+)\s+method\s+shows")


def parse_inference_output(text: str) -> Dict:
    """
    Parse the structured text produced by any of the infer_*.py scripts.

    Returns a dict keyed by animation → distance_type → metric values, e.g.
    {
      "walking": {
        "geodesic": {
          "emb_pearson": 0.47, "emb_spearman": 0.43, "emb_r2": 0.22,
          "raw_pearson": 0.38, "raw_spearman": 0.35, "raw_r2": 0.14,
          "winner": "Embedding_L2"
        },
        "dtw": {...},
      },
      ...
    }
    """
    results: Dict = {}

    # split() with groups returns [before, g1, g2, g3, after_block, g1, ...]
    parts = _RE_SECTION.split(text)

    idx = 1  # skip preamble
    while idx + 3 <= len(parts):
        anim   = parts[idx].lower()
        _subset = parts[idx + 1]
        metric = parts[idx + 2].lower()   # "geodesic" or "dtw"
        block  = parts[idx + 3]
        idx += 4

        m_p  = _RE_PEARSON.search(block)
        m_s  = _RE_SPEARMAN.search(block)
        m_r  = _RE_R2.search(block)
        m_rp = _RE_RAW_PEARSON.search(block)
        m_rs = _RE_RAW_SPEARMAN.search(block)
        m_rr = _RE_RAW_R2.search(block)
        m_w  = _RE_WINNER.search(block)

        entry = {
            "emb_pearson":  float(m_p.group("r"))   if m_p  else None,
            "emb_spearman": float(m_s.group("r"))   if m_s  else None,
            "emb_r2":       float(m_r.group("r2"))  if m_r  else None,
            "raw_pearson":  float(m_rp.group("r"))  if m_rp else None,
            "raw_spearman": float(m_rs.group("r"))  if m_rs else None,
            "raw_r2":       float(m_rr.group("r2")) if m_rr else None,
            "winner":       m_w.group("winner")     if m_w  else None,
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

_ANIM_ORDER = ["walking", "pointing", "picking"]


def _emb_geo_dtw(mdict: Dict):
    """Extract (emb metrics, geodesic raw metrics, dtw raw metrics) from one animation dict."""
    geo = mdict.get("geodesic", {})
    dtw = mdict.get("dtw", {})
    # emb metrics are identical in both distance-type entries; prefer geodesic
    ep = geo.get("emb_pearson")  if geo.get("emb_pearson")  is not None else dtw.get("emb_pearson")
    es = geo.get("emb_spearman") if geo.get("emb_spearman") is not None else dtw.get("emb_spearman")
    er = geo.get("emb_r2")       if geo.get("emb_r2")       is not None else dtw.get("emb_r2")
    return (
        {"pearson": ep, "spearman": es, "r2": er},
        {"pearson": geo.get("raw_pearson"), "spearman": geo.get("raw_spearman"), "r2": geo.get("raw_r2")},
        {"pearson": dtw.get("raw_pearson"), "spearman": dtw.get("raw_spearman"), "r2": dtw.get("raw_r2")},
    )


def print_comparison_table(all_results: List[Dict]) -> None:
    """
    One section per animation; one row per experiment.
    Columns: Emb-L2  |  Geodesic  |  DTW  (Pearson / Spearman / R² each).
    """
    def fmt(v):
        return f"{v:.4f}" if v is not None else "  n/a"

    for anim in _ANIM_ORDER:
        print(f"\n  {anim.upper()}  ·  Emb-L2 | Geodesic | DTW  (Pearson / Spearman / R²)")
        print(f"  {'─'*110}")
        print(
            f"  {'Experiment':<28}"
            f"  {'Emb-P':>7}  {'Emb-S':>7}  {'Emb-R²':>7}"
            f"  │  {'Geo-P':>7}  {'Geo-S':>7}  {'Geo-R²':>7}"
            f"  │  {'DTW-P':>7}  {'DTW-S':>7}  {'DTW-R²':>7}"
        )
        print(f"  {'─'*110}")

        for r in all_results:
            mdict = r.get("metrics", {}).get(anim)
            if mdict is None:
                continue
            emb, geo, dtw = _emb_geo_dtw(mdict)
            print(
                f"  {r['experiment']:<28}"
                f"  {fmt(emb['pearson']):>7}  {fmt(emb['spearman']):>7}  {fmt(emb['r2']):>7}"
                f"  │  {fmt(geo['pearson']):>7}  {fmt(geo['spearman']):>7}  {fmt(geo['r2']):>7}"
                f"  │  {fmt(dtw['pearson']):>7}  {fmt(dtw['spearman']):>7}  {fmt(dtw['r2']):>7}"
            )


def save_comparison_csv(all_results: List[Dict]) -> None:
    """One row per (experiment × animation): emb, geodesic, and DTW as column groups."""
    rows = []
    for r in all_results:
        for anim in _ANIM_ORDER:
            mdict = r.get("metrics", {}).get(anim)
            if mdict is None:
                continue
            emb, geo, dtw = _emb_geo_dtw(mdict)
            rows.append({
                "experiment":   r["experiment"],
                "description":  r["description"],
                "checkpoint":   r.get("checkpoint", ""),
                "animation":    anim,
                "emb_pearson":  emb["pearson"],
                "emb_spearman": emb["spearman"],
                "emb_r2":       emb["r2"],
                "geo_pearson":  geo["pearson"],
                "geo_spearman": geo["spearman"],
                "geo_r2":       geo["r2"],
                "dtw_pearson":  dtw["pearson"],
                "dtw_spearman": dtw["spearman"],
                "dtw_r2":       dtw["r2"],
            })
    if not rows:
        print("  No metrics to write to CSV.")
        return

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _RESULTS_DIR / "comparison.csv"
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
