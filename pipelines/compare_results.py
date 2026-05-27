#!/usr/bin/env python3
"""
compare_results.py — Aggregate and display saved experiment results.

Reads every  experiments/<name>/results.json  produced by run_all_experiments.py
and renders three views:

  1. A per-animation ranked table (Pearson, Spearman, R²) for each
     distance metric type (Geodesic, DTW).
  2. An overall winner count per experiment.
  3. A tidy CSV at  experiments/comparison.csv  (overwrites if exists).

Usage
-----
    python pipelines/compare_results.py                 # all saved experiments
    python pipelines/compare_results.py --anim walking  # filter by animation
    python pipelines/compare_results.py --sort pearson  # sort column
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_EXPERIMENTS_DIR = _ROOT / "experiments"

DIST_METRICS = ["geodesic", "dtw"]
ANIMS_ORDER  = ["walking", "pointing", "picking"]

# Colour helpers (ANSI, disabled on Windows or when stdout is not a tty)
_USE_COLOUR = sys.stdout.isatty() and os.name != "nt"
_GREEN  = "\033[92m" if _USE_COLOUR else ""
_YELLOW = "\033[93m" if _USE_COLOUR else ""
_RESET  = "\033[0m"  if _USE_COLOUR else ""


def _c(text: str, colour: str) -> str:
    return f"{colour}{text}{_RESET}" if colour else text


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_all_results(experiments_dir: Path) -> List[Dict]:
    results = []
    for p in sorted(experiments_dir.glob("*/results.json")):
        try:
            with open(p) as f:
                results.append(json.load(f))
        except Exception as e:
            print(f"  Warning: could not read {p}: {e}")
    return results


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float], best: bool = False, second: bool = False) -> str:
    if v is None:
        return "  —   "
    s = f"{v:+.4f}"
    if best:
        return _c(s, _GREEN)
    if second:
        return _c(s, _YELLOW)
    return s


def ranked_table(
    results: List[Dict],
    anim_filter: Optional[str],
    sort_col: str,
) -> None:
    """Print a ranked table for every (animation × distance_metric) slice."""

    # Flatten into rows
    rows = []
    for r in results:
        for anim, mdict in r.get("metrics", {}).items():
            if anim_filter and anim != anim_filter.lower():
                continue
            for dist, entry in mdict.items():
                rows.append({
                    "exp":          r["experiment"],
                    "anim":         anim,
                    "dist":         dist,
                    "pearson":      entry.get("emb_pearson"),
                    "spearman":     entry.get("emb_spearman"),
                    "r2":           entry.get("emb_r2"),
                    "raw_pearson":  entry.get("raw_pearson"),
                    "raw_spearman": entry.get("raw_spearman"),
                    "raw_r2":       entry.get("raw_r2"),
                    "winner":       entry.get("winner") or "",
                })

    if not rows:
        print("  No data to display.")
        return

    anims   = anim_filter.split(",") if anim_filter else ANIMS_ORDER
    # include any anim that actually appears in data but isn't in ANIMS_ORDER
    extra_anims = sorted({r["anim"] for r in rows} - set(ANIMS_ORDER))
    anims = [a for a in anims if a in {r["anim"] for r in rows}] + extra_anims

    sort_key = {"pearson": "pearson", "spearman": "spearman", "r2": "r2"}.get(sort_col, "pearson")

    col_exp = 28
    col_v   = 11

    for anim in anims:
        for dist in DIST_METRICS:
            subset = [r for r in rows if r["anim"] == anim and r["dist"] == dist]
            if not subset:
                continue

            # Sort descending by chosen metric (None → last)
            subset.sort(key=lambda r: (r[sort_key] is None, -(r[sort_key] or 0)))

            # Mark best and second-best
            valid_vals = [r[sort_key] for r in subset if r[sort_key] is not None]
            best_val   = valid_vals[0] if valid_vals else None
            second_val = valid_vals[1] if len(valid_vals) > 1 else None

            title = f"\n  {anim.upper()}  ·  Embedding L2 vs {dist.upper()}"
            print(title)
            print("  " + "─" * 100)
            hdr = (
                f"  {'Experiment':<{col_exp}}"
                f"{'Emb-Pear':>{col_v}}"
                f"{'Raw-Pear':>{col_v}}"
                f"{'Emb-Spear':>{col_v}}"
                f"{'Raw-Spear':>{col_v}}"
                f"{'Emb-R²':>{col_v}}"
                f"{'Raw-R²':>{col_v}}"
                f"{'Beats?':>8}"
            )
            print(hdr)
            print("  " + "─" * 100)

            for r in subset:
                v = r[sort_key]
                is_best   = (v is not None and v == best_val)
                is_second = (v is not None and v == second_val and not is_best)
                ep = r["pearson"]
                rp = r["raw_pearson"]
                beats = ("YES" if ep > rp else "no") if (ep is not None and rp is not None) else ""
                line = (
                    f"  {r['exp']:<{col_exp}}"
                    f"{_fmt(r['pearson'],  is_best and sort_key=='pearson',  is_second and sort_key=='pearson'):>{col_v}}"
                    f"{_fmt(r['raw_pearson']):>{col_v}}"
                    f"{_fmt(r['spearman'], is_best and sort_key=='spearman', is_second and sort_key=='spearman'):>{col_v}}"
                    f"{_fmt(r['raw_spearman']):>{col_v}}"
                    f"{_fmt(r['r2'],       is_best and sort_key=='r2',       is_second and sort_key=='r2'):>{col_v}}"
                    f"{_fmt(r['raw_r2']):>{col_v}}"
                    f"{beats:>8}"
                )
                print(line)


# ---------------------------------------------------------------------------
# Winner count
# ---------------------------------------------------------------------------

def winner_counts(results: List[Dict]) -> None:
    """Print how many times each experiment's embedding outperformed raw features."""
    counts: Dict[str, int] = {}
    totals: Dict[str, int] = {}
    for r in results:
        name = r["experiment"]
        for anim, mdict in r.get("metrics", {}).items():
            for dist, entry in mdict.items():
                totals[name] = totals.get(name, 0) + 1
                if entry.get("winner", "").startswith("Embedding"):
                    counts[name] = counts.get(name, 0) + 1

    if not totals:
        return

    print("\n\n  WINS  (embedding beats raw features)")
    print("  " + "─" * 50)
    ranked = sorted(totals.keys(), key=lambda n: -(counts.get(n, 0) / totals[n]))
    for name in ranked:
        wins  = counts.get(name, 0)
        total = totals[name]
        pct   = 100.0 * wins / total if total else 0
        bar   = "█" * wins + "░" * (total - wins)
        print(f"  {name:<30}  {bar}  {wins}/{total}  ({pct:.0f}%)")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(results: List[Dict], csv_path: Path) -> None:
    rows = []
    for r in results:
        for anim, mdict in r.get("metrics", {}).items():
            for dist, entry in mdict.items():
                ep = entry.get("emb_pearson")
                rp = entry.get("raw_pearson")
                rows.append({
                    "experiment":    r["experiment"],
                    "description":   r.get("description", ""),
                    "checkpoint":    r.get("checkpoint", ""),
                    "animation":     anim,
                    "distance_type": dist,
                    "emb_pearson":   ep,
                    "emb_spearman":  entry.get("emb_spearman"),
                    "emb_r2":        entry.get("emb_r2"),
                    "raw_pearson":   rp,
                    "raw_spearman":  entry.get("raw_spearman"),
                    "raw_r2":        entry.get("raw_r2"),
                    "emb_beats_raw": (ep > rp) if (ep is not None and rp is not None) else None,
                    "winner":        entry.get("winner", ""),
                })
    if not rows:
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved → {csv_path.relative_to(_ROOT)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display and compare saved experiment results."
    )
    parser.add_argument(
        "--anim", type=str, default=None,
        metavar="NAME",
        help="Filter to a single animation type (walking / pointing / picking).",
    )
    parser.add_argument(
        "--sort", type=str, default="pearson",
        choices=["pearson", "spearman", "r2"],
        help="Metric to sort the ranked table by (default: pearson).",
    )
    parser.add_argument(
        "--no-csv", action="store_true",
        help="Skip writing the CSV file.",
    )
    args = parser.parse_args()

    results = load_all_results(_EXPERIMENTS_DIR)
    if not results:
        print(f"No results found under {_EXPERIMENTS_DIR}.")
        print("Run  python pipelines/run_all_experiments.py  first.")
        return

    print(f"\nLoaded {len(results)} experiment(s): {', '.join(r['experiment'] for r in results)}\n")

    print("=" * 70)
    print("  RANKED METRICS  (Embedding L2 on validation subset)")
    print(f"  Sorted by: {args.sort.upper()}  |  Green = best, Yellow = second")
    print("=" * 70)
    ranked_table(results, args.anim, args.sort)

    winner_counts(results)

    if not args.no_csv:
        export_csv(results, _EXPERIMENTS_DIR / "comparison.csv")


if __name__ == "__main__":
    main()
