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


def _extract(mdict: Dict):
    """Return (emb, geo, dtw) metric dicts from one animation's metrics block."""
    geo = mdict.get("geodesic", {})
    dtw = mdict.get("dtw", {})
    # emb metrics are the same in both distance-type entries; prefer geodesic
    ep = geo.get("emb_pearson")  if geo.get("emb_pearson")  is not None else dtw.get("emb_pearson")
    es = geo.get("emb_spearman") if geo.get("emb_spearman") is not None else dtw.get("emb_spearman")
    er = geo.get("emb_r2")       if geo.get("emb_r2")       is not None else dtw.get("emb_r2")
    return (
        {"pearson": ep, "spearman": es, "r2": er},
        # geo/dtw raw metrics are invariant across experiments for a given animation
        {"pearson": geo.get("raw_pearson"), "spearman": geo.get("raw_spearman"), "r2": geo.get("raw_r2")},
        {"pearson": dtw.get("raw_pearson"), "spearman": dtw.get("raw_spearman"), "r2": dtw.get("raw_r2")},
    )


def ranked_table(
    results: List[Dict],
    anim_filter: Optional[str],
    sort_col: str,
) -> None:
    """
    One section per animation, one row per experiment.
    Columns: Emb-L2 | Geodesic | DTW  (Pearson / Spearman / R² each).
    Geodesic and DTW values are raw-motion baselines, invariant across experiments.
    """
    rows = []
    for r in results:
        for anim in ANIMS_ORDER:
            mdict = r.get("metrics", {}).get(anim)
            if mdict is None:
                continue
            if anim_filter and anim != anim_filter.lower():
                continue
            emb, geo, dtw = _extract(mdict)
            rows.append({
                "exp":          r["experiment"],
                "anim":         anim,
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
        print("  No data to display.")
        return

    anims = [a for a in ANIMS_ORDER if a in {r["anim"] for r in rows}]
    extra_anims = sorted({r["anim"] for r in rows} - set(ANIMS_ORDER))
    anims += extra_anims

    sort_key = {
        "pearson":  "emb_pearson",
        "spearman": "emb_spearman",
        "r2":       "emb_r2",
    }.get(sort_col, "emb_pearson")

    col_exp = 28
    col_v   = 10

    for anim in anims:
        subset = [r for r in rows if r["anim"] == anim]
        if not subset:
            continue

        subset.sort(key=lambda r: (r[sort_key] is None, -(r[sort_key] or 0)))

        valid_vals = [r[sort_key] for r in subset if r[sort_key] is not None]
        best_val   = valid_vals[0] if valid_vals else None
        second_val = valid_vals[1] if len(valid_vals) > 1 else None

        print(f"\n  {anim.upper()}  ·  Emb-L2  |  Geodesic (raw)  |  DTW (raw)")
        print("  " + "─" * 110)
        print(
            f"  {'Experiment':<{col_exp}}"
            f"  {'Emb-P':>{col_v}}  {'Emb-S':>{col_v}}  {'Emb-R²':>{col_v}}"
            f"  │  {'Geo-P':>{col_v}}  {'Geo-S':>{col_v}}  {'Geo-R²':>{col_v}}"
            f"  │  {'DTW-P':>{col_v}}  {'DTW-S':>{col_v}}  {'DTW-R²':>{col_v}}"
        )
        print("  " + "─" * 110)

        for r in subset:
            v = r[sort_key]
            is_best   = (v is not None and v == best_val)
            is_second = (v is not None and v == second_val and not is_best)
            line = (
                f"  {r['exp']:<{col_exp}}"
                f"  {_fmt(r['emb_pearson'],  is_best and sort_key=='emb_pearson',  is_second and sort_key=='emb_pearson'):>{col_v}}"
                f"  {_fmt(r['emb_spearman'], is_best and sort_key=='emb_spearman', is_second and sort_key=='emb_spearman'):>{col_v}}"
                f"  {_fmt(r['emb_r2'],       is_best and sort_key=='emb_r2',       is_second and sort_key=='emb_r2'):>{col_v}}"
                f"  │  {_fmt(r['geo_pearson']):>{col_v}}  {_fmt(r['geo_spearman']):>{col_v}}  {_fmt(r['geo_r2']):>{col_v}}"
                f"  │  {_fmt(r['dtw_pearson']):>{col_v}}  {_fmt(r['dtw_spearman']):>{col_v}}  {_fmt(r['dtw_r2']):>{col_v}}"
            )
            print(line)


# ---------------------------------------------------------------------------
# Winner count
# ---------------------------------------------------------------------------

def winner_counts(results: List[Dict]) -> None:
    """
    Count per animation how often embedding Pearson beats geodesic and DTW.
    geo/dtw baselines are raw-motion constants, so totals = num animations with data.
    """
    geo_wins: Dict[str, int] = {}
    dtw_wins: Dict[str, int] = {}
    totals:   Dict[str, int] = {}
    for r in results:
        name = r["experiment"]
        for anim, mdict in r.get("metrics", {}).items():
            emb, geo, dtw = _extract(mdict)
            totals[name] = totals.get(name, 0) + 1
            if emb["pearson"] is not None and geo["pearson"] is not None and emb["pearson"] > geo["pearson"]:
                geo_wins[name] = geo_wins.get(name, 0) + 1
            if emb["pearson"] is not None and dtw["pearson"] is not None and emb["pearson"] > dtw["pearson"]:
                dtw_wins[name] = dtw_wins.get(name, 0) + 1

    if not totals:
        return

    for label, wins_dict in [("GEODESIC", geo_wins), ("DTW", dtw_wins)]:
        print(f"\n\n  EMB BEATS {label} (Pearson, per animation)")
        print("  " + "─" * 50)
        ranked = sorted(totals.keys(), key=lambda n: -(wins_dict.get(n, 0) / totals[n]))
        for name in ranked:
            wins  = wins_dict.get(name, 0)
            total = totals[name]
            pct   = 100.0 * wins / total if total else 0
            bar   = "█" * wins + "░" * (total - wins)
            print(f"  {name:<30}  {bar}  {wins}/{total}  ({pct:.0f}%)")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(results: List[Dict], csv_path: Path) -> None:
    """
    One row per (experiment × animation).
    emb_* varies by experiment; geo_* and dtw_* are raw-motion constants
    (invariant across experiments for a given animation, repeated for convenience).
    """
    rows = []
    for r in results:
        for anim in ANIMS_ORDER:
            mdict = r.get("metrics", {}).get(anim)
            if mdict is None:
                continue
            emb, geo, dtw = _extract(mdict)
            rows.append({
                "experiment":   r["experiment"],
                "description":  r.get("description", ""),
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
