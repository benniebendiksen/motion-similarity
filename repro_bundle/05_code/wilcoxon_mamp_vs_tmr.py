#!/usr/bin/env python3
"""
Paired Wilcoxon signed-rank: MAMP+pose vs TMR, in BOTH regimes (raw, and after identical perceptual
fine-tune), per action, over the n=25 matched nested-CV folds. TWO-SIDED (H1: the two differ), because
TMR wins some cells -- we want to know, per cell, whether the difference is significant or a tie.

Pairing: every source stores 25 folds tagged (repeat,fold); we align on that key.
Sources (all in ../03_results, corpus-matched comparator = cut1):
  raw MAMP+pose      : cvn_mamppose_cut1.json   [k]['folds'][i]['raw_test'][action]
  raw TMR            : tmr_raw_perfold.json      ['folds'][i]['raw_test'][action]
  ft  MAMP+pose      : rankdiag_headline.json    ['B']['folds'][i]['test'][action]
  ft  TMR            : tmr_perc_finetune_results.json ['folds'][i]['raw_test'][action]
"""
import json, os, math
from wilcoxon_paired import _rankdata  # reuse validated rank helper

RES = os.path.join(os.path.dirname(__file__), "..", "03_results")
ACTIONS = ["walking", "pointing", "picking"]

def _load_folds(fname, extract):
    d = json.load(open(os.path.join(RES, fname)))
    rows = extract(d)
    rows = sorted(rows, key=lambda r: (r["repeat"], r["fold"]))
    return rows

def mamp_raw():
    d = json.load(open(os.path.join(RES, "cvn_mamppose_cut1.json")))
    k = list(d.keys())[0]
    rows = sorted(d[k]["folds"], key=lambda r: (r["repeat"], r["fold"]))
    return {a: [r["raw_test"][a] for r in rows] for a in ACTIONS}

def tmr_raw():
    rows = _load_folds("tmr_raw_perfold.json", lambda d: d["folds"])
    return {a: [r["raw_test"][a] for r in rows] for a in ACTIONS}

def mamp_ft():
    d = json.load(open(os.path.join(RES, "rankdiag_headline.json")))
    rows = sorted(d["B"]["folds"], key=lambda r: (r["repeat"], r["fold"]))
    return {a: [r["test"][a] for r in rows] for a in ACTIONS}

def tmr_ft():
    rows = _load_folds("tmr_perc_finetune_results.json", lambda d: d["folds"])
    return {a: [r["raw_test"][a] for r in rows] for a in ACTIONS}

def wilcoxon_two_sided(x, y):
    """Two-sided paired Wilcoxon signed-rank, normal approx + continuity + tie correction.
    Returns (W=min(R+,R-), p_two_sided, median_diff)."""
    d = [xi - yi for xi, yi in zip(x, y) if (xi - yi) != 0.0]
    n = len(d)
    if n < 1:
        return 0.0, float("nan"), 0.0
    r = _rankdata([abs(v) for v in d])
    r_plus = sum(ri for ri, di in zip(r, d) if di > 0)
    r_minus = sum(ri for ri, di in zip(r, d) if di < 0)
    W = min(r_plus, r_minus)
    mn = n * (n + 1) / 4.0
    from collections import Counter
    tie = Counter(abs(v) for v in d)
    se = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - sum(t**3 - t for t in tie.values()) / 48.0)
    if se == 0:
        return W, float("nan"), 0.0
    z = (r_plus - mn)
    z = (z - 0.5) if z > 0 else (z + 0.5)   # continuity toward mean
    z /= se
    p = 2.0 * (0.5 * math.erfc(abs(z) / math.sqrt(2.0)))   # two-sided
    med = sorted([xi - yi for xi, yi in zip(x, y)])[len(x) // 2]
    return W, min(p, 1.0), med

def run(label, mamp, tmr):
    print(f"\n=== {label}: MAMP+pose vs TMR (two-sided paired Wilcoxon, n=25) ===")
    print(f"{'action':<9} {'MAMP':>6} {'TMR':>6} {'medΔ':>7} {'p':>9}  verdict")
    out = {}
    for a in ACTIONS:
        x, y = mamp[a], tmr[a]
        assert len(x) == len(y) == 25, (a, len(x), len(y))
        W, p, med = wilcoxon_two_sided(x, y)
        mm = sum(x) / len(x); tt = sum(y) / len(y)
        if p < .05:
            verdict = "MAMP+pose > TMR" if mm > tt else "TMR > MAMP+pose"
        else:
            verdict = "tie (n.s.)"
        print(f"{a:<9} {mm:>6.3f} {tt:>6.3f} {med:>+7.3f} {p:>9.2e}  {verdict}")
        out[a] = {"mamp": round(mm, 3), "tmr": round(tt, 3), "median_delta": round(med, 3),
                  "p_two_sided": p, "verdict": verdict}
    return out

results = {
    "raw": run("RAW (unsupervised)", mamp_raw(), tmr_raw()),
    "fine_tuned": run("FINE-TUNED (matched)", mamp_ft(), tmr_ft()),
}
json.dump(results, open(os.path.join(RES, "wilcoxon_mamp_vs_tmr_results.json"), "w"), indent=2)
print("\nwrote", os.path.join(RES, "wilcoxon_mamp_vs_tmr_results.json"))
