#!/usr/bin/env python3
"""
Paired Wilcoxon signed-rank tests over the n=25 nested-CV TEST folds.

Pairing basis (verified): every model's cvn_*.json stores 25 folds tagged (repeat,fold);
baselines_nested_results.json stores 75 [action,score] rows emitted rep-major/fold/action
(baselines_nested.py L84-96) using the SAME `seed+1000*rep` make_folds as the models, so
baseline triple i == fold (repeat=i//5, fold=i%5). Thus positional pairing == (repeat,fold)
pairing. Test = MAMP+pose fold-Spearman vs each comparator fold-Spearman, one-sided (greater),
per action. This is the "appropriate statistical test" for checklist item 4.11.
"""
import json, os, math

def _rankdata(a):
    """Average ranks, 1-based (ties get mean rank)."""
    idx = sorted(range(len(a)), key=lambda i: a[i])
    ranks = [0.0] * len(a)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[idx[j+1]] == a[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # mean of positions i..j, 1-based
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks

def wilcoxon_greater(x, y):
    """One-sided paired Wilcoxon signed-rank (H1: median(x-y) > 0), normal approx
    with continuity correction + tie correction. Matches scipy default (mode='approx',
    zero_method='wilcox') for n>=~20. Returns (W_statistic, p_value)."""
    d = [xi - yi for xi, yi in zip(x, y)]
    d = [v for v in d if v != 0.0]          # wilcox zero_method: drop zeros
    n = len(d)
    absd = [abs(v) for v in d]
    r = _rankdata(absd)
    r_plus = sum(ri for ri, di in zip(r, d) if di > 0)
    r_minus = sum(ri for ri, di in zip(r, d) if di < 0)
    W = min(r_plus, r_minus)               # scipy reports min(R+,R-) as statistic
    mn = n * (n + 1) / 4.0
    # tie correction on |d|
    from collections import Counter
    tie = Counter(absd)
    tie_term = sum(t**3 - t for t in tie.values())
    se = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0)
    if se == 0:
        return W, float('nan')
    # z for H1: R+ large (x>y). continuity correction toward mean.
    z = (r_plus - mn)
    z = (z - 0.5) if z > 0 else (z + 0.5)
    z /= se
    # one-sided upper-tail p
    p = 0.5 * math.erfc(z / math.sqrt(2.0))
    return W, p

class _S:  # tiny shim so call sites read like scipy
    @staticmethod
    def wilcoxon(x, y, alternative="greater", zero_method="wilcox"):
        W, p = wilcoxon_greater(x, y)
        return type("R", (), {"statistic": W, "pvalue": p})()
stats = _S()

RES = os.path.join(os.path.dirname(__file__), "..", "03_results")
BAS = os.path.join(os.path.dirname(__file__), "..", "01_baselines", "baselines_nested_results.json")
ACTIONS = ["walking", "pointing", "picking"]

def model_folds(fname, mkey=None):
    d = json.load(open(os.path.join(RES, fname)))
    mkey = mkey or list(d.keys())[0]
    folds = d[mkey]["folds"]
    # order by (repeat,fold) to guarantee alignment
    folds = sorted(folds, key=lambda r: (r["repeat"], r["fold"]))
    return {a: [f["raw_test"][a] for f in folds] for a in ACTIONS}

def baseline_folds(name):
    d = json.load(open(BAS))
    flat = d["rows"][name]  # 75 [action,score], rep-major/fold/action
    trip = [flat[i:i+3] for i in range(0, len(flat), 3)]
    out = {a: [] for a in ACTIONS}
    for t in trip:
        dd = dict((k, v) for k, v in t)
        for a in ACTIONS:
            out[a].append(dd[a])
    return out

head = model_folds("cvn_mamppose.json", "MAMP+pose")

comparators = {
    "DTW":        baseline_folds("dtw"),
    "geodesic":   baseline_folds("geo"),
    "vanilla":    model_folds("cvn_vrot1vel0.json"),
    "MLD-AE":     model_folds("cvn_mldae.json"),
    "MAMP":       model_folds("cvn_mamp.json"),
}

print(f"{'comparator':<10} {'action':<9} {'n':>3} {'W':>7} {'p(1-sided)':>11}  {'median Δ':>9}  sig")
print("-"*62)
results = {}
for name, comp in comparators.items():
    results[name] = {}
    for a in ACTIONS:
        x = head[a]; y = comp[a]
        assert len(x) == len(y) == 25
        diff = [xi - yi for xi, yi in zip(x, y)]
        # one-sided: MAMP+pose > comparator
        w = stats.wilcoxon(x, y, alternative="greater", zero_method="wilcox")
        med = sorted(diff)[len(diff)//2]
        star = "***" if w.pvalue < .001 else "**" if w.pvalue < .01 else "*" if w.pvalue < .05 else "ns"
        print(f"{name:<10} {a:<9} {len(x):>3} {w.statistic:>7.1f} {w.pvalue:>11.2e}  {med:>+9.3f}  {star}")
        results[name][a] = {"W": w.statistic, "p_greater": w.pvalue, "median_delta": med, "n": len(x)}
    print()

json.dump(results, open(os.path.join(RES, "wilcoxon_paired_results.json"), "w"), indent=2)
print("wrote", os.path.join(RES, "wilcoxon_paired_results.json"))
