#!/usr/bin/env python
"""
TMLR stats fix, tier A: fold-aware inference by averaging folds WITHIN each repeat.

The 25 nested-CV scores (5 repeats x 5 folds) are not independent: every repeat re-partitions
the SAME 56 Effort classes, so fold scores from different repeats reuse test stimuli. Averaging
the 5 folds of a repeat gives one estimate per full pass over all classes; the 5 repeat-means
are the (conservative) independent-ish units. Test: paired t on the 5 per-repeat differences
(df=4). A Wilcoxon on n=5 has a p-floor of 1/32 (one-sided) and so cannot survive Holm; it is
not used. The OLD 25-fold Wilcoxon p is reported alongside for comparison only.

Holm is applied within pre-declared families (one family per paper table / claim group).
All comparisons pair on (repeat, fold); every source uses make_folds(seed=42+1000*rep).
"""
import json, os, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "repro_bundle", "03_results")
CHI = os.path.join(HERE, "inputs_chimera")
A = ["walking", "pointing", "picking"]


def cvn(path, field="raw_test"):
    d = json.load(open(path))
    k = list(d)[0]
    rows = d[k]["folds"] if "folds" in d[k] else d[k]
    return _grid(rows, field)


def _grid(rows, field):
    g = np.full((3, 5, 5), np.nan)                         # action x repeat x fold
    for r in rows:
        for i, a in enumerate(A):
            g[i, r["repeat"], r["fold"]] = r[field][a]
    assert not np.isnan(g).any(), "missing (repeat,fold) cell"
    return g


def baseline(name):
    flat = json.load(open(os.path.join(RES, "..", "01_baselines",
                                       "baselines_nested_results.json")))["rows"][name]
    g = np.full((3, 5, 5), np.nan)
    for t in range(25):                                   # rep-major / fold / action
        for act, s in flat[3 * t:3 * t + 3]:
            g[A.index(act), t // 5, t % 5] = s
    assert not np.isnan(g).any()
    return g


def plainrows(path, key=None, field="test"):
    d = json.load(open(path))
    rows = d[key]["folds"] if key else d["folds"]
    return _grid(rows, field)


R = lambda f: os.path.join(RES, f)
C = lambda f: os.path.join(CHI, f)
M = {
    "MAMP+pose":          cvn(R("cvn_mamppose.json")),
    "MAMP":               cvn(R("cvn_mamp.json")),
    "MAMP-UNet":          cvn(R("cvn_mampuencfull.json")),
    "MAMP-UNet+pose":     cvn(C("cvn_mampuencfullpose.json")),
    "DTW":                baseline("dtw"),
    "Geodesic":           baseline("geo"),
    "Plain(recon)":       cvn(R("cvn_vrot1vel0.json")),
    "Plain(recon+vel)":   cvn(R("cvn_vrot1vel1.json")),
    "Plain(vel)":         cvn(R("cvn_vrot0vel1.json")),
    "Plain+skips(recon)": cvn(C("cvn_vanskips_full_results.json")),
    "Plain-35M(recon)":   cvn(C("cvn_bigcap_full_results.json")),
    "MLD-AE(recon)":      cvn(R("cvn_mldae.json")),
    "MLD-AE_full(recon)": cvn(C("cvn_mldae_full_results.json")),
    "MLD-AE-noskip":      cvn(C("cvn_mldnoskip_full_results.json")),
    "MLD-AE(recon+vel)":  cvn(R("cvn_mldaevel.json")),
    "MLD-AE(vel)":        cvn(R("cvn_aevelonly.json")),
    "MLD-VAE(recon)":     cvn(R("cvn_mldvae.json")),
    # corpus-matched (cut1) track -- NEVER mixed with the full-corpus rows above
    "cut1 MAMP+pose":     cvn(R("cvn_mamppose_cut1.json")),
    "cut1 TMR":           plainrows(R("tmr_raw_perfold.json"), field="raw_test"),
    "cut1 MAMP+pose+FT":  plainrows(R("rankdiag_headline.json"), key="B"),
    "cut1 TMR+FT":        plainrows(R("tmr_perc_finetune_results.json"), field="raw_test"),
}

# (family, x, y, sidedness)  -- 'greater' tests x>y; 'two' tests x!=y
FAMILIES = {
    "F1 headline vs baselines (tab:baselines)": [
        ("MAMP+pose", "DTW", "greater"), ("MAMP+pose", "Geodesic", "greater"),
        ("MAMP+pose", "Plain(recon)", "greater")],
    "F2 pose-head x U-Net 2x2 (tab:posehead, tab:archObj)": [
        ("MAMP+pose", "MAMP", "two"), ("MAMP-UNet+pose", "MAMP-UNet", "two"),
        ("MAMP-UNet", "MAMP", "two"), ("MAMP-UNet+pose", "MAMP+pose", "two")],
    "F3 backbone + skips under reconstruction (tab:backbone, 4-corner)": [
        ("MLD-AE(recon)", "Plain(recon)", "two"),
        ("Plain+skips(recon)", "Plain(recon)", "two"),
        ("MLD-AE-noskip", "MLD-AE_full(recon)", "two")],
    "F4 masking vs reconstruction refs (tab:sweep; MLD-VAE honesty)": [
        ("MAMP+pose", "MLD-AE(recon)", "two"), ("MAMP+pose", "MLD-VAE(recon)", "two")],
    "F5 velocity objectives (tab:velLoss)": [
        ("Plain(recon+vel)", "Plain(recon)", "two"), ("Plain(vel)", "Plain(recon)", "two"),
        ("MLD-AE(recon+vel)", "MLD-AE(recon)", "two"), ("MLD-AE(vel)", "MLD-AE(recon)", "two")],
    "F6 capacity control (supp tab:capacity)": [
        ("Plain-35M(recon)", "Plain(recon)", "two")],
    "F7 TMR corpus-matched (tab:tmr)": [
        ("cut1 MAMP+pose", "cut1 TMR", "two"), ("cut1 MAMP+pose+FT", "cut1 TMR+FT", "two")],
    "F8 perceptual fine-tune lift (cut1)": [
        ("cut1 MAMP+pose+FT", "cut1 MAMP+pose", "greater")],
}


def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); order = np.argsort(ps)
    adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return adj


def test(x, y, side):
    d_fold = (x - y)                                       # [5 rep, 5 fold]
    d_rep = d_fold.mean(axis=1)                            # 5 repeat-means
    alt = "greater" if side == "greater" else "two-sided"
    t = stats.ttest_1samp(d_rep, 0.0, alternative=alt)
    w = stats.wilcoxon(d_fold.ravel(), alternative=alt)     # OLD (dependent) reference
    se_rep = d_rep.std(ddof=1) / np.sqrt(5)
    return dict(delta=float(d_rep.mean()), se_rep=float(se_rep),
                ci95=[float(d_rep.mean() - stats.t.ppf(.975, 4) * se_rep),
                      float(d_rep.mean() + stats.t.ppf(.975, 4) * se_rep)],
                d_rep=[round(float(v), 4) for v in d_rep],
                p_rep=float(t.pvalue), p_old_wilcoxon25=float(w.pvalue),
                n_rep_same_sign=int(max((d_rep > 0).sum(), (d_rep < 0).sum())))


def main():
    out, lines = {}, []
    # per-model repeat-mean table (means unchanged; new SE = SD of 5 repeat-means / sqrt 5)
    lines.append("## Per-model: mean | old SE (25 folds) | fold-aware SE (5 repeat-means)\n")
    lines.append("| model | " + " | ".join(A) + " |\n|---|---|---|---|")
    for name, g in M.items():
        cells = []
        for i in range(3):
            rm = g[i].mean(axis=1)
            cells.append(f"{g[i].mean():.3f} ({g[i].std()/5:.3f} → {rm.std(ddof=1)/np.sqrt(5):.3f})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    for fam, comps in FAMILIES.items():
        rows = []
        for x, y, side in comps:
            for i, a in enumerate(A):
                r = test(M[x][i], M[y][i], side); r.update(x=x, y=y, action=a, side=side)
                rows.append(r)
        for r, pa in zip(rows, holm([r["p_rep"] for r in rows])):
            r["p_rep_holm"] = float(pa)
        for r, pa in zip(rows, holm([r["p_old_wilcoxon25"] for r in rows])):
            r["p_old_holm"] = float(pa)
        out[fam] = rows
        lines.append(f"## {fam}  (Holm m={len(rows)})\n")
        lines.append("| x vs y | action | Δ | 95% CI (t4) | repeats same sign | old p (W25) → Holm | "
                     "repeat-mean p → Holm | verdict |\n|---|---|---|---|---|---|---|---|")
        for r in rows:
            old = r["p_old_holm"] < .05; new = r["p_rep_holm"] < .05
            verdict = ("SURVIVES" if old and new else "LOST" if old and not new
                       else "GAINED" if new else "null (both)")
            lines.append(f"| {r['x']} vs {r['y']} ({'1s' if r['side']=='greater' else '2s'}) | {r['action']} | "
                         f"{r['delta']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | "
                         f"{r['n_rep_same_sign']}/5 | {r['p_old_wilcoxon25']:.2g} → {r['p_old_holm']:.2g} | "
                         f"{r['p_rep']:.2g} → {r['p_rep_holm']:.2g} | {verdict} |")
        lines.append("")
    json.dump(out, open(os.path.join(HERE, "results_repeatmean.json"), "w"), indent=1)
    open(os.path.join(HERE, "results_repeatmean.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
