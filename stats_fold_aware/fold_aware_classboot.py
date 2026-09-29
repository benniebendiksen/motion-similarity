#!/usr/bin/env python
"""
TMLR stats fix, tier B (PRIMARY): paired cluster bootstrap over Effort classes.

Unit of generalization = the 56 non-neutral Effort classes per action (all 25 nested-CV folds reuse
them). One draw = sample 56 classes with replacement, independently per action. Each drawn copy
keeps its class's fold in every repeat. For every (repeat, fold) the statistic is recomputed exactly
as in cv_nested: Spearman(embedding distance, d_perc) over test-fold pairs of DISTINCT classes, with
duplicate copies counted by multiplicity (pair weight c_i*c_j; equivalent to expanding the copies);
self-pairs have no d_perc and are dropped. Then the mean over the 25 folds -> the paper's number.
Checkpoint selection is held at each fold's observed selected_epoch (SELECT is disjoint from TEST).
Both models in a comparison see the SAME draw -> paired bootstrap distribution of the difference.

p (two-sided) = 2*min(P*(D<=0), P*(D>=0)), one-sided = P*(D<=0), each with the +1/(B+1) correction;
95% CI = percentile. Holm within the same pre-declared families as tier A.

GATE: with all multiplicities = 1 the statistic must reproduce every one of the 25 stored fold rho
per model/action to 1e-6, else abort.
"""
import json, os, sys, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PD = os.path.join(HERE, "pairdist")
A = ["walking", "pointing", "picking"]

FAMILIES = {
    "F1 headline vs baselines (tab:baselines)": [
        ("MAMP+pose", "DTW", "greater"), ("MAMP+pose", "Geodesic", "greater"),
        ("MAMP+pose", "Plain(recon)", "greater")],
    "F2 pose-head x U-Net 2x2 (tab:posehead, tab:archObj)": [
        ("MAMP+pose", "MAMP", "two"), ("MAMP-UNet+pose", "MAMP-UNet", "two"),
        ("MAMP-UNet", "MAMP", "two"), ("MAMP-UNet+pose", "MAMP+pose", "two")],
    "F3b backbone under reconstruction (tab:backbone)": [
        ("MLD-AE(recon)", "Plain(recon)", "two")],
    "F4 masking vs reconstruction refs (tab:sweep; MLD-VAE honesty)": [
        ("MAMP+pose", "MLD-AE(recon)", "two"), ("MAMP+pose", "MLD-VAE(recon)", "two")],
}


def wranks(v, w):
    """Average ranks of v under integer weights w (== ranks after expanding each item w times),
    returned per item as the mean rank of its copies."""
    o = np.argsort(v, kind="mergesort"); vs, ws = v[o], w[o]
    cw = np.cumsum(ws); start = cw - ws                      # copies occupy ranks start+1..cw
    r = np.empty_like(vs, dtype=float)
    i = 0; n = len(vs)
    while i < n:
        j = i
        while j + 1 < n and vs[j + 1] == vs[i]:
            j += 1
        r[i:j + 1] = (start[i] + 1 + cw[j]) / 2.0            # tie group spans start[i]+1..cw[j]
        i = j + 1
    out = np.empty_like(r); out[o] = r
    return out


def wspearman(x, y, w):
    rx, ry = wranks(x, w), wranks(y, w)
    W = w.sum(); mx = (w * rx).sum() / W; my = (w * ry).sum() / W
    cxy = (w * (rx - mx) * (ry - my)).sum()
    den = np.sqrt((w * (rx - mx) ** 2).sum() * (w * (ry - my) ** 2).sum())
    return cxy / den if den > 0 else np.nan


class Action:
    def __init__(self, a, manifest):
        z = np.load(os.path.join(PD, f"pairdist_{a}.npz"), allow_pickle=True)
        self.P = z["dperc"]; self.F = z["fold_of"]; n = self.P.shape[0]
        self.iu = np.triu_indices(n, 1)
        self.D = {}
        for k in z.files:
            if k.startswith("D|"):
                _, label, ep = k.split("|"); self.D[(label, int(ep))] = z[k]
        self.sel = {lab: {tuple(map(int, rf.split("_"))): e for rf, e in m["selected_epoch"].items()}
                    for lab, m in manifest["models"].items()}
        for b in ("DTW", "Geodesic"):
            if (b, 0) in self.D:               # only once the baseline matrices are dumped
                self.sel[b] = {(r, f): 0 for r in range(5) for f in range(5)}
        # per (rep, fold): pair index arrays among test classes with GT
        self.pairs = {}
        for r in range(5):
            for f in range(5):
                i, j = self.iu
                m = (self.F[r, i] == f) & (self.F[r, j] == f) & np.isfinite(self.P[i, j])
                self.pairs[(r, f)] = (i[m], j[m])

    def stat(self, label, c):
        """Mean over 25 folds of weighted Spearman; c = class multiplicities (len 56)."""
        vals = []
        for (r, f), (i, j) in self.pairs.items():
            w = c[i] * c[j]; k = w > 0
            if k.sum() < 4:
                continue
            D = self.D[(label, self.sel[label][(r, f)])]
            vals.append(wspearman(D[i[k], j[k]], self.P[i[k], j[k]], w[k].astype(float)))
        return float(np.nanmean(vals))

    def fold_rhos(self, label):
        one = np.ones(self.P.shape[0])
        out = {}
        for (r, f), (i, j) in self.pairs.items():
            D = self.D[(label, self.sel[label][(r, f)])]
            out[(r, f)] = wspearman(D[i, j], self.P[i, j], one[i] * one[j])
        return out


def observed_fold_rhos(label, action, manifest):
    if label in ("DTW", "Geodesic"):
        flat = json.load(open(os.path.join(HERE, "..", "repro_bundle", "01_baselines",
                                           "baselines_nested_results.json")))["rows"][
            "dtw" if label == "DTW" else "geo"]
        return {(t // 5, t % 5): s for t in range(25) for act, s in flat[3 * t:3 * t + 3] if act == action}
    jf = os.path.join(HERE, "inputs_cvn", os.path.basename(manifest["models"][label]["cvn"]))
    d = json.load(open(jf)); rows = d[list(d)[0]]["folds"]
    return {(r["repeat"], r["fold"]): r["raw_test"][action] for r in rows}


def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); order = np.argsort(ps)
    adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260925)
    args = ap.parse_args()
    manifest = json.load(open(os.path.join(PD, "manifest.json")))
    acts = {a: Action(a, manifest) for a in A}
    labels = sorted({x for fam in FAMILIES.values() for c in fam for x in c[:2]})
    labels = [l for l in labels if l in acts["walking"].sel]

    # ---- GATE: reproduce all stored fold rho ----
    worst = 0.0
    for a in A:
        for lab in labels:
            got = acts[a].fold_rhos(lab); ref = observed_fold_rhos(lab, a, manifest)
            err = max(abs(got[k] - ref[k]) for k in ref)
            worst = max(worst, err)
            print(f"gate {a:9} {lab:16} max|Δρ_fold| = {err:.2e}")
    if worst > 1e-6:
        sys.exit(f"GATE FAILED (worst {worst:.2e}) -- do not use bootstrap results")
    print("GATE PASSED\n")

    rng = np.random.default_rng(args.seed)
    boot = {a: {lab: np.empty(args.B) for lab in labels} for a in A}
    obs = {a: {lab: acts[a].stat(lab, np.ones(56)) for lab in labels} for a in A}
    for b in range(args.B):
        for a in A:
            c = np.bincount(rng.integers(0, 56, 56), minlength=56).astype(float)
            for lab in labels:
                boot[a][lab][b] = acts[a].stat(lab, c)
        if (b + 1) % 500 == 0:
            print(f"  {b + 1}/{args.B}", flush=True)
    np.savez(os.path.join(HERE, "classboot_draws.npz"),
             **{f"{a}|{lab}": boot[a][lab] for a in A for lab in labels})

    out, lines = {"B": args.B, "seed": args.seed, "models": {}, "families": {}}, []
    lines.append(f"# Class-bootstrap (B={args.B}, 56 Effort classes/action, paired)\n")
    lines.append("## Per-model mean with class-bootstrap 95% CI\n")
    lines.append("| model | " + " | ".join(A) + " |\n|---|---|---|---|")
    for lab in labels:
        cells = []
        for a in A:
            lo, hi = np.percentile(boot[a][lab], [2.5, 97.5])
            cells.append(f"{obs[a][lab]:.3f} [{lo:.3f}, {hi:.3f}]")
            out["models"].setdefault(lab, {})[a] = dict(mean=obs[a][lab], ci95=[lo, hi],
                                                       se_boot=float(boot[a][lab].std(ddof=1)))
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    lines.append("")
    for fam, comps in FAMILIES.items():
        rows = []
        for x, y, side in comps:
            if x not in labels or y not in labels:
                continue
            for a in A:
                d = boot[a][x] - boot[a][y]; B = len(d)
                lo_t = (1 + (d <= 0).sum()) / (B + 1); hi_t = (1 + (d >= 0).sum()) / (B + 1)
                p = lo_t if side == "greater" else min(1.0, 2 * min(lo_t, hi_t))
                lo, hi = np.percentile(d, [2.5, 97.5])
                rows.append(dict(x=x, y=y, action=a, side=side, delta=obs[a][x] - obs[a][y],
                                 ci95=[float(lo), float(hi)], p=float(p)))
        for r, pa in zip(rows, holm([r["p"] for r in rows])):
            r["p_holm"] = float(pa)
        out["families"][fam] = rows
        lines.append(f"## {fam}  (Holm m={len(rows)})\n")
        lines.append("| x vs y | action | Δ | 95% CI (class boot) | p | p Holm | survives |\n|---|---|---|---|---|---|---|")
        for r in rows:
            lines.append(f"| {r['x']} vs {r['y']} ({'1s' if r['side']=='greater' else '2s'}) | {r['action']} | "
                         f"{r['delta']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['p']:.2g} | "
                         f"{r['p_holm']:.2g} | {'YES' if r['p_holm'] < .05 else 'no'} |")
        lines.append("")
    json.dump(out, open(os.path.join(HERE, "results_classboot.json"), "w"), indent=1)
    open(os.path.join(HERE, "results_classboot.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
