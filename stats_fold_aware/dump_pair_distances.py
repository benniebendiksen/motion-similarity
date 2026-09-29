#!/usr/bin/env python
"""
TMLR stats fix, tier B input: dump FULL pairwise distance matrices so the nested-CV statistic can be
recomputed under a cluster (Effort-class) bootstrap. No training; encode only.

For each model: read its canonical cvn_*.json, take the DISTINCT per-fold selected epochs, encode the 57
unmirrored clips/action at each (cv.encode -> identical encode path as cv_nested), and store the 56x56
L2 distance matrix over non-neutral Effort keys. Baselines: full 56x56 DTW and DTW-geodesic matrices
(same pairwise functions as baselines_nested.py). Also stores d_perc (NaN where no GT) and the exact
fold partitions (cv.make_folds, seed 42+1000*rep) so the local gate can reproduce all 25 fold rho.

Output: <out>/pairdist_<action>.npz  (+ <out>/manifest.json)
"""
import os, sys, json, glob, time, argparse
import numpy as np

MS_ROOT = "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/motion-similarity"
TR_ROOT = "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets"
sys.path.insert(0, MS_ROOT); os.chdir(MS_ROOT)
import torch
from Config import Config
import cv_preliminary as cv

ACTIONS = ["walking", "pointing", "picking"]
CVN = os.path.join(TR_ROOT, "cvn_results")
# label -> (ENCODERS registry name, canonical cvn json)
MODELS = {
    "MAMP+pose":         ("MAMP+pose",         f"{CVN}/cvn_mamppose.json"),
    "MAMP":              ("MAMP",              f"{CVN}/cvn_mamp.json"),
    "MAMP-UNet":         ("MAMP-uencfull",     f"{CVN}/cvn_mampuencfull.json"),
    "MAMP-UNet+pose":    ("MAMP-uencfullpose", f"{CVN}/cvn_mampuencfullpose.json"),
    "Plain(recon)":      ("vanilla-rot1vel0",  f"{CVN}/cvn_vrot1vel0.json"),
    "MLD-AE(recon)":     ("MLD-AE-holdout",    f"{CVN}/cvn_mldae.json"),
    "MLD-VAE(recon)":    ("MLD-VAE-holdout",   f"{CVN}/cvn_mldvae.json"),
    "cut1 MAMP+pose":    ("MAMP+pose-cut1",    f"{CVN}/cvn_mamppose_cut1.json"),
}


def ts(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def l2_matrix(emb_dir, action, keys):
    cap = cv.ACTION_CAP[action]
    E = []
    for k in keys:
        fp = os.path.join(emb_dir, f"{cap}_{'_'.join(map(str, k))}_emb.pt")
        E.append(torch.load(fp, map_location="cpu", weights_only=True).flatten().numpy())
    E = np.stack(E).astype(np.float64)
    return np.linalg.norm(E[:, None, :] - E[None, :, :], axis=-1)


def tuples_to_matrix(tuples, keys):
    idx = {k: i for i, k in enumerate(keys)}
    D = np.full((len(keys), len(keys)), np.nan)
    for d, k1, k2 in tuples:
        e1 = k1[1] if isinstance(k1, tuple) and len(k1) == 2 and isinstance(k1[0], str) else k1
        e2 = k2[1] if isinstance(k2, tuple) and len(k2) == 2 and isinstance(k2[0], str) else k2
        i, j = idx[tuple(e1)], idx[tuple(e2)]
        D[i, j] = D[j, i] = float(d)
    np.fill_diagonal(D, 0.0)
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--work-dir", default="/tmp/pairdist_work")
    ap.add_argument("--out", default=os.path.join(MS_ROOT, "stats_fold_aware"))
    args = ap.parse_args()
    cv.ACTIONS = ACTIONS
    os.makedirs(args.out, exist_ok=True); os.makedirs(args.work_dir, exist_ok=True)
    cfg = Config()
    modules, keys = {}, {}
    for a in ACTIONS:
        modules[a], keys[a] = cv.load_action_keys(a, cfg)
        keys[a] = [tuple(k) for k in keys[a]]
        assert len(keys[a]) == 56, (a, len(keys[a]))

    # load any partial output so reruns only add what is missing
    store = {a: {} for a in ACTIONS}
    for a in ACTIONS:
        fp = os.path.join(args.out, f"pairdist_{a}.npz")
        if os.path.exists(fp):
            z = np.load(fp, allow_pickle=True)
            store[a] = {k: z[k] for k in z.files}

    def save():
        for a in ACTIONS:
            np.savez(os.path.join(args.out, f"pairdist_{a}.npz"), **store[a])

    manifest = {"models": {}, "keys": {a: [list(k) for k in keys[a]] for a in ACTIONS}}
    for a in ACTIONS:
        n = len(keys[a]); P = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i + 1, n):
                v = cv.get_inverse_direct_comparison_value((a, keys[a][i]), (a, keys[a][j]), modules[a])
                if v is not None:
                    P[i, j] = P[j, i] = v
        store[a]["keys"] = np.array(keys[a]); store[a]["dperc"] = P
        # exact fold partitions: folds[rep][fold] -> list of key indices
        F = np.full((5, n), -1, int)
        for rep in range(5):
            folds = cv.make_folds(keys[a], 5, 42 + 1000 * rep)
            for f, ks in enumerate(folds):
                for k in ks:
                    F[rep, keys[a].index(tuple(k))] = f
        assert (F >= 0).all()
        store[a]["fold_of"] = F
        ts(f"{a}: {np.isfinite(P[np.triu_indices(n, 1)]).sum()} GT pairs")
    save()

    for label in args.models:
        enc, jf = MODELS[label]
        d = json.load(open(jf)); rows = d[list(d)[0]]["folds"]
        sel = {(r["repeat"], r["fold"]): r["selected_epoch"] for r in rows}
        eps = sorted(set(sel.values()))
        ck = dict(cv.list_checkpoints(enc, 1))
        ts(f"== {label} ({enc}): {len(eps)} distinct selected epochs {eps}")
        for ep in eps:
            if ep not in ck:
                raise RuntimeError(f"{label}: checkpoint for epoch {ep} not found in {cv.ENCODERS[enc]['ckpt_dir']}")
            for a in ACTIONS:
                name = f"D|{label}|{ep}"
                if name in store[a]:
                    continue
                od = os.path.join(args.work_dir, f"{enc}_ep{ep}_{a}")
                if not glob.glob(os.path.join(od, "*_emb.pt")):
                    cv.encode(enc, ck[ep], a, od)
                store[a][name] = l2_matrix(od, a, keys[a])
            save(); ts(f"   {label} ep{ep} done")
        manifest["models"][label] = {"enc": enc, "cvn": jf,
                                     "selected_epoch": {f"{r}_{f}": e for (r, f), e in sel.items()}}
        json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)

    if args.baselines:
        sys.path.insert(0, os.path.join(MS_ROOT, "pipelines"))
        import importlib.util as ilu
        spec = ilu.spec_from_file_location("iev", os.path.join(MS_ROOT, "pipelines", "infer_emb_vec.py"))
        iev = ilu.module_from_spec(spec); spec.loader.exec_module(iev)
        for a in ACTIONS:
            if "D|DTW|0" in store[a] and "D|Geodesic|0" in store[a]:
                continue
            raw = iev.get_raw_features_without_dataloader(a, cfg, valid_indices=None,
                                                         balance_class_frame_counts=False)
            rb = {}
            for k, v in raw.items():
                e = k[1] if isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], str) else k
                rb[tuple(e)] = v
            sub = {k: rb[k] for k in keys[a]}
            store[a]["D|DTW|0"] = tuples_to_matrix(iev.calculate_real_variable_length_dtw(sub), keys[a])
            store[a]["D|Geodesic|0"] = tuples_to_matrix(iev.compute_geodesic_distances_dtw(sub, num_joints=28), keys[a])
            save(); ts(f"baselines {a} done")
    ts("ALL DONE")


if __name__ == "__main__":
    main()
