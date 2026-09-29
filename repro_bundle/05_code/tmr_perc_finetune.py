#!/usr/bin/env python3
"""
TMR perceptual fine-tune --- the SYMMETRIC control for the "fine-tuned MAMP+pose beats TMR"
claim. Applies the IDENTICAL perceptual ranking objective + nested-CV protocol as
perc_finetune.py, but fine-tunes TMR's own (ACTOR-style) motion encoder instead of MAMP+pose.

Rationale: reviewers can object that we only perceptually-supervise OUR encoder and compare to
RAW TMR. This gives TMR the same fine-tune, so the comparison is fine-tuned-vs-fine-tuned.

Design (well-posed given the non-differentiable BVH->SMPL front-end):
  - The SMPL fit is FROZEN upstream (already computed). We start from the persisted per-stimulus
    263-dim HumanML3D guofeats: bvh2tmr_pipeline/scaled/feats263/<Cap_e0_e1_e2_e3>.npy [T-1,263].
  - We fine-tune TMR's motion_encoder (ACTORStyleEncoder) end-to-end from those fixed 263-feats:
      263-feat -> normalizer -> ACTOR encoder -> 256-d latent -> L2 -> triplet-rank loss.
  - Same folds (make_folds seed 42+1000*rep), early-stop on SELECT (peak raw-Spearman), eval TEST.
  - Freeze the text branch; fine-tune only the motion encoder (that is the representation compared).

Run on chimera in the TMR conda env (hydra/lightning/transformers). See sbatch alongside.
Compare output tmr_perc_finetune_results.json vs MAMP+pose rankdiag_headline.json.
"""
import sys, os, json, time, argparse
import numpy as np
import torch

# --- paths (chimera) ---
TR   = "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets"
MS   = "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/motion-similarity"
TMR_REPO = os.path.join(TR, "learned_baselines", "tmr", "repo")
TMR_RUN  = os.path.join(TR, "learned_baselines", "tmr", "models", "models", "tmr_humanml3d_guoh3dfeats")
FEATS263 = os.path.join(TR, "learned_baselines", "bvh2tmr_pipeline", "scaled", "feats263")
sys.path.insert(0, MS)          # Config, dperc_folds (d_perc + folds, dep-light)
# NOTE: we import dperc_folds (NOT cv_preliminary) to avoid the `src` namespace collision with
# TMR's repo (both use a bare `src` package). dperc_folds carries byte-faithful copies of the
# four helpers we need with no `src`/pymo/tensorflow chain. TMR_REPO is added to sys.path only
# INSIDE load_tmr_encoder (after dperc_folds is imported), so `src` resolves to TMR there.
from Config import Config
import dperc_folds as cv                                   # get_inverse_direct_comparison_value,
                                                           # get_directed_alpha, make_folds, load_action_keys

ACTIONS = ["walking", "pointing", "picking"]
CAP = {"walking": "Walking", "pointing": "Pointing", "picking": "Picking"}


# ---- perceptual triplet losses (byte-faithful copies from perc_finetune.py; torch-only, no cv) ----
# Inlined rather than imported because perc_finetune imports cv_preliminary at module top, which
# would drag in the `src` chain and re-trigger the collision. These are identical to the MAMP arms.
def _norm01(d):
    return (d - d.min()) / (d.max() - d.min() + 1e-8)

def _triplet_fixed(embs, keys, alpha_fn, fixed_margin):
    idx = {k: i for i, k in enumerate(keys)}
    pairs = []
    for a_i in range(len(keys)):
        for b_i in range(a_i + 1, len(keys)):
            if alpha_fn(keys[a_i], keys[b_i]) is None:
                continue
            pairs.append((embs[idx[keys[a_i]]] - embs[idx[keys[b_i]]]).pow(2).sum().clamp_min(1e-12).sqrt())
    if not pairs:
        return None
    return torch.relu(fixed_margin - _norm01(torch.stack(pairs))).mean()

def _triplet_rank(embs, keys, dperc):
    idx = {k: i for i, k in enumerate(keys)}
    raw, pmap = {}, {}
    for a_i in range(len(keys)):
        for b_i in range(a_i + 1, len(keys)):
            v = dperc(keys[a_i], keys[b_i])
            if v is None:
                continue
            d = (embs[idx[keys[a_i]]] - embs[idx[keys[b_i]]]).pow(2).sum().clamp_min(1e-12).sqrt()
            raw[(a_i, b_i)] = d; pmap[(a_i, b_i)] = v
    if len(raw) < 2:
        return None
    ks = list(raw)
    dnv = _norm01(torch.stack([raw[k] for k in ks]))
    dn = {ks[t]: dnv[t] for t in range(len(ks))}
    def pair_dist(i, j):
        return dn[(i, j)] if (i, j) in dn else dn.get((j, i))
    terms = []
    for i in range(len(keys)):
        rated = [j for j in range(len(keys)) if j != i
                 and pair_dist(min(i, j), max(i, j)) is not None
                 and pmap.get((min(i, j), max(i, j))) is not None]
        for a in range(len(rated)):
            for b in range(a + 1, len(rated)):
                j, k = rated[a], rated[b]
                pj, pk = pmap[(min(i, j), max(i, j))], pmap[(min(i, k), max(i, k))]
                if pj == pk:
                    continue
                near, far = (j, k) if pj < pk else (k, j)
                gap = abs(pk - pj)
                terms.append(torch.relu(pair_dist(min(i, near), max(i, near))
                                        - pair_dist(min(i, far), max(i, far)) + gap))
    if not terms:
        return None
    return torch.stack(terms).mean()

def _triplet_alpha(embs, keys, alpha_fn):
    idx = {k: i for i, k in enumerate(keys)}
    pairs, margins = [], []
    for a_i in range(len(keys)):
        for b_i in range(a_i + 1, len(keys)):
            al = alpha_fn(keys[a_i], keys[b_i])
            if al is None:
                continue
            aL, aR = al
            d = (embs[idx[keys[a_i]]] - embs[idx[keys[b_i]]]).pow(2).sum().clamp_min(1e-12).sqrt()
            pairs.append(d); margins.append(aL)
            pairs.append(d); margins.append(aR)
    if not pairs:
        return None
    dn = _norm01(torch.stack(pairs))
    m = torch.tensor(margins, device=dn.device, dtype=dn.dtype)
    return torch.where(m >= 0, torch.relu(m - dn), torch.relu(dn + m)).mean()

def ts(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ---- TMR encoder loading + differentiable encode -------------------------------------------
def load_tmr_encoder(device):
    """Load TMR, return (motion_encoder trainable, normalizer, collate_x_dict, encode_fn)."""
    # TMR's src.load / Hydra config resolution is cwd-sensitive (relative config_path);
    # the official encode_motion.py runs from the repo root. chdir there before loading.
    _prev_cwd = os.getcwd()
    os.chdir(TMR_REPO)
    # `src` is a namespace package (no __init__.py) in BOTH codebases. By now the MS side
    # (TripletMining etc.) may have bound `src*` in sys.modules to the MS location, shadowing
    # TMR's. Evict any `src*` entries so the TMR `src` resolves fresh from TMR_REPO. Safe: all
    # MS d_perc/fold objects we need are already built and hold their own refs; nothing below
    # re-imports MS `src`.
    for _m in [k for k in list(sys.modules) if k == "src" or k.startswith("src.")]:
        del sys.modules[_m]
    # Namespace packages aggregate EVERY `src/` on sys.path, and MS's src (no load.py) shadows
    # TMR's. Temporarily REMOVE MS-side paths so `src` resolves UNAMBIGUOUSLY to TMR_REPO. Safe:
    # all MS d_perc/fold objects are already constructed; nothing below imports MS.
    _saved_path = list(sys.path)
    sys.path[:] = [p for p in sys.path if "virtual_reality/motion-similarity" not in p]
    sys.path.insert(0, TMR_REPO)
    from src.load import load_model_from_cfg
    from src.config import read_config           # from src.config (NOT demo.model, which needs transformers)
    from src.data.collate import collate_x_dict
    from hydra.utils import instantiate
    cfg = read_config(TMR_RUN)
    cfg["run_dir"] = TMR_RUN     # config.json stores a RELATIVE run_dir; make it absolute so
                                 # load_model_from_cfg finds last_weights/*.pt regardless of cwd
    model = load_model_from_cfg(cfg, "last", eval_mode=False, device=device)  # eval_mode False -> trainable
    normalizer = instantiate(cfg["data"]["motion_loader"]["normalizer"])
    # fine-tune ONLY the motion encoder; freeze text branch + decoder
    for p in model.parameters():
        p.requires_grad = False
    for p in model.motion_encoder.parameters():
        p.requires_grad = True
    os.chdir(_prev_cwd)
    # restore MS-inclusive sys.path (but keep TMR modules loaded in sys.modules so later
    # model.encode() calls -- which touch TMR src.* -- still resolve from the cached modules)
    sys.path[:] = _saved_path
    if TMR_REPO not in sys.path:
        sys.path.append(TMR_REPO)
    return model, normalizer, collate_x_dict

def feats_path(action, key):
    return os.path.join(FEATS263, CAP[action] + "_" + "_".join(map(str, key)) + ".npy")

def load_feats(action, key, normalizer, device):
    p = feats_path(action, key)
    if not os.path.exists(p):
        return None
    m = torch.from_numpy(np.load(p)).to(torch.float)
    m = normalizer(m)
    return m.to(device)

def encode_one(model, collate_x_dict, motion):
    """Differentiable 263-feat [T,263] -> 256-d latent (mirrors encode_motion.py, sample_mean)."""
    x_dict = collate_x_dict([{"x": motion, "length": len(motion)}])
    return model.encode(x_dict, sample_mean=True)[0].squeeze(0)   # [256]

# ---- eval: raw Spearman(L2, d_perc) on a key set (mirrors perc_finetune.spearman_eval) -----
def spearman_eval(model, collate_x_dict, normalizer, action, keys, module, device):
    from scipy import stats
    model.eval()
    embs = {}
    with torch.no_grad():
        for k in keys:
            if k == (0, 0, 0, 0):
                continue
            m = load_feats(action, k, normalizer, device)
            if m is None:
                continue
            embs[k] = encode_one(model, collate_x_dict, m).cpu().numpy()
    kl = sorted(embs); d, p = [], []
    for i in range(len(kl)):
        for j in range(i + 1, len(kl)):
            v = cv.get_inverse_direct_comparison_value((action, kl[i]), (action, kl[j]), module)
            if v is not None:
                d.append(float(np.linalg.norm(embs[kl[i]] - embs[kl[j]]))); p.append(v)
    return stats.spearmanr(d, p).correlation if len(d) > 3 else float("nan")

# ---- one fold: fine-tune on TRAIN, early-stop on SELECT, report TEST -----------------------
def finetune_fold(train_keys, select_keys, test_keys, modules, normalizer, collate_x_dict,
                  device, epochs, lr, loss_mode):
    model, normalizer, collate_x_dict = load_tmr_encoder(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    best_sel, best_state, best_ep = -2.0, None, -1
    for ep in range(epochs):
        model.train(); opt.zero_grad(); total = 0.0
        for a in ACTIONS:
            kl = [k for k in sorted(train_keys[a]) if k != (0, 0, 0, 0) and os.path.exists(feats_path(a, k))]
            if len(kl) < 2:
                continue
            embs = torch.stack([encode_one(model, collate_x_dict, load_feats(a, k, normalizer, device)) for k in kl])
            dp = lambda k1, k2, a=a: cv.get_inverse_direct_comparison_value((a, k1), (a, k2), modules[a])
            af = lambda kL, kR, a=a: cv.get_directed_alpha((a, kL), (a, kR), modules[a])
            if loss_mode == "rank":
                L = _triplet_rank(embs, kl, dp)
            elif loss_mode == "fixed":
                L = _triplet_fixed(embs, kl, af, 0.25)
            elif loss_mode == "alpha":
                L = _triplet_alpha(embs, kl, af)
            else:
                raise ValueError(loss_mode)
            if L is not None:
                L.backward(); total += float(L)
        opt.step()
        # early-stop on SELECT mean raw-Spearman
        sel = np.nanmean([spearman_eval(model, collate_x_dict, normalizer, a, select_keys[a], modules[a], device)
                          for a in ACTIONS])
        if sel > best_sel:
            best_sel, best_ep = sel, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= 15:      # patience, mirrors project early-stop
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    ts(f"      early-stop ep={best_ep} SELECT-meanS={best_sel:.3f}")
    return {a: spearman_eval(model, collate_x_dict, normalizer, a, test_keys[a], modules[a], device)
            for a in ACTIONS}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-5)           # low LR: foreign encoder, forgetting risk
    ap.add_argument("--loss", choices=["rank", "fixed", "alpha"], default="rank")
    ap.add_argument("--out", default=os.path.join(MS, "tmr_perc_finetune_results.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config()
    modules, all_keys = {}, {}
    for a in ACTIONS:
        modules[a], all_keys[a] = cv.load_action_keys(a, cfg)
    _, normalizer, collate_x_dict = load_tmr_encoder(device)

    folds_out = []
    for rep in range(args.repeats):
        folds = {a: cv.make_folds(all_keys[a], args.k, args.seed + 1000 * rep) for a in ACTIONS}
        for f in range(args.k):
            test  = {a: folds[a][f] for a in ACTIONS}
            sel   = {a: folds[a][(f + 1) % args.k] for a in ACTIONS}
            train = {a: set().union(*[folds[a][g] for g in range(args.k) if g not in (f, (f + 1) % args.k)])
                     for a in ACTIONS}
            ts(f"rep{rep} fold{f}: fine-tuning TMR ({args.loss})")
            res = finetune_fold(train, sel, test, modules, normalizer, collate_x_dict,
                                device, args.epochs, args.lr, args.loss)
            folds_out.append({"repeat": rep, "fold": f, "raw_test": res})
            ts(f"  -> TEST {res}")
            json.dump({"folds": folds_out}, open(args.out, "w"), indent=2)

    # aggregate
    agg = {}
    for a in ACTIONS:
        vals = [fo["raw_test"][a] for fo in folds_out if fo["raw_test"][a] == fo["raw_test"][a]]
        agg[a] = {"mean": float(np.mean(vals)), "sem": float(np.std(vals, ddof=1) / np.sqrt(len(vals))), "n": len(vals)}
    json.dump({"folds": folds_out, "aggregate": agg}, open(args.out, "w"), indent=2)
    ts(f"DONE. aggregate={agg}  -> {args.out}")

if __name__ == "__main__":
    main()
