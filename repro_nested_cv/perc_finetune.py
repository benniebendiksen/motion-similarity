#!/usr/bin/env python
"""
Perceptual fine-tuning of MAMP+pose@400 (user option 2): backprop a perceptual-alignment
loss INTO the encoder so it learns to POOL in a way that preserves perceptually-discriminative
structure (e.g. pointing arm/wrist timing) that mean-pooling otherwise averages away.

Two variants, run inside the leakage-clean nested-CV fold structure:
  A) FROZEN-BLOCKS + LEARNABLE-POOL : transformer blocks frozen; replace mean-pool with a
     learnable attention-pool + small head; train ONLY those. Tests whether the timing the
     patches still contain can be recovered by a better pooling. No catastrophic forgetting.
  B) LIGHT FULL FINE-TUNE : unfreeze the whole encoder, low LR + early-stop. More capacity to
     reshape what patches encode. Forgetting risk mitigated by low LR + early stop on SELECT.

Per fold: fine-tune on TRAIN-rated clips, EARLY-STOP on SELECT (peak raw-Spearman), report on
the fold's untouched TEST. Strict leakage: the encoder's perceptual training never sees TEST.

The perceptual loss is the distance<->d_perc alignment (same target as create_batch_integrated_loss):
for clip pairs present in the batch with human ratings, push pooled-embedding L2 distance toward
d_perc = 1 - count_normalized[(0,2)]. Direct, differentiable, no semi-hard freeze.
"""
import os, sys, glob, argparse, time, json, random
import numpy as np
import torch
import torch.nn as nn
import yaml

TR_ROOT = "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets"
MS_ROOT = "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/motion-similarity"
sys.path.insert(0, MS_ROOT)
sys.path.insert(0, os.path.join(TR_ROOT, "MAMP"))   # model_mamp.transformer
sys.path.insert(0, TR_ROOT)                          # common.MotionDataset, bvhReader.bvh
os.chdir(MS_ROOT)

from Config import Config
import cv_preliminary as cv     # reuse d_perc lookup, raw_spearman, folds, action keys

ACTIONS = ["walking", "pointing", "picking"]
ACTION_CAP = {"walking": "Walking", "pointing": "Pointing", "picking": "Picking"}
BARS = {"walking": 0.478, "pointing": 0.370, "picking": 0.431}
LMA_DIR = os.path.join(TR_ROOT, "datasets", "lma_perform_reorganized_cmu")


# ---- encoder (MAMP) loading + per-clip patch features --------------------------------------
def load_mamp(ckpt_epoch=400):
    from model_mamp.transformer import Transformer
    from common.norm_utils import load_norm_stats
    cfg = yaml.safe_load(open(os.path.join(TR_ROOT, "MAMP", "config", "lma_mamp_pretrain.yaml")))
    model = Transformer(**cfg["model_args"])
    sd = torch.load(os.path.join(TR_ROOT, "MAMP", "output_dir", "lma_mamp_pose_holdout",
                                 f"checkpoint-{ckpt_epoch}.pth"), map_location="cpu",
                    weights_only=False)
    model.load_state_dict(sd.get("model", sd), strict=False)
    ns = load_norm_stats(os.path.join(TR_ROOT, "datasets", "norm_stats.npz"))
    return model, ns["mean"].astype(np.float32), ns["std"].astype(np.float32), cfg


def load_clip_tensor(action, effort, mean, std, win):
    """Load one rated LMA clip -> normalized [1, C, T, V, 1] tensor (encoder input)."""
    from common.MotionDataset import extract_root_and_rotations
    from bvhReader.bvh import BVH
    stem = f"{ACTION_CAP[action]}_{'_'.join(map(str, effort))}"
    fp = os.path.join(LMA_DIR, stem + ".bvh")
    b = BVH(); b.load(fp)
    raw = extract_root_and_rotations(b).astype(np.float32)        # [T,V,C]
    T = raw.shape[0]
    if T < win:
        raw = np.concatenate([raw, np.tile(raw[-1:], (win - T, 1, 1))], 0)
    raw = raw[:win]
    x = (raw - mean) / std; np.clip(x, -10, 10, out=x)          # [T,V,C]
    # mirror encode_mamp_lma EXACTLY: [T,V,C] -> permute(2,0,1) -> [C,T,V] -> [1,C,T,V,1]
    xt = torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1)[None, :, :, :, None]  # [1,C,T,V,1]
    N, C, T, V, M = xt.shape
    return xt.permute(0, 4, 2, 3, 1).contiguous().view(N * M, T, V, C)  # [1, T, V, C]


# ---- learnable attention pool over patches -------------------------------------------------
class AttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Parameter(torch.randn(dim) * 0.02)
        self.proj = nn.Linear(dim, dim)
    def forward(self, feats):           # feats [B, P, dim]
        w = torch.softmax(feats @ self.q / (feats.shape[-1] ** 0.5), dim=1)  # [B,P]
        pooled = (w.unsqueeze(-1) * feats).sum(1)                            # [B,dim]
        return self.proj(pooled)


class PercModel(nn.Module):
    """Encoder (frozen or trainable) -> pool (mean or learnable attn) -> L2-comparable embedding."""
    def __init__(self, encoder, dim, learnable_pool, train_encoder):
        super().__init__()
        self.encoder = encoder
        self.train_encoder = train_encoder
        if not train_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.learnable_pool = learnable_pool
        self.pool = AttnPool(dim) if learnable_pool else None
    def forward(self, x):               # x [B,C,T,V,1]
        ctx = torch.enable_grad() if self.train_encoder else torch.no_grad()
        with ctx:
            feats, _, _, _ = self.encoder.forward_encoder(x, mask_ratio=0.0, motion_aware_tau=0.0)
        if self.train_encoder:
            feats = feats
        else:
            feats = feats.detach()
        return self.pool(feats) if self.learnable_pool else feats.mean(dim=1)


# ---- perceptual loss: align pairwise L2 with d_perc ---------------------------------------
def perceptual_loss(embs, keys, dperc):
    """embs [N,d] for classes `keys`; dperc(k1,k2)->float. MSE(normalized L2, d_perc) over GT pairs."""
    d, p = [], []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            v = dperc(keys[i], keys[j])
            if v is not None:
                d.append((embs[i] - embs[j]).pow(2).sum().clamp_min(1e-12).sqrt())
                p.append(v)
    if len(d) < 2:
        return None
    d = torch.stack(d); p = torch.tensor(p, device=d.device, dtype=d.dtype)
    dn = (d - d.min()) / (d.max() - d.min() + 1e-8)     # normalize to [0,1] like the eval
    return ((dn - p) ** 2).mean()


def spearman_eval(model, action, keys, clips, module, device):
    model.eval()
    embs = {}
    with torch.no_grad():
        for k in keys:
            if k == (0, 0, 0, 0) or (action, k) not in clips:
                continue
            embs[k] = model(clips[(action, k)].to(device)).squeeze(0).cpu().numpy()
    kl = sorted(embs)
    from scipy import stats
    d, p = [], []
    for i in range(len(kl)):
        for j in range(i + 1, len(kl)):
            v = cv.get_inverse_direct_comparison_value((action, kl[i]), (action, kl[j]), module)
            if v is not None:
                d.append(float(np.linalg.norm(embs[kl[i]] - embs[kl[j]]))); p.append(v)
    return stats.spearmanr(d, p).correlation if len(d) > 3 else float("nan")


def finetune_fold(variant, train_keys, select_keys, test_keys, modules, clips, dim, encoder,
                  device, epochs, lr, ts):
    """Fine-tune (variant A frozen+pool / B full) on TRAIN, early-stop on SELECT, eval TEST."""
    learnable_pool = (variant == "A")
    train_encoder = (variant == "B")
    model = PercModel(encoder, dim, learnable_pool, train_encoder).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    best_sel, best_state, best_ep = -2.0, None, -1
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        total = 0.0
        for a in ACTIONS:
            kl = [k for k in sorted(train_keys[a]) if k != (0, 0, 0, 0) and (a, k) in clips]
            if len(kl) < 2:
                continue
            embs = torch.stack([model(clips[(a, k)].to(device)).squeeze(0) for k in kl])
            dp = lambda k1, k2, a=a: cv.get_inverse_direct_comparison_value((a, k1), (a, k2), modules[a])
            L = perceptual_loss(embs, kl, dp)
            if L is not None:
                L.backward(); total += float(L)
        opt.step()
        every = 1 if os.environ.get("PERC_DIAG") else 5
        if ep % every == 0 or ep == epochs - 1:
            sel = np.nanmean([spearman_eval(model, a, select_keys[a], clips, modules[a], device)
                              for a in ACTIONS])
            if os.environ.get("PERC_DIAG"):
                ts(f"        ep{ep:3d} train_loss={total:.4f} SELECT-meanS={sel:.4f}")
            if sel > best_sel:
                best_sel, best_ep = sel, ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    ts(f"      variant {variant}: early-stop ep={best_ep} SELECT-meanS={best_sel:.3f}")
    return {a: spearman_eval(model, a, test_keys[a], clips, modules[a], device) for a in ACTIONS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["A", "B"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr-A", type=float, default=1e-3)
    ap.add_argument("--lr-B", type=float, default=1e-5)
    ap.add_argument("--ckpt-epoch", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(MS_ROOT, "perc_finetune_results.json"))
    args = ap.parse_args()
    cv.ACTIONS = ACTIONS
    def ts(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = Config()
    encoder, mean, std, mcfg = load_mamp(args.ckpt_epoch)
    encoder = encoder.to(device)
    dim = mcfg["model_args"]["dim_feat"]
    win = mcfg["model_args"]["num_frames"]
    ts(f"loaded MAMP+pose@{args.ckpt_epoch}, dim_feat={dim}, win={win}")

    modules, all_keys = {}, {}
    for a in ACTIONS:
        modules[a], all_keys[a] = cv.load_action_keys(a, cfg)
    # preload all rated clips once (canonical only)
    clips = {}
    for a in ACTIONS:
        for k in all_keys[a] + [(0, 0, 0, 0)]:
            try:
                clips[(a, k)] = load_clip_tensor(a, k, mean, std, win)
            except Exception:
                pass
    ts(f"preloaded {len(clips)} clip tensors")

    results = {v: {"folds": []} for v in args.variants}
    for rep in range(args.repeats):
        folds = {a: cv.make_folds(all_keys[a], args.k, args.seed + 1000 * rep) for a in ACTIONS}
        for f in range(args.k):
            test = {a: folds[a][f] | {(0, 0, 0, 0)} for a in ACTIONS}
            sel_i = (f + 1) % args.k
            select = {a: folds[a][sel_i] | {(0, 0, 0, 0)} for a in ACTIONS}
            tr_i = [i for i in range(args.k) if i != f and i != sel_i]
            train = {a: set().union(*[folds[a][i] for i in tr_i]) | {(0, 0, 0, 0)} for a in ACTIONS}
            ts(f"[r{rep}f{f}]")
            for variant in args.variants:
                lr = args.lr_A if variant == "A" else args.lr_B
                # fresh encoder copy per variant/fold (avoid weight carryover)
                enc2, _, _, _ = load_mamp(args.ckpt_epoch); enc2 = enc2.to(device)
                tt = finetune_fold(variant, train, select, test, modules, clips, dim, enc2,
                                   device, args.epochs, lr, ts)
                results[variant]["folds"].append({"repeat": rep, "fold": f, "test": tt})
                json.dump(results, open(args.out, "w"), indent=2)
                ts(f"      variant {variant} TEST={ {a: round(tt[a],3) for a in ACTIONS} }")

    ts("===== AGGREGATE (perceptual fine-tune, fold-mean +/- std) =====")
    for variant in args.variants:
        ts(f"  --- variant {variant} ---")
        for a in ACTIONS:
            vals = [r["test"][a] for r in results[variant]["folds"]
                    if r["test"][a] == r["test"][a]]
            m = float(np.mean(vals)); s = float(np.std(vals)); sem = s / np.sqrt(len(vals))
            ts(f"    {a:9} {m:.3f} +/- {s:.3f} (SEM {sem:.3f}) bar {BARS[a]} "
               f"| mean {'>' if m > BARS[a] else '<='} bar | 95%CI-clears: {(m - 1.96*sem) > BARS[a]}")
        results[variant]["aggregate"] = {
            a: {"mean": float(np.mean([r['test'][a] for r in results[variant]['folds'] if r['test'][a]==r['test'][a]])) }
            for a in ACTIONS}
    json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
