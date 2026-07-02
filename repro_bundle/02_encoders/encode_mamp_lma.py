"""Extract per-clip embeddings from a pretrained MAMP encoder for LMA clips.
Runs the encoder with mask_ratio=0 (full sequence), then pools patch features
to one vector/clip. Pooling: mean (default) or attention (learned, --pool attn
uses a simple softmax over a learned query — but for a frozen encoder we default
to mean, which is MAMP-native linprobe pooling). Saves <stem>_emb.pt.
"""
import os, sys, argparse
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent          # MAMP/
_ROOT = _HERE.parent
for p in (str(_ROOT), str(_HERE)):
    if p not in sys.path: sys.path.insert(0, p)

from model_mamp.transformer import Transformer
from common.MotionDataset import extract_root_and_rotations
from common.norm_utils import load_norm_stats
from bvhReader.bvh import BVH
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="config/lma_mamp_pretrain.yaml")
    ap.add_argument("--action", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pool", default="mean", choices=["mean", "max"])
    ap.add_argument("--lma-dir", default="datasets/lma_perform_reorganized_cmu")
    ap.add_argument("--norm-stats", default="norm_stats.npz",
                    help="norm_stats file the encoder was TRAINED with (cut1 -> norm_stats_cut1.npz)")
    args = ap.parse_args()

    device = torch.device("cpu")
    cfg = yaml.safe_load(open(_HERE / args.config))
    model = Transformer(**cfg["model_args"]).to(device).eval()
    st = torch.load(_HERE / args.ckpt, map_location=device, weights_only=False)
    sd = st.get("model", st.get("model_state_dict", st))
    model.load_state_dict(sd, strict=False)

    ns = load_norm_stats(_ROOT / "datasets" / args.norm_stats)
    print(f"  [encode_mamp] norm_stats={args.norm_stats}", flush=True)
    mean = ns["mean"].astype(np.float32); std = ns["std"].astype(np.float32)
    WIN = cfg["model_args"]["num_frames"]

    out_dir = _ROOT / args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    clips = sorted(p for p in (_ROOT / args.lma_dir).glob(args.action + "_*.bvh")
                   if "Mirror" not in p.stem)
    n = 0; dim = 0
    for src in clips:
        b = BVH(); b.load(str(src))
        raw = extract_root_and_rotations(b).astype(np.float32)
        raw = raw[:WIN] if len(raw) >= WIN else np.concatenate(
            [raw, np.tile(raw[-1:], (WIN-len(raw),1,1))], axis=0)
        x = (raw - mean) / std; np.clip(x,-10,10,out=x)        # [T,V,C]
        xt = torch.from_numpy(x).permute(2,0,1)[None,:,:,:,None]  # [1,C,T,V,1]
        N,C,T,V,M = xt.shape
        xt = xt.permute(0,4,2,3,1).contiguous().view(N*M, T, V, C)  # mirror model.forward
        with torch.no_grad():
            feats, _, _, _ = model.forward_encoder(xt, mask_ratio=0.0, motion_aware_tau=0.0)
            # feats: [1, 1020, dim_feat]
            if args.pool == "mean":
                emb = feats.mean(dim=1)        # [1, dim]
            else:
                emb = feats.max(dim=1).values
        emb = emb.cpu().contiguous(); dim = emb.shape[1]
        torch.save(emb, out_dir / (src.stem + "_emb.pt")); n += 1
    print("%s -> %s : %d files dim=%d (pool=%s)" % (args.action, out_dir.name, n, dim, args.pool))


if __name__ == "__main__":
    main()
