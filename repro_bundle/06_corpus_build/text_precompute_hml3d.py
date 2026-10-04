#!/usr/bin/env python3
"""PHASE 2 — precompute FROZEN all-mpnet-base-v2 sentence embeddings for the captioned
hybrid corpus (hml3d_captioned.json). Uses transformers (no sentence-transformers dep):
mean-pool last_hidden_state over the attention mask, then L2-normalize == the canonical
all-mpnet-base-v2 sentence embedding. Saves {stem: tensor[n_caps, 768]} for InfoNCE sampling.
"""
import json, sys, time
import torch, numpy as np
from transformers import AutoTokenizer, AutoModel

R = "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets"
MAN = f"{R}/datasets/hml3d_captioned.json"
OUT = f"{R}/datasets/hml3d_text_mpnet.pt"
MODEL = "sentence-transformers/all-mpnet-base-v2"
SMOKE = "--smoke" in sys.argv
dev = "cuda" if torch.cuda.is_available() else "cpu"

def mean_pool(last_hidden, mask):
    m = mask.unsqueeze(-1).float()
    return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)

def main():
    man = json.load(open(MAN))
    stems = list(man)
    if SMOKE: stems = stems[:5]
    # dedupe captions across the corpus to embed each unique string once
    uniq = {}
    for s in stems:
        for c in man[s]["captions"]:
            uniq.setdefault(c, None)
    caps = list(uniq)
    print(f"clips={len(stems)} unique_captions={len(caps)} device={dev}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval()
    emb = np.zeros((len(caps), 768), dtype=np.float32)
    B = 128; t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(caps), B):
            batch = caps[i:i+B]
            enc = tok(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
            out = model(**enc).last_hidden_state
            v = mean_pool(out, enc["attention_mask"])
            v = torch.nn.functional.normalize(v, p=2, dim=1)
            emb[i:i+len(batch)] = v.cpu().numpy()
            if (i // B) % 20 == 0:
                print(f"  {i+len(batch)}/{len(caps)}  {time.time()-t0:.0f}s", flush=True)
    cap2idx = {c: k for k, c in enumerate(caps)}
    # group per stem
    data = {}
    for s in stems:
        idx = [cap2idx[c] for c in man[s]["captions"]]
        data[s] = torch.from_numpy(emb[idx].copy())
    if SMOKE:
        s0 = stems[0]
        print("[smoke] stem", s0, "caps", len(man[s0]["captions"]), "emb", tuple(data[s0].shape),
              "norm", float(data[s0][0].norm()))
        return
    torch.save({"model": MODEL, "dim": 768, "emb": data}, OUT)
    print("wrote", OUT, "clips", len(data), f"{time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
