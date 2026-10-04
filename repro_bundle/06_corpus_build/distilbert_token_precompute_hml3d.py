#!/usr/bin/env python3
"""PHASE 2 (re-arch) — precompute FROZEN distilbert-base-uncased TOKEN embeddings for the
captioned hybrid corpus, per UNIQUE caption (variable-length [N,768]). These feed a trainable
ACTORStyleEncoder text head (TMR recipe), NOT a pooled sentence vector. The existing MPNet
sentence vectors (cut1_hml_text_mpnet.pt) stay — for the false-neg FILTER only.

Frozen: distilbert used purely as a feature extractor (eval, no_grad). Matches TMR's
text_to_token_emb = distilbert-base-uncased (reference_tmr_architecture).
"""
import json, sys, time
import torch, numpy as np
from transformers import AutoTokenizer, AutoModel

R = "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets"
MAN = f"{R}/datasets/hml3d_captioned.json"
OUT = f"{R}/datasets/hml3d_text_distilbert_tokens.pt"
MODEL = "distilbert-base-uncased"
SMOKE = "--smoke" in sys.argv
dev = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    man = json.load(open(MAN))
    stems = list(man)
    if SMOKE: stems = stems[:5]
    uniq = {}
    for s in stems:
        for c in man[s]["captions"]:
            uniq.setdefault(c, None)
    caps = list(uniq)
    print(f"clips={len(stems)} unique_captions={len(caps)} device={dev}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False

    emb = {}           # caption_str -> float16 [N,768] (real tokens, no padding)
    B = 128; t0 = time.time(); maxlen = 0
    with torch.no_grad():
        for i in range(0, len(caps), B):
            batch = caps[i:i+B]
            enc = tok(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
            hs = model(**enc).last_hidden_state          # [b, N, 768]
            lens = enc["attention_mask"].sum(1)          # real token counts
            for j, c in enumerate(batch):
                n = int(lens[j]); maxlen = max(maxlen, n)
                emb[c] = hs[j, :n].to(torch.float16).cpu().numpy()
            if (i // B) % 20 == 0:
                print(f"  {i+len(batch)}/{len(caps)}  {time.time()-t0:.0f}s  maxlen={maxlen}", flush=True)
    if SMOKE:
        c0 = caps[0]
        print(f"[smoke] cap={c0!r} tokens={emb[c0].shape} dtype={emb[c0].dtype} maxlen={maxlen}")
        return
    torch.save({"model": MODEL, "dim": 768, "max_tokens": maxlen, "emb": emb}, OUT)
    tot = sum(v.shape[0] for v in emb.values())
    print(f"wrote {OUT}  captions={len(emb)}  total_tokens={tot}  maxlen={maxlen}  {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
