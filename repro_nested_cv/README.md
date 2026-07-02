# Repeated Nested Cross-Validation — Reproduction Package

This directory contains the **exact code** used for the paper's §6 results: the leakage-clean,
repeated nested cross-validation (CV) that evaluates every motion encoder and the geometric
baselines on identical held-out folds, plus the perceptual fine-tuning experiment (§6.3).

All numbers in §6.1, §6.3, §6.4, and §6.6 of `docs/paper_draft.md` come from these scripts.

---

## 1. What each script does

| script | purpose | produces |
|--------|---------|----------|
| `cv_nested.py` | **The main harness.** Repeated (R×K) leakage-clean nested CV for any encoder: per outer fold, select a checkpoint on a disjoint inner fold by raw held-out Spearman, evaluate raw embedding distances on the untouched test fold. Reports fold-mean ± std ± SEM with a 95%-CI-clears-bar verdict. | §6.1, §6.4, §6.6 raw grid |
| `baselines_nested.py` | Re-evaluates **DTW** and the **DTW-corrected, unpadded geodesic** baselines on the *identical* R×K test folds, fold-averaged the same way (apples-to-apples vs. the encoder). | §6.1 baseline row |
| `perc_finetune.py` | **Perceptual fine-tuning of the encoder** (§6.3): back-propagates a human-derived alpha-as-margin perceptual loss into the MAMP+pose encoder (variant A = learnable attention-pool, blocks frozen; variant B = light full-encoder fine-tune), strictly inside each CV fold. | §6.3 perceptual table |
| `cv_preliminary.py` | Shared library imported by the others: the encoder registry, checkpoint enumeration, the `encode()` dispatch (MAMP / vanilla AETransformer / MLD AE+VAE), `raw_spearman`, `make_folds`, `train_triplet_on_fold`, and the `d_perc` lookup. | (imported, not run directly) |

---

## 2. The protocol (what makes it leakage-clean and apples-to-apples)

- **Repeated nested CV:** 5 repeats × 5 outer folds = **25 held-out test estimates**. Folds are
  magnitude-stratified per action with seed `42 + 1000·repeat`.
- **3-way split per outer fold (no selection leakage):**
  - **TEST** = the outer fold — reported, *never* seen during training or checkpoint selection.
  - **SELECT** = a disjoint inner fold — used *only* to pick the encoder checkpoint by raw
    held-out Spearman (a metric-independent, perceptually-grounded early-stop criterion).
  - **TRAIN** = the remaining folds — used to fit any learned metric / perceptual fine-tune.
- **Canonical evaluation universe = exactly 57 effort classes per action** (1 neutral + 24
  states [two non-zero efforts] + 32 drives [three non-zero efforts]). The stored embedding
  dirs also contain *non-canonical*, **unrated** clips; these carry no `d_perc` and must be kept
  out of the shuffle pool, or they silently perturb which rated classes land in each fold
  (most severely for pointing). `cv_preliminary.load_action_keys` returns the canonical set.
- **Matched selection pressure + PSP-bounded candidate windows.** Every encoder gets **exactly 7
  candidate checkpoints** (identical inner-loop degrees of freedom for all models). The 7 are
  placed evenly within each model's **`[init, PSP]` window**, where **PSP (Perceptual-Spearman
  Plateau)** is the epoch at which that model's SELECT-fold raw perceptual Spearman plateaus
  (`psp_profiler.py`). Per-model windows differ — and that is *correct*: each window ends at that
  model's **measured perceptual ceiling**, not at where its training job happened to stop. This
  replaces the earlier "span each model's own trajectory" rule, whose endpoints were an artifact
  of job scheduling (a model run longer got a wider net — an unintended selection advantage).
  Candidates are never placed past PSP, which excludes the over-trained regime (§4.4.1) by
  construction. **PSP methodology rules (pre-registered, applied uniformly):**
  1. *Uniform epoch grid* for profiling: every model is sampled at **the same 100-epoch grid**
     `{100, 200, …}` — *not* per-model strides and *not* "every checkpoint" (save intervals
     differ: MAMP 20, MLD-VAE 25, MLD-AE/vanilla 100). 100 = LCM of the save intervals, so every
     model has a checkpoint at each grid point → identical sampling resolution for all. PSP onset
     is therefore resolvable to ±100 epochs for every model (the coarsest common grid; finer is
     impossible without re-saving the 100-interval models, and uniformity is the non-cherry-pick
     choice).
  2. *Sufficiency = `reached_psp`, not a separate loss-plateau check.* `reached=True` means the
     trajectory provably extends past the perceptual plateau (a stronger condition than the loss
     plateau, which arrives much later — e.g. MAMP+pose PSP 240 vs loss-flat ~1199). `reached=False`
     ⇒ the model is resumed **until PSP is observed** (perceptual criterion), not until loss flattens.
  3. *Pre-registered detector parameters* (fixed once, never tuned per model): noise band
     `MIN_BAND=0.02`, 3-point moving-average smoothing, PSP = earliest epoch within the peak's
     SEM band, `reached` = NOT(peak-at-end AND end-leg-gain > band).
  4. *Two mandatory sensitivity checks* (the cherry-pick defenses): **(a)** band-sensitivity —
     PSPs must be stable for `MIN_BAND ∈ {0.01, 0.02, 0.03}`; **(b)** window-sensitivity — the
     nested-CV **headline must be ~unchanged** whether selection ranges over `[0,PSP]`, `[0,2·PSP]`,
     or `[0,last]`. (b) is the master defense: PSP is measured on the SELECT fold and selection
     also uses SELECT, so if the headline depended on the window that would be circular; showing
     it does *not* depend on the window neutralizes the concern and renders the detector
     parameters non-load-bearing. When in doubt, bound the window **generously** — a wider window
     only adds candidates (more chances for a fluke *against* the headline), so it is conservative.
- **Baselines on the identical folds:** `baselines_nested.py` computes DTW and the corrected
  geodesic on the same 25 test folds. The geodesic is the **DTW-aligned, unpadded**
  `compute_geodesic_distances_dtw` (NOT the legacy frame-padded one — see §5).
- **Perceptual target:** `d_perc = 1 − count_normalized[(selected0=0, selected1=2)]` per rated
  pair; correlation is Spearman of pairwise L2 distance vs. `d_perc` (plain L2, no
  neutral-centering), over the test-fold pairs that carry a human rating.

---

## 3. Environment and dependencies (chimera)

These scripts run on the **chimera** cluster (UMass Boston), where the checkpoints, encode
scripts, and BVH/loader code live. They use **absolute chimera paths** by design:

- `MS_ROOT = .../virtual_reality/motion-similarity` — the python package (Config, loaders,
  `networks/similarity_network.py`, `embedding_dataset.py`).
- `TR_ROOT = .../virtual_reality/triplets` — checkpoints, encode scripts
  (`MAMP/encode_mamp_lma.py`, `probes/encode_aetransformer.py`, `probes/encode_sweep.py`),
  and the BVH/common/bvhReader code.
- conda env **`torch_gpu_cu12`** (has torch+CUDA; does **not** have tensorflow — the harness
  avoids the TF-importing modules deliberately).
- env vars: `MOTION_IS_REMOTE=1` (selects chimera data paths in `Config`), `PYTHONWARNINGS=ignore`.
- GPU: runs on the **pomplun** partition (H200, account `cs_funda.durupinarbabur`) or
  **impact/DGXA100** (A100). Both used; results are partition-independent.

> To run elsewhere, repoint `MS_ROOT`/`TR_ROOT` at the top of each script and ensure the
> checkpoints, encode scripts, norm_stats, and rated BVH clips are present.

---

## 4. Exact reproduction commands

**Per-encoder raw nested-CV** (one sbatch per encoder; matched 7-candidate sets):

```bash
# MAMP+pose (headline) and motion-only MAMP — authors' 400-epoch budget region
python cv_nested.py --encoders MAMP+pose --k 5 --repeats 5 --n-triplet-seeds 5 --triplet-epochs 200 \
    --ckpt-epochs 200 300 400 500 600 700 800

python cv_nested.py --encoders MAMP --k 5 --repeats 5 --n-triplet-seeds 5 --triplet-epochs 200 \
    --ckpt-epochs 200 300 400 500 600 700 800

# vanilla plain-transformer AEs (converge ~8–10k epochs)
python cv_nested.py --encoders AE-vel   --ckpt-epochs 100 1700 3400 5100 6700 8300 10000  --k 5 --repeats 5 ...
python cv_nested.py --encoders AE-recon --ckpt-epochs 100 1400 2700 4100 5400 6700 8000   --k 5 --repeats 5 ...

# MLD-AE family (converge ~6.7k epochs)
for e in MLD-AE-holdout MLD-AE-vel MLD-AE-prov; do
  python cv_nested.py --encoders $e --ckpt-epochs 100 1200 2300 3400 4500 5600 6700 --k 5 --repeats 5 ...
done

# MLD-VAE (deterministic mu encoding) — trajectory built with the patched
# train_vae_mld.py --keep-all-checkpoints; candidate epochs span its trajectory.
```

**Baselines on the identical folds:**
```bash
python baselines_nested.py --k 5 --repeats 5
```

**Perceptual fine-tune (§6.3), variants A and B:**
```bash
python perc_finetune.py --variants A B --k 5 --repeats 5 --epochs 60 --lr-A 0.01 --lr-B 0.00001
```

Each writes a JSON results file and prints a per-fold log plus the final AGGREGATE
(mean ± std, SEM, and the 95%-CI-clears-bar verdict per action).

---

## 5. Critical correctness notes (pitfalls that cost real debugging)

1. **Single-split estimates are unreliable.** Pointing-raw Spearman has fold-std ≈ 0.15; any
   one split is uninformative. Earlier single-split numbers (raw pointing +0.263, and a
   "+triplet lifts pointing" result) were partition artifacts. Use the repeated CV mean.
2. **Checkpoint identity.** MAMP+pose appears under two budgets: the headline encoder is
   **epoch-400** (authors' budget); a suffix-less embedding dir in the archive is the
   **converged epoch-1199** checkpoint — a different representation (cosine ≈ 0.67). Confirm the
   epoch (and md5) before comparing embeddings.
3. **The geodesic baseline must be the corrected one.** `compute_geodesic_distances_dtw`
   (DTW-aligned, raw unpadded, `num_joints=28`) — NOT the legacy `compute_geodesic_distances`
   that padded clips to 137 frames (which deflated short pointing clips). The corrected function
   lives in `pipelines/infer_emb_vec.py`; `baselines_nested.py` imports that copy explicitly
   (there is a stale duplicate at the repo root — do not import it).
4. **Code/version parity.** The metric MLP must be the LayerNorm version; the embedding loader
   must support single-vector `_emb.pt` (`single_embedding_mode`); both must be the versions on
   the execution host, not stale copies.
5. **Encode dispatch by family** (`cv_preliminary.encode`): MAMP via `encode_mamp_lma.py`
   (ckpt/config paths relative to `MAMP/`, action **capitalized**, needs `--mem 32G`); vanilla
   via `encode_aetransformer.py --n-windows 1`; MLD AE/VAE via `encode_sweep.py --kind ae|vae
   --n-windows 1`. Checkpoint epoch is parsed as the **trailing integer** of the filename
   (vanilla files are `cp_PROV_noVel_<epoch>.pth`, MLD are `epoch_<epoch>.pt`).

---

## 6. Final results (for verification)

Raw Spearman, repeated nested-CV (n = 25), mean ± std. Bars to clear: walking > 0.478,
pointing > 0.370, picking > 0.431 (the stronger of DTW / geodesic per action, on the same folds).

| encoder | walking | pointing | picking | clears all 3 (95% CI)? |
|---------|---------|----------|---------|:---:|
| Plain-transformer recon (AE-recon) | 0.483 ± 0.127 | 0.206 ± 0.178 | 0.372 ± 0.153 | no |
| Plain-transformer +vel (AE-vel-plain) | 0.451 ± 0.133 | 0.213 ± 0.177 | 0.323 ± 0.137 | no |
| MLD-AE recon (holdout) | 0.526 ± 0.117 | 0.326 ± 0.195 | 0.493 ± 0.143 | no (pointing) |
| MLD-AE +vel | 0.561 ± 0.110 | 0.347 ± 0.174 | 0.497 ± 0.153 | no (pointing) |
| MLD-AE rotMSE+vel (prov) | 0.544 ± 0.122 | 0.356 ± 0.179 | 0.488 ± 0.146 | no (pointing) |
| MLD-VAE (holdout) | *pending re-trained trajectory* | | | |
| motion-only MAMP | 0.356 ± 0.151 | 0.285 ± 0.160 | 0.501 ± 0.194 | no |
| **MAMP+pose** | **0.559 ± 0.126** | **0.429 ± 0.147** | **0.543 ± 0.170** | **yes** |

Baselines on the same folds: DTW 0.490 / 0.273 / 0.303; geodesic 0.371 / 0.234 / 0.264.

Perceptual fine-tune of MAMP+pose (§6.3): raw 0.559/0.429/0.543 → variant A (pool)
0.614/0.421/0.595 → **variant B (full encoder) 0.666/0.443/0.598** (only B clears all three with
the 95% CI excluding the bar).

**Headline:** MAMP+pose is the only encoder whose 95% CI clears all three baseline bars;
pointing is the discriminator. Perceptual fine-tuning of the encoder improves all three further.
