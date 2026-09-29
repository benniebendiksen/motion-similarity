# Experiment Protocols — the ONE canonical reference

**Purpose:** a single, authoritative place for HOW we evaluate, so every result is comparable and reproducible,
and so protocol drift is caught. **Revisit + update this each experimental round.** If a run deviates, say so
explicitly and record why. Companions: `PIPELINE_TRACKER.md` (status), `MECHANISM_FINDINGS.md` (results).

Provenance of the perceptual protocol: **the PAPER is canonical** — supplement `\section{Checkpoint-Selection
Protocol}` (`app:checkpoint`) in `docs/overleaf/supplement.tex`; mirrored in `docs/specs/HYBRID_…SPEC.md`
(REPRODUCE-EXACTLY); IMPLEMENTED in `motion-similarity/cv_nested.py` + `cv_preliminary.py`.

---

## 1. Perceptual alignment (LMA ρ) — nested cross-validation + SELECT
The headline metric: raw **Spearman ρ** between encoder embedding-distances and human `d_perc`, per action.

- **Harness:** `cv_nested.py --encoders <ENC> --k 5 --repeats 5` (RAW). Optional triplet arm: `--n-triplet-seeds 5
  --triplet-epochs 200` (see §1a). **n = 25 = 5 folds × 5 repeats.**
- **Per outer fold:** 3-way disjoint **TEST / SELECT / TRAIN**. Folds shuffle only the 57 Effort classes/action.
  Seeds stratified `42 + 1000·repeat`.
- **⭐ SELECT (the perceptual-relevance step) — WE DO APPLY IT:** for each fold, the evaluated checkpoint is the
  one, among the grid of **every 100th epoch within `[100, PLATEAU]`**, that **MAXIMIZES mean raw-Spearman across
  the 3 actions on the SELECT fold**. (Action-agnostic; SELECT used ONCE, for the perceptual pick.) Reported ρ =
  the mean over the 25 TEST estimates using each fold's SELECT-chosen checkpoint.
- **PLATEAU (the grid UPPER BOUND) = each encoder's OWN reconstruction-loss plateau** — `val_recon` (MSE in
  normalized space, 5% held-out split, seed 0), NEVER perceptual (so SELECT isn't double-used). Operational def:
  3-pt smoothed `val_recon`; asymptote = min over final 20%; PLATEAU = first grid-100 epoch where smoothed ≤1.02×
  asymptote and stays ≤1.03× thereafter. **Masked-prediction family (incl. the hybrid) can DIVERGE late ⇒ PLATEAU
  = the PRE-DIVERGENCE epoch.**
- **⚠ "identical protocol" means same GRID BOUND.** SELECT is always applied; but the candidate SET it chooses
  from = `[100, PLATEAU]`, so two encoders are only comparable if run with the **same `--loss-plateau` (and
  `--epoch-grid`)**. Comparing a new run to a STORED number computed under a different PLATEAU is the confound to
  avoid — re-run the comparator under the same bound. (Empirical check 2026-09-22: base MAMP+pose-cut1 under
  `[100,1199]` reproduced its stored .462/.302/.481 exactly ⇒ SELECT is robust to a too-wide window here.)
- **RESOLVED for the hybrid (2026-09-22):** combined training loss (masked+pose+0.1·contrastive) is monotone-
  flattening to ep1199 with **NO late divergence** (final-20% min 0.4871 = last 0.4873, max 0.4936). Operational def
  on this trajectory ⇒ **PLATEAU ≈ 800** (ep800=0.4968=1.02×asymptote, stays below after; ep700=0.4990 fails). Since
  there is no divergence, the eval's [100,1199] window is a benign superset of [100,800] — same case as the base
  empirical check. Exact per-checkpoint `val_recon` (decode on 5% held-out) is DEFERRED as unnecessary given
  no-divergence + contrastive weight only 0.1. **Per-fold `selected_epoch` IS recorded** in `cv_nested_*.json`
  (`folds[i].selected_epoch`): hybrid median **900** (modal 13/25), base median **700** (modal 9/25) — SELECT did
  NOT drift to the 1199 budget, so the LMA-ρ numbers are genuine SELECT picks near the recon plateau.

### 1a. Triplet fine-tune arm (`trip_test`)
`cv_nested` also runs a per-fold **perceptual triplet fine-tune** (a small triplet-metric net trained on the
TRAIN fold's rankings, 5 seeds × 200 epochs, no early-stop on test). It reports `trip_test` alongside `raw_test`.
**We compare RAW encoder ρ (`raw_test`) as the headline;** the triplet arm is secondary (it has consistently
*degraded* the raw encoder in our runs — see [[project_nested_cv_final]]). It is slow (dominates wall-time); when
only RAW is needed, it can be minimized/skipped in future runs.

## 2. SNP@k — Semantic Neighborhood Precision (post-paper eval-suite metric)
Measures whether the MOTION manifold is semantically organized (manifold-quality proxy; `probes/snp_eval.py`).
- For a set of captioned clips, encode each → motion embedding (`forward_encoder` mask_ratio=0 → **mean-pool**,
  the SAME embedding as §1). For each clip: its top-k nearest by **motion**-embedding cosine (`motionKNN_k`) and
  top-k nearest by **caption** (frozen MPNet) cosine (`semKNN_k`).
- **SNP@k = mean over clips of `|motionKNN_k ∩ semKNN_k| / k`** (k=5,10). Exclude near-duplicate captions
  (cos>0.95, TMR's `threshold_selfsim_metrics`) from the semantic neighbors.
- **Companion: dist-Spearman** = Spearman between pairwise motion-distances and pairwise caption-distances
  (random pair sample). Report **base-vs-hybrid Δ**, same clips / same ckpt / same metric (no protocol confound).
- CAVEAT: captions = action semantics ⇒ SEMANTIC (coarse) proxy for perceptual, complements §1 (LMA ρ), not a
  substitute. In-sample for a pilot whose contrastive saw all captioned clips; reserve a held-out captioned split
  for a generalization number.

## 3. A′ re-probe (mechanism)
Rerun the Phase-1 mechanism probes on a TRAINED encoder to see HOW its representation reshaped:
**`probes/peakedness_v2_param.py`** (parametrized: `--code-dir --config --ckpt --norm-stats --out`; temporal/joint
concentration, subset-only@{10,25,50}%) + decodability. ⚠ pass an **ABSOLUTE `--out`** (relative write fails —
process CWD ≠ triplets). Base A1 lives in `probes/peakedness_v2_mamp_pose_cut1.json`. **Selectivity(fr) = rho(top-k
salient slices only) − rho(bottom-k only)**; positive ⇒ that axis' peaks carry the geometry. **Diagnostic criterion
for the H1/H2 fixes:** a good fix should RESTORE base's positive temporal-localization (esp. picking Tsel@10 +.138)
and pointing joint-selectivity toward base — NOT merely recover overall rho (see MECHANISM_FINDINGS A′ result).

## 4. Encode & environment
- Encoders registered in `cv_preliminary.ENCODERS` (family/ckpt_dir/config/code_dir/norm_stats/pool). mamp family
  encodes via `{code_dir}/encode_mamp_lma.py --norm-stats <ns> --pool mean`. cut1 models use `norm_stats_cut1.npz`.
- Env (matched to paper): Rocky Linux 8.10, A100(80GB)/H200(141GB), PyTorch 2.4.1 / CUDA 12.1 / Python 3.10;
  our env `torch_gpu_cu12`. GPU jobs on pomplun (`cs_funda.durupinarbabur`, dodges the impact cap); impact
  aicore/h200 = fallback.

---
_Last revised: 2026-09-22 (pilot round). NEXT-ROUND CHECKLIST: (a) confirm hybrid PLATEAU from its val_recon;
(b) any new metric added here first; (c) note every deviation with a reason._
