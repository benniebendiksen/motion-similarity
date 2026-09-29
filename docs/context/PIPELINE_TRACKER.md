# Pipeline Tracker — four-phase program

**Status board** for the post-AAAI research program. This file = *what's done / what's next*.
It cross-links; it does not duplicate. The three companions:
- **`docs/RESEARCH_ROADMAP.md`** — the WHY (vision, mechanism thesis, Job DAG).
- **`docs/provenance_ledger.md`** — the PROOF (every number → job → config → checkpoint).
- **`docs/specs/HYBRID_mamppose_tmr_SPEC.md`** — the HOW (hybrid build detail).

Thesis under test: **perception ≠ faithful representation** — the encoder that best predicts
human similarity is NOT the one that most faithfully reconstructs/decodes the motion.

Legend: ✅ done · 🔵 in progress · ⬜ queued · ⛔ blocked · ⭐ decision locked

---

## Phase 1 — UNDERSTAND (mechanism probes; no training)
Goal: localize *where/what* carries perceptual signal, and show recon-fidelity ≠ perception
across the discriminative encoders. Cheap, runs in parallel with Phase 2.

- ✅🔵 **A1 Peakedness** (selective-vs-generalized), MAMP+pose — `probes/peakedness_mamp_v2.py`
  - v1 (ablate-from-mean) = NULL, but that test is washout-confounded (mean-pool robust to frame drop).
  - v2 (SUBSET-ONLY: rho(top-k only) vs rho(bottom-k only)) at 10% subset → **RESULT 2026-08-27:**
    - TEMPORAL concentration REAL, action-graded **picking>pointing>walking**: picking top-10%=0.473
      (=full 0.472) vs bot=0.336; pointing top=0.223 vs bot=0.166; walking top=0.302<full=0.406 (distributed).
    - JOINT concentration = NONE: selectivity ~0/negative all actions; joint-saliency FLAT (max/mean~1.4)
      and ACTION-INVARIANT (same core/root joints idx 2,8,19,7,3 top every action). ⇒ no "perceptual joint."
  - IMPLICATION: perceptual signal is TEMPORALLY localizable but JOINT-distributed. Hardens entanglement
    caveat; VALIDATES Shapley reframing (per-part color = per-comparison share, not intrinsic specialization).
  - ⬜ CAVEAT: ckpt-1199 may be post-divergence (pointing rho_all=0.216 < nested-CV 0.302); re-confirm at
    pre-divergence plateau ckpt (concentration structure is relative → expected robust). result json:
    `probes/peakedness_v2_mamp_pose_cut1.json`.
  - ⬜ TODO: extend to MLD family via INPUT-space ablation (latent-query pool, not mean).
- ✅ **A2 Decodability** — COMPLETE & PUBLISHED (main §8 + supplement app:probe). AUTHORITATIVE =
  v1 probe (`localization_probe.py` → `probe_*.json`), full-corpus: MAMP+pose ρ=.434/linR²=.509/
  mlpR²=.645; vanilla+attn ρ=.213/.727/.829; others between. Nonlinear MLP DONE (preserves ordering
  ⇒ pose present-but-less-linearly-accessible). vel/acc R²<0. Pointing-only. `probe2_*.json` (v2,
  SD(R²)-primary, pose-PCA=2) is SECONDARY/exploratory — NOT the paper numbers. → `MECHANISM_FINDINGS.md`.
  - (optional, non-blocking) revisit v2 SD(R²) as its own lens; extend to uencfull. Not needed for paper.
- ✅ Capacity dissociation (35M reconstructs best, perceives worst) — ledger.
- ✅ U-Net skip "relief valve" (skips help recon, neutralize the pose head) — ledger.

## Phase 2 — IMPROVE (hybrid training; the long pole)
Goal: RAISE perceptual ρ by adding semantic (caption) supervision to masking+pose, WITHOUT
moving motion out of our representation.

- ⭐ **Representation decision (LOCKED):** motion stays in our CMU-33 `(T,34,6)`; masking well-posed.
  TMR's 263-dim vector REJECTED (masking ill-posed/leaky there). Import LABELS, not format.
- ⭐ **Corpus decision (LOCKED, REVISED 2026-08-27):** masking+pose on ALL 14,143 cut1 (self-supervised,
  no captions needed, REUSE existing base — NO retrain); contrastive on the 10,174 EXACT-matched captioned
  subset. Fuzzy 91.7% was collision-inflated; EXACT = 10,174/14,143 = 71.9% (cut1 over-sampled AMASS
  subsets HML3D annotated sparsely — BioMotionLab cut1 3061 vs HML3D ~373). DEFER HumanAct12 (+1,191, needs
  NEW tooling + distribution-shift risk) and full-HML3D (14,616 vs our staged 11,713) behind an EVIDENCE
  GATE (only if contrastive proves caption-starved). Corpus size = 2nd-order; objective design = 1st-order.
- ⭐ **Grid (LOCKED):** {contrastive × KL} 2×2 on masking+pose base. Hypothesis:
  contrastive→pointing · KL→walking · masking+pose→picking. (This is where VAE/KL returns as a
  controlled arm — not the discarded apples-to-apples cut.)
- 🔵 **Grid DESIGN REVISED IN DISCUSSION (2026-09, not yet re-locked — needs user GO):**
  The 2×2 was right in SPIRIT (two levers) but the KL was bolted in the WRONG place. CORRECTED:
  - **Contrastive lever — TEXT SIDE REVISED 2026-09-19 (code-verified TMR, [[reference_tmr_architecture]]):**
    frozen-MPNet-SENTENCE target is WRONG/superseded (rigid SBERT hypersphere hostile to a motion-joint warp;
    pooled 1-D kills sequence). ADOPT TMR's recipe = frozen `distilbert-base-uncased` TOKEN embeddings
    (`save_token_embeddings`) → LIGHT trainable **non-VAE** `ACTORStyleEncoder` head → text latent; contrast
    motion-latent vs it. NON-VAE deliberate: VAE/KL lives on MOTION (variational-MAMP); a text-latent KL would be
    footless here (no recon through it). τ=0.7; FNF on EVERY arm (separate frozen MPNet caption-caption cos>0.6
    mask); keep head light (~30k caps<45k → overfit guard). Frozen-sentence = optional ablation only. `g_θ`
    superseded (it was a minimal echo of exactly this).
  - **⭐ KL lever RELOCATED — the VAE-did-well correction (user-caught):** MLD-VAE genuinely beat MLD-AE
    (walking 0.597 best-in-table; carried AE on CONTROL grounds not perf — [[project_mld_ae_vs_vae_justification]]).
    My "KL footless" claim held ONLY for pooled-KL-on-side-readout (arm-1a trap). CORRECT spot = **variational
    MAMP**: μ,σ,sample on the PER-PATCH latents between encoder↔decoder, small β·KL, masked-pred+pose flows
    THROUGH z. Regularizes the perception-carrying bottleneck via the perception-WINNING objective (masked-pred,
    NOT faithful full-recon) ⇒ avoids arm-1a. Hypothesis: helps WALKING (parallels MLD-VAE), ~inert pointing.
    Option A = per-patch (simple, distributed bottleneck); Option D (reserve) = tight pooled global-context
    bottleneck the masked-decoder also predicts from. Eval deterministic (use μ).
  - ⇒ **Corrected 2×2 = {contrastive × variational-MAMP}**, both respecting the masked backbone. Primary single
    lever to launch first = **+contrastive** (grid-1). See webpage §09.
- ✅ cut1 construction provenance nailed (ledger, reviewer-defense).
- ✅ Exact-matcher `cut1_exact_match.py` → **10,174 captioned, 30,417 captions (2.99/clip)**; manifest
  `datasets/cut1_hml_captioned.json`. Validated (BioMotionLab confirms genuine absences, not naming bug).
- 🔵 Precompute frozen all-mpnet-base-v2 sentence embeddings (transformers mean-pool+norm, tmr env):
  `probes/text_precompute.py` → `datasets/cut1_hml_text_mpnet.pt` = {stem: [n_caps, 768]}. RUNNING.
- ✅ Contrastive (InfoNCE) + KL heads built in isolated copy `MAMP_hybrid/` (feeder+engine+model):
  full-vis pooled emb → `text_proj` 256→768, symmetric CLIP-style InfoNCE (τ=0.1) over captioned
  subset of batch; optional VAE-KL bottleneck (to_mu/to_logvar+reparam). Config-driven (model_args +
  train_feeder_args.text_emb_path), NO main.py changes. Feeder samples 1 caption/clip/epoch, aligns
  by basename, has_text=0 for uncaptioned. TODO: TMR-style false-neg filter (text-text cos>0.8).
- ✅ MODEL SMOKE PASSED (synthetic, all 4 arms): losses finite; grads flow to text_proj (contrastive)
  and to_mu (KL) per arm; baseline untouched. Grid configs pushed: `lma_hybrid_{C1K0,C0K1,C1K1}_cut1.yaml`
  (cw=0.1, kw=1e-3; baseline C0K0 = existing mamp_pose). NOTE weights are starting points (may tune).
- ✅ **RE-ARCH COMPLETE 2026-09-19 (TMR-recipe text side; supersedes the MPNet-sentence + text_proj + pooled-KL
  build above).** In `MAMP_hybrid`: (1) frozen distilbert TOKEN embeddings precomputed → `cut1_hml_text_distilbert_tokens.pt`
  (27,909 caps, 448,935 tokens, maxlen 64, ~1GB); (2) `model_mamp/text_head.py` = light non-VAE `ActorTextHead`
  (2-layer, 768→256, learnable summary token) — faithful ACTOR port; (3) feeder serves sampled-caption token
  seq (pad-64) + mask + that caption's mpnet vec (FNF) + has_text; (4) engine 5-tuple; (5) model: motion_proj
  (256→256) + ActorTextHead → symmetric InfoNCE (τ=0.7) + FNF (mpnet caption-caption cos>0.6). MPNet = filter only.
  SMOKES PASSED: synthetic model (loss finite, grads flow to motion_proj+text_head, baseline untouched, FNF ok);
  text-alignment (caption→tokens, sent-bank aligned to manifest order, 0 missing/500). Config `lma_hybrid_C1K0_cut1.yaml`
  updated (contrastive-only pilot). (kl_weight reserved for variational-MAMP; C0K1/C1K1 configs stale = old pooled-KL.)
- ✅ Full-run REAL-FEEDER smoke PASSED (2026-09-19): feeder 14,140 clips / tok_bank 27,909 / sent_bank 10,174;
  batches captioned ~7/8; loss finite; grads flow to motion_proj + ACTOR text_head. Engine 5-tuple contract OK.
- 🔵 **PILOT RUNNING: job 1020752, pomplun chimera21, NVIDIA H200 143GB.** `training/hybrid_C1K0_pomplun.sbatch`
  → `MAMP_hybrid/output_dir/lma_hybrid_C1K0`. Contrastive-only (τ=0.7, FNF); masking+pose on 14,143 / contrastive
  on 10,174. Progress 2026-09-20: ~ep1060/1200, loss 2.15→~0.49 clean (no NaN), ~3h to finish. Fallback if
  pomplun full = impact aicore/h200 (pomplun preferred — dodges billing cap).
- **EVAL JOBS (built; run ON pilot completion — placeholders):**
  - ⬜ **LMA ρ (nested-CV):** `MAMP-hybrid` registered in `cv_preliminary.ENCODERS` (`.bak_hybrid`); run
    `cv_nested.py --encoders MAMP-hybrid --k 5 --repeats 5`. RESULT: walk __ / point __ / pick __ (vs base .462/.302/.481).
  - ⬜ **SNP@k:** `probes/snp_eval.py` (base/hybrid via --code-dir/--config/--ckpt); evaluator smoke-tested on base.
    RESULT: SNP@5 base __ / hybrid __ (Δ __); SNP@10 __; dist-Spearman __.
  - ⬜ **A′ re-probe:** rerun `peakedness_mamp_v2.py` + decodability on the hybrid ckpt. RESULT: temporal/joint shift __.
- ⬜ (was: train the grid — now unblocked, pending launch)
- ⬜ **Eval suite** (pilot): (1) LMA ρ via EXACT nested-CV (k5×repeats5, TEST/SELECT/TRAIN, plateau-on-recon)
  = GUARDRAIL + weak proxy (compare base .462/.302/.481; big drop=over-collapse warning, gain=bonus, flat≠fail);
  (2) **SNP@k** = Semantic Neighborhood Precision — for a held-out captioned split, mean |motionKNN_k ∩ semKNN_k|/k
  (MPNet caption oracle; exclude near-dup captions cos>0.95 = TMR threshold_selfsim_metrics); companion =
  Spearman(motion-dist, caption-dist); + text→motion R@k (TMR-comparable). Report baseline-vs-hybrid Δ. Measures
  motion-manifold semantic organization (coarse proxy for perceptual; complements LMA ρ). (3) A′ re-probe.
- ⬜ **A′ re-probe** the trained hybrid (rerun Phase-1 probes on it → *why* it works).

## Phase 3 — VALIDATE (new 20-clip triplet study)
Goal: enrich/validate the whole-body distance D and the per-part attribution with fresh human data.

- ⭐ **Reframe (LOCKED):** per-part = Shapley SHARE of validated whole-body distance D,
  NOT a per-part perceptual distance. Study ENRICHES D; doesn't need to prove per-part perception.
- ✅ Budget analysis: 20 clips → validate ONE part deeply (~190 pairs) OR broad D-coverage.
- ⬜ Selection: construct stimuli from RAW per-part kinematics (non-circular); model only validates.
- ⛔ Gated on Phase 2 (wants the improved D first). Phase-1 joint-profile seeds *which* part to probe.

## Phase 4 — DEPLOY (web-based per-part motion guidance)
Goal: camera motion + reference (YouTube) pose → whole-body D + per-part color feedback, real-time.

- ✅ Math in hand: φ_P = ⟨u_P, Σ_Q u_Q⟩ (squared-dist + additive mean-pool sub-embeddings);
  efficiency Σφ_P = D exact; pairwise coupling ⟨u_P,u_Q⟩.
- ✅ Caveat carried: z_P is arm-*located* not arm-*exclusive* (global attention) — OK for
  attribution/coloring, NOT isolation-search.
- ⬜ Pose-extraction front end (camera + YouTube) → encoder → φ_P overlay. (post Phase 2–3)

---
_Last touched: 2026-09-19 — Phase 2 contrastive RE-ARCHITECTED to TMR recipe (frozen distilbert tokens → light
non-VAE ACTOR head; MPNet→filter-only) after code-verifying TMR ([[reference_tmr_architecture]]). Data + model +
feeder + engine + config all built; synthetic + text-alignment smokes PASSED. Launch-ready — GATED on the
grid-launch discussion/GO. Next: full-run (real-feeder) smoke → sbatch → contrastive-only pilot; variational-MAMP
(per-patch KL) is the second lever, still to build._
