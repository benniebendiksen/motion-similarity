# Perceptual Motion Alignment — Research Roadmap & Program Manifest

**Purpose (anti-drift):** the single forward-looking north star for the *per-body-part perceptual alignment tool*
program that follows the AAAI-2027 paper. Read this at the start of any working session on this direction.
Backward-looking paper provenance = `docs/REPRO_TRUTH.md`; durable facts = the memory dir (`MEMORY.md` index).
Live status board = `docs/context/PIPELINE_TRACKER.md`; hybrid build detail = `docs/specs/HYBRID_mamppose_tmr_SPEC.md`.
Last major update: 2026-09-19.

---

## 0. CURRENT STATE (2026-09-19) — Phase 2 hybrid built & launch-ready
- **Objective (re-anchored):** raise the GENERAL perceptual relevance of the manifold (for downstream Phase-3
  sample-selection + verification), NOT the LMA-effort ranking. LMA effort tuples = the current eval's labels,
  not the objective; LMA ρ is a GUARDRAIL + weak proxy, not the arbiter.
- **Hybrid = MAMP masked backbone + a semantic contrastive scaffold** (NOT TMR-minus-recon; we dropped
  reconstruction per Dissociation 2, which un-couples TMR's single-latent design — see webpage §09).
- **TMR arch CODE-VERIFIED** ([[reference_tmr_architecture]]): text = FROZEN distilbert-base-uncased →
  TRAINABLE ACTORStyleEncoder head; all-mpnet-base-v2 FROZEN for the false-neg filter ONLY; loss =
  contrastive(τ=0.7, filtered) + recons(SmoothL1, cross-modal) + KL(4 terms) + latent(L_E). Settled a
  disagreement: NOT distilroberta, NOT a trainable LM.
- **Contrastive lever (BUILT, smoke-passed, launch-ready — gated on grid-launch GO):** adopt TMR's text recipe
  — frozen distilbert TOKEN embeddings → LIGHT non-VAE ACTOR head → text latent; motion pooled → proj; symmetric
  InfoNCE (τ=0.7) + FNF (mpnet caption-caption cos>0.6). MPNet=filter only. `MAMP_hybrid`, config
  `lma_hybrid_C1K0_cut1.yaml`. Frozen-MPNet-SENTENCE target REJECTED (rigid SBERT hypersphere hostile to a
  motion-joint warp).
- **Variational lever (2nd, TO BUILD after pilot):** variational-MAMP = per-patch μ,σ,KL on the masked-pred
  bottleneck (where KL has footing; pooled-KL was footless = arm-1a trap). Motivated by MLD-VAE's walking win.
- **Eval suite:** LMA ρ (nested-CV, guardrail) + **SNP@k** (Semantic Neighborhood Precision — motion-motion
  kNN vs caption-kNN on a held-out captioned split, MPNet oracle, exclude cos>0.95; the motion-manifold cousin
  of TMR's text↔motion R@k; companion = motion-dist↔caption-dist Spearman) + A′ re-probe + Phase-3 (true verdict).
- **Grid:** {contrastive × variational-MAMP}; LAUNCH ORDER = contrastive-only pilot first (from scratch, 1200 ep,
  ≈1.5× base cost, pomplun GPU).

## 1. The vision (what we're building)
A web tool for **real-time perceptual alignment** between a user's camera-recorded motion and pose sequences
extracted from YouTube single-person motion. Two feedback levels: (a) a **whole-body gestalt** alignment
indicator, and (b) **per-body-part** guidance (color-coding) at a finer grain. Pose extraction + skeleton
rendering (user cam + YouTube) already work (from a sibling project).

## 2. What grounds it (established findings — compressed; detail in REPRO_TRUTH + memories)
The paper + chimera jobs established **how perceptual alignment arises in a motion embedding**:
- **Perception ≠ faithful representation.** Alignment is NOT bought by capacity, skip-plumbing, reconstruction
  fidelity, or linear pose-accessibility (three dissociations: capacity 35M reconstructs best/perceives worst;
  decodability inverse on pointing ρ.434/R².509 vs ρ.213/R².727; U-Net skips help recon but neutralize the
  pose head). It IS produced by the **training objective** (masking + pose). See [[project_localization_probe]],
  [[project_swap_pooling]], REPRO_TRUTH §26–29.
- **Localized vs distributed axis.** Pointing = localized (few apex frames / upper body); walking = distributed
  /periodic; picking = intermediate. This axis predicts pooling & per-part effects.
- **Pooling is an interaction, not a main effect** (attn-pool wins only in MLD); MAMP+pose is **readout-robust**
  (mean .434 ≈ max .453 ≈ perc-pool .424 on pointing) → its signal is pooling-robust ("spread" route). Job-1
  results REPRO_TRUTH §29.2.
- **TMR is a reconstruction VAE + light contrastive**, not purely semantic — both it and MAMP+pose share a
  reconstruction base; the differentiator is **masking vs text**. [[project_tmr_reconstruction_framing]].

## 3. ⭐ THE REFRAMING that de-risks the tool (2026-08-27) — Shapley = contribution, not per-part distance
**Per-part color-coding does NOT require a validated per-part perceptual *distance*.** It requires only a valid
*decomposition of the whole-body distance* into per-part **contributions**:
- Shapley attribution: `D(U,R) = Σ_P φ_P` (efficiency axiom, exact). φ_P = **the part's share of the
  perceptually-validated whole-body mismatch** = "how much modifying this part would improve whole-body alignment."
- Grounded in only two things WE ALREADY HAVE: (1) whole-body D ≈ human d_perc (the paper); (2) Shapley
  efficiency (a math identity). **No per-part human labels needed to deploy the colors honestly.**
- **Color by Shapley** (clean decomposition summing to D); **guidance = the ranking** ("focus on your arm").
- GAINS: tool deploys on whole-body validation alone; **entanglement stops mattering** (located-not-exclusive is
  fine for a contribution); **decouples the tool from the per-part-perception *science*** (§9).
- GIVES UP (honestly): not an absolute per-part perceptual *scale* — a *relative contribution* (the right
  construct for guidance). Caveats: real-arm modification ≈ model z_arm move (entanglement, approximate); the
  tool's eventual empirical check becomes an **efficacy** study ("do users improve following guidance?"), not a
  per-part-perception study.

## 4. Next model — Hybrid MAMP+pose × TMR (masking + pose + reconstruction + semantic-contrastive)
Goal: a top-performing whole-body encoder (potentially first to clear all 3 actions **at matched corpus**) whose
better **D** improves both the Shapley contributions AND the sample-selection manifold. **Full spec (migrated,
version-controlled):** `docs/specs/HYBRID_mamppose_tmr_SPEC.md`. Recommended = Option A (graft frozen-text
InfoNCE onto our MAMP+pose, our pipeline).
- **⭐ RUN AT MATCHED CORPUS (cut1), not full-corpus** — for TWO reasons: (a) fair to TMR; (b) the full corpus
  (5,453) contains 91 effort-stylized LMA clips ⇒ a pretraining↔eval distribution *home-field* prior (cut1 has
  MORE clips, 14,143 AMASS, yet WORSE pointing .302<.434 ⇒ full-corpus's edge is effort familiarity, NOT more
  data; not leakage, but a confound). Clean base = **cut1 MAMP+pose (.462/.302/.481)**; TMR anchor
  **(.452/.425/.408)**; test = does +contrastive close pointing .302→.425 w/o losing pick.
- **Load-bearing caveat:** TMR's semantic axis is action-level but our task is style-within-action → hypothesis
  is *representational* (semantic pretraining shapes gesture priors), not discriminative; a null result is still
  publishable (routes not composable).
- **Reproduce the exact nested-CV protocol** (supplement §app:checkpoint; pinned in the spec) so numbers stay
  comparable — the hybrid is a masked-prediction-family model ⇒ PLATEAU = pre-divergence epoch.
- Blocked on chimera reachable + §8 decisions (incl. verifying cut1 carries HumanML3D captions).

## 5. ⭐ Next user study — 20-clip broad-coverage to ENRICH D (the study-design note, reframed 2026-08-27)
**Purpose:** enrich/validate the **whole-body** perceptual embedding D — NOT per-part validation (that moved to
the optional deeper track §9, per the §3 reframing).
- **Budget:** 20 clips → C(20,3)=1,140 triples, C(20,2)=**190 pairs**, each pair in 18 triples → a well-sampled
  **20×20 whole-body dissimilarity matrix**. (20 chosen to keep the full triplet set rateable.)
- **Break from LMA:** no effort factors; no within-action-only constraint (cross-action triplets allowed).
- **Selection = broad, perceptually-meaningful manifold coverage** using our **best perceptual encoder**
  (MAMP+pose now; the hybrid once trained) — e.g. farthest-point / max-spread sampling in the *embedding* so the
  190 pairs span a wide perceptual range. **NOT the reconstruction-VAE manifold** (selects reconstruction-diverse
  ≠ perceptually-diverse — capacity/decodability lesson). **NOT per-part-isolating** (that was the old framing).
- **Output → role 3:** the 20×20 human d_perc matrix fine-tunes/validates the embedding → better D → **better
  Shapley contributions for free** (the tool's colors ride the improved D, no per-part labels).
- **Per-part role of this study: none directly.** Per-part attribution is downstream of the improved D via
  Shapley (§3, §6). This study's job is purely whole-body enrichment.
- **Why 20 can't do per-part (for the record):** 20/5 ≈ 4 clips/part ≈ 6 pairs/part = too thin; a rigorous 5-part
  study needs ≈25–30+ clips; a single-part (arm/gesture) pilot could use all 20 (190 arm-pairs) IF we ever want
  it — but under §3 we don't need it for the tool. See §9.
- **Open selection knob (§8):** the coverage criterion + corpus source.

## 6. Per-part attribution mechanics (the tool's coloring)
- **Closed form** (squared distance + additive mean-pool sub-embeddings): `φ_P = <u_P, Σ_Q u_Q>`, `u_P =
  z_P^user − z_P^ref`. ONE encode/clip → 5 sub-embeddings → instant per-part shares summing to D². Real-time.
- **Rigor cross-check:** 32-coalition **mask-token** Shapley (masking is in-distribution for MAMP — elegant fit).
- **Pairwise couplings (optional v2):** `<u_P, u_Q>` = arm-torso synergy → draw as **edges** between joints
  (node color = per-part share; edge weight = coupling). Not needed for v1 (per-part colors already absorb
  interdependence).
- **Per-part sub-embeddings status:** additive decomposition `e = Σ_P z_P` is extractable from the existing
  forward pass (currently discarded by pooling) but **entangled** (arm-*located*, not arm-*exclusive*, via global
  attention). Fine for attribution; NOT for isolation-search (§9 uses raw kinematics instead). Clean per-part-
  *exclusive* embeddings = the future disentangled 5-encoder architecture.

## 7. Job DAG / sequencing (chimera UP 2026-08-27, VPN)
- **Phase A — CHEAP PROBES (no training; run in parallel with B, across ALL discriminative encoders**
  MAMP+pose, vanilla, MLD-AE, MLD-VAE, MLD-attn, uencfull): (A1) **peakedness / selective-vs-generalized**
  (frame-ablation, concentrate-vs-spread — the "selective vs generalized" question); (A2) **decodability**
  (linear+nonlinear pose R² — extend the reconstruction≠perception dissociation from 2 encoders to all).
  Needs: code A1 (like ssl_pool_eval), extend A2 script. [[project_mechanism_teasing_jobs]].
- **Phase B — HYBRID TRAINING** (§4): cut1 UNBLOCKED (59.7%+ captioned; contrastive on captioned subset,
  masking+pose on all). Build: refine name-norm → precompute frozen MPNet text embs → add contrastive head to
  MAMP_dualdec → smoke → train {contrastive×KL} grid → eval nested-CV (exact protocol).
- **Phase A′ — RE-PROBE the trained hybrid** (A1+A2) → characterize *why* it works.
- **Phase C — deeper (later):** decoder swap (MLD's residual edge); d_model sweep (down-capacity).
- **Stage 0′ (per-part Shapley map)** + **20-clip study** (§5): after a strong D exists.

## 8. OPEN DECISIONS (awaiting user)
- Hybrid: **Option A (our space, recommended) vs B (TMR 263-D space)**; corpus = **HumanML3D-annotated AMASS
  subset** (captions + breadth) — confirm/retrain; λ_sem grid {0.05,0.1,0.3,1.0}; frozen text (recommended).
- Study: coverage/selection criterion (farthest-point vs other); corpus source for the 20 clips.

## 9. OPTIONAL DEEPER TRACK — per-part-perception science (OFF the tool's critical path)
Under §3 the tool needs none of this, but it's genuinely interesting: **is per-part perception separable?**
- **Controlled-variation study:** triplets isolating one part (constructed via **raw per-part kinematics**, NOT
  the entangled embedding — non-circular) → whole-body ratings become weak per-part labels. Needs ≈25–30+ clips
  for 5 parts, or a 20-clip single-arm (standing-gesture, legs-held) pilot.
- **Separability axes:** entanglement (located vs exclusive) + additivity (main-effects vs interactions).
- **Disentangled 5-encoder architecture** (Gemini design) with **bypass discipline** (force each limb's signal
  through its own latent — the U-Net-skip lesson) + *learned* (not fixed-linear) fusion.

## 10. GUARDRAILS (the honesty rules we keep hitting — enforce on every new job/claim)
- Perception ≠ faithful representation (don't chase reconstruction/capacity/linear-pose for perception).
- **Bypass discipline:** force perceptual signal through the (per-part) latent; any decoder shortcut neutralizes
  the objective (U-Net-skip lesson).
- Geometric proxies < learned; **dynamics-proxies > static-geometry** (jerk/phase > position-DTW/ROM).
- VAE-manifold selects reconstruction-diversity, NOT perceptual — sample via the *perceptual* encoder.
- Semantic axis (action-level) ⊥ our task (style-within-action) — semantic helps *representationally* at best.
- Shapley = **contribution/guidance**, not a per-part perceptual distance (§3).
- **Matched-corpus for any cross-model comparison** — the full corpus's 91 effort-stylized LMA clips give a
  pretraining↔eval home-field prior (full-corpus MAMP+pose .434 is "effort-warm" vs cut1 .302 "cold"; cut1 has
  MORE clips yet lower pointing ⇒ it's familiarity, not data). Compare objectives at matched corpus (cut1).
- **Verify-before-burn:** smoke-test every job before GPU (the cv_preliminary API-bug lesson); re-verify chimera
  paths after outages.

## Pointers
Memories: [[project_mechanism_teasing_jobs]] · [[project_tmr_reconstruction_framing]] · [[project_swap_pooling]]
· [[project_localization_probe]] · [[reference_pooling_two_family_map]] · [[project_tmr_smpl_pipeline]] ·
[[reference_canonical_results_tree]]. Ledger: `docs/REPRO_TRUTH.md` §26–29. Specs: `docs/specs/HYBRID_mamppose_tmr_SPEC.md`
(migrated 2026-08-27). CV protocol (reproduce exactly): `docs/overleaf/supplement.tex` §app:checkpoint. Teaching
artifact: the mechanism-synthesis page (claude.ai artifact 64905385).
