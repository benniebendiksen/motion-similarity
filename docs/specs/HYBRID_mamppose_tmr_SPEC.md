# HYBRID EXPERIMENT SPEC — MAMP+pose × TMR (masking + pose + reconstruction + semantic-contrastive)
Migrated from scratchpad + corrected 2026-08-27. North star: `docs/RESEARCH_ROADMAP.md` §4.

## Goal
A top-performing perceptually-aligned **whole-body** encoder — potentially the first to clear all three
actions **at matched corpus** — whose better **D** then serves:
  (1) sharper per-part **Shapley contribution** attributions (the tool's colors ride an improved D);
  (2) **perceptually-meaningful sample selection** for the 20-clip study (broad manifold coverage).
  (3) [REFRAMED 2026-08-27] the study **enriches D (whole-body)**; per-part *human* validation is now the
      OPTIONAL deeper track (Roadmap §9), NOT a required role — the tool deploys on whole-body validation via
      Shapley (Roadmap §3). Do not gate this model on per-part validation.

STATUS: DESIGN ONLY (chimera unreachable 2026-08-17; re-verify all chimera paths when back). Grounded in the
verified TMR objective ([[project_tmr_reconstruction_framing]]), public repo github.com/Mathux/TMR (MIT,
TEMOS-based, precomputed MPNet text, 263-D Guo feats), and our staged TMR integration
([[project_tmr_smpl_pipeline]]).

## VERIFIED TMR OBJECTIVE (the semantic ingredient to graft)
L_total = L_TEMOS + λ_NCE·L_NCE ; λ_NCE=0.1.
L_TEMOS = L_R(smooth-L1 motion recon, cross-modal) + λ_KL·L_KL(4 terms) + λ_E·L_E(smooth-L1 emb-sim); λ_KL=λ_E=1e-5.
L_NCE = InfoNCE on cos(z^T, z^M), τ=0.1, false-negatives filtered (text-text sim > 0.8 via MPNet discarded).
⇒ TMR = reconstruction-VAE backbone + 0.1-weight contrastive. Graft = its **contrastive-to-text** term.

## DESIGN — Option A (RECOMMENDED): graft contrastive onto OUR MAMP+pose (34-joint 6D pipeline)
Keep masking+pose in our codebase (MAMP_dualdec); eval drop-in identical to every encoder; our representation
is what the perceptual eval is calibrated on.
- **Objective:** L = L_motion(masked) + λ_pose·L_pose + **λ_sem·L_NCE(pooled motion latent ↔ frozen text emb)**.
  Reuse TMR InfoNCE + false-neg filtering (τ=0.1, thr 0.8). Text = **frozen** MPNet embeddings (TMR stores text
  as a table → no text encoder to co-train; only our motion latent aligns).
- **Held-out eval:** the 342 LMA clips strictly held out; they have NO text ⇒ contrastive is a PRETRAINING-only
  signal; eval = pure motion-embedding L2 vs d_perc.
- **Option B (fallback/cross-check):** graft masking onto TMR (263-D, TMR env); uses TMR text natively; heavier.

## ⭐ RUN AT MATCHED CORPUS (cut1) — TWO reasons (corpus-fairness + effort-exposure confound)
The ablation base MUST be corpus-matched, else corpus confounds objective. Reasons:
1. **Fair to TMR** (both trained on ~HumanML3D-budget, non-effort data).
2. **⭐ Effort-exposure confound (user-raised 2026-08-27):** the FULL corpus (5,453 = 91 non-eval LMA effort +
   Bandai 3,077 + CMU 2,285) contains **effort-stylized exemplars** (same PERFORM/LMA generation as the eval
   clips). cut1 (14,143 AMASS, NO LMA effort) does NOT. Note cut1 has MORE clips yet WORSE pointing (.302<.434)
   ⇒ full-corpus's edge is *effort-distribution familiarity* (a pretraining↔eval distribution overlap; NOT
   leakage — the 91 are disjoint from the 342 eval — but a home-field prior), not "more data". So full-corpus
   .434 is "warm", cut1 .302 is "cold". Matched corpus removes this confound.

**Anchors we ALREADY have (matched corpus, nested-CV, n=25):**
| model @ matched corpus | walk | **point** | pick | bars cleared (.478/.370/.431) |
|---|---|---|---|---|
| cut1 MAMP+pose (masking+pose, λ_sem=0) = the BASE | .462 | **.302** | .481 | picking only |
| TMR (recon+KL+contrastive) | .452 | **.425** | .408 | pointing only |
| **Hybrid (masking+pose+contrastive)** = target | ? | **? → .425+ w/o losing pick** | ? | **all three?** |

Clean causal test (corpus fixed): does adding contrastive to cut1 MAMP+pose **close the pointing gap
.302→.425 (TMR's level) WITHOUT losing picking**? A hybrid clearing all three at matched corpus = combines
both routes = the headline. (Full-corpus MAMP+pose already clears all three but is NOT corpus-fair to TMR and
is effort-warm — it's the absolute reference, not the comparison base.)

## ⚠️ LOAD-BEARING CAVEAT (sharpened by the cut1 lens)
TMR's semantic axis is **action-level**; our task is **style-within-action** (text "a person points" is
CONSTANT across the effort variants we rank). So contrastive gives ~ZERO *within-action* discriminative signal
at eval. Yet TMR still WINS pointing at matched corpus (.425) ⇒ its edge is **representational** (semantic
pretraining shapes gesture priors), not eval-time discrimination — exactly the mechanism the hybrid tries to
graft. HYPOTHESIS = representational transfer; success = "semantic priors lift within-action pointing on
text-less LMA clips w/o hurting walk/pick." NULL result still publishable: routes not composable.

## ABLATION GRID (all at MATCHED corpus) — {contrastive × KL} on the masking+pose base
⭐ KEY HYPOTHESIS (each ingredient may fix a DIFFERENT action, per the localized/distributed axis):
**contrastive → POINTING** (semantic/communicative); **KL/variational bottleneck → WALKING** (smooth latent
suits periodic/distributed motion — MLD-VAE's +.065 walking edge; cut1 walk .462 is BELOW the .478 bar so KL is
the natural lift); **masking+pose → PICKING**. FULL hybrid tests whether combining clears ALL THREE at matched
corpus.
| run | masking | pose | contrastive λ_sem | KL | purpose |
|---|---|---|---|---|---|
| cut1 MAMP+pose (BASE) | ✓ | ✓ | 0 | ✗ | control (.462/.302/.481) |
| +contrastive | ✓ | ✓ | {0.05,0.1,0.3,1.0} | ✗ | pointing lift → TMR's .425? |
| +KL | ✓ | ✓ | 0 | λ_KL swept | walking lift? (cut1 .462<.478; MLD-VAE .597) |
| +contrastive+KL (FULL) | ✓ | ✓ | best | best | all three cleared at matched corpus? |
| contrastive-only | ✗ | ✗ | 1.0 | ✗ | isolate semantic (our-space TMR-lite) |
| masking+contrastive (no pose) | ✓ | ✗ | 0.1 | ✗ | pose-interaction check |
DISCIPLINE: NOT a full 2⁴ factorial — only the key combinations above. Always log all 3 actions (interference).
KL rationale = [[project_mld_ae_vs_vae_justification]] (VAE did well, esp. walking; dropped only for deterministic
apples-to-apples — reconsidered here since TMR is a VAE anyway).

## ⭐ DESIGN REVISION 2026-09 (supersedes the KL-column placement above; not yet re-locked — needs user GO)
Working through KL + contrastive on a MASKED backbone changed two things:
1. **KL PLACEMENT — relocate to the per-patch masked-pred bottleneck (variational MAMP).** The grid above left
   "KL" unplaced; the version first BUILT put μ,σ on the POOLED side-readout feeding only contrastive — that is
   FOOTLESS for perception (regularizes a captioned-only readout, not the representation) AND, if instead given a
   pooled pose-recon role, is LITERALLY arm-1a (collapsed pointing .265). CORRECT spot = μ,σ,sample on the
   ENCODER's PER-PATCH latents (encoder↔decoder), small β·KL, masked-pred+pose flows THROUGH z. This regularizes
   the perception-carrying bottleneck via the perception-WINNING objective (masked-pred, not faithful full-recon)
   ⇒ avoids arm-1a; mirrors WHY the MLD-VAE helped. Hypothesis unchanged: helps WALKING, ~inert pointing.
   Eval deterministic (μ). Option A = per-patch (simple; distributed bottleneck, imperfect MLD analog);
   Option D (reserve) = tight pooled global-context vector the masked-decoder also predicts from.
2. **CONTRASTIVE TEXT SIDE — REVISED AGAIN 2026-09-19 (code-verified TMR arch, [[reference_tmr_architecture]]).**
   The frozen-MPNet-SENTENCE-vector target is WRONG and superseded (raw pooled 1-D vector from an SBERT model
   whose geometry is a rigid text-text hypersphere — hostile to being warped into a motion-joint space; also
   destroys linguistic sequence). TMR does NOT contrast on raw frozen embeddings: it uses FROZEN
   `distilbert-base-uncased` TOKEN embeddings → a TRAINABLE `ACTORStyleEncoder` head → text latent.
   ⇒ **ADOPT TMR's recipe:** frozen DistilBERT token embeddings (reuse `save_token_embeddings`) → a LIGHT
   trainable **non-VAE** ACTORStyleEncoder head (nbtokens=1, deterministic) → text latent; contrast motion-latent
   vs it. **Non-VAE is deliberate:** our VAE/KL lever is variational-MAMP on the MOTION bottleneck; a variational
   TEXT latent's KL would be FOOTLESS here (feeds only contrastive, no recon through it — same orphan we rejected
   on the pooled-motion side; TMR's text-KL only earns footing via cross-modal recon `D(z_T)→m`, which we dropped).
   - **FNF on EVERY arm:** false-neg filter uses the SEPARATE frozen MPNet caption-caption cos>0.6 mask,
     independent of the target encoder (exactly as TMR). MPNet's role = filter only.
   - **Params:** τ=0.7; keep the ACTOR head LIGHT (few layers) — captioned set ~30k < TMR's ~45k → overfit guard.
   - **Frozen-MPNet-sentence = OPTIONAL cheap ablation only** (to empirically show it underperforms; FNF still on).
3. **OBJECTIVE RE-ANCHORED (user 2026-09-19).** Phase-2 goal = general PERCEPTUAL RELEVANCE of the manifold (for
   downstream Phase-3 sample-selection + verification), NOT the LMA-effort ranking. LMA-effort ρ = the current
   eval's labels, NOT the objective. So LMA ρ is a GUARDRAIL + weak proxy (big drop = over-collapse warning; gain
   = bonus), NOT the arbiter. Judge the pilot via: LMA ρ (guardrail) + A′ re-probe + NN/retrieval sanity on
   captioned held-out; real verdict = Phase 3's new triplet study.
⇒ Corrected 2×2 = **{contrastive × variational-MAMP}**; contrastive text side = frozen-DistilBERT-tokens →
   light non-VAE ACTOR head + FNF. LAUNCH ORDER: contrastive-only pilot (grid-1) first.

## ⭐ REPRODUCE-EXACTLY CV PROTOCOL (from supplement §Checkpoint-Selection, app:checkpoint — DO NOT DEVIATE)
Any new training/eval MUST match this so numbers are comparable to MAMP+pose/TMR:
- **Harness:** `cv_nested.py --k 5 --repeats 5 --n-triplet-seeds 5 --triplet-epochs 200`.
- **Per outer fold:** 3-way disjoint **TEST / SELECT / TRAIN**. Raw Spearman on TEST. **n=25 = 5 folds × 5 repeats.**
- **Seeds:** stratified **42 + 1000·repeat**; folds shuffle ONLY the canonical **57 Effort classes per action**.
- **Checkpointing:** full trajectory every **100 epochs**.
- **Checkpoint selection (the perceptual-relevance step — make explicit; supplement is only implicit here):**
  among the grid of every 100th epoch within **[100, PLATEAU]**, the reported checkpoint is the one that
  **MAXIMIZES mean raw-Spearman across the 3 actions on the SELECT fold** (action-agnostic; this is where
  perceptual relevance enters — verified in `cv_nested.py` SELECT step). The window bound PLATEAU = the
  encoder's OWN reconstruction-loss plateau (val_recon = MSE in normalized space on a 5% held-out split,
  seed 0) — reconstruction-defined, NEVER perceptual, ONLY so the SELECT fold isn't used twice (once to bound
  the window would double-use it; instead SELECT is used once, for the perceptual pick WITHIN the window).
- **PLATEAU operational def:** 3-point moving-average smooth of val_recon; asymptote = min over final 20% of
  trajectory; PLATEAU = first grid-100 epoch where smoothed ≤ 1.02× asymptote AND stays ≤ 1.03× thereafter.
- **Masked-prediction family (⟵ the hybrid IS one):** can DIVERGE if over-trained ⇒ PLATEAU = the
  **pre-divergence** epoch under the same rule. (MAMP cap ~1199; the hybrid needs its own val_recon curve read.)
- **Perceptual fine-tuning (if run):** fixed 200-epoch budget, NO early stopping on test sets.
- **Env (matched):** Rocky Linux 8.10, A100(80GB)/H200(141GB), PyTorch 2.4.1 / CUDA 12.1 / Python 3.10
  (TMR side: PyTorch 2.5.1 / Lightning 2.6). Hybrid trains in our `torch_gpu_cu12` env; contrastive-to-TMR-text
  needs the TMR text table (frozen) staged in.

## HONEST RISKS
1. Multi-objective interference: contrastive may pull the latent off the kinematic structure masking built →
   walk/pick drop. Uncertain → good experiment.
2. Text-motion pairing: align HumanML3D caption IDs ↔ our AMASS retargeting (HumanML3D gives the AMASS source
   map). Feasible engineering step. **VERIFY cut1 composition (below).**
3. [DOWNGRADED per §3 reframing] Whole-body gain ≠ per-part faithfulness — no longer a blocker; per-part is the
   optional deeper track (Roadmap §9). Whole-body validation SUFFICES for the tool (Shapley contribution).

## OPEN DECISIONS (user)
- Option A (our space, recommended) vs B (TMR space)?
- λ_sem grid {0.05,0.1,0.3,1.0}; align raw pooled latent vs a light projection head?
- Frozen text embeddings (recommended) vs co-trained text encoder?
- **✅ RESOLVED 2026-08-27 (cut1 composition):** cut1 = 14,143 **AMASS clips** retargeted to CMU-33 (source-named,
  `datasets/cut1_flat`, `norm_stats_cut1.npz`), an AMASS budget-match — NOT the exact HumanML3D-captioned set.
  BUT HumanML3D captions ARE staged (`tmr/repo/datasets/annotations/humanml3d/annotations.json` = id→source-path→
  captions). **DEFINITIVE classification (exhaustive clip-id match, `cut1_final_classify.py`): 12,972/14,143 =
  91.7% CAPTIONED; 1,171 = 8.3% GENUINE ABSENCES** (structural, not normalization): TCD dataset (61) is NOT in
  HumanML3D at all; cmu(498)/BMLhandball(361)/BioMotionLab(239) are subset-exclusions HumanML3D never annotated.
  ⇒ cut1 = AMASS budget-match overlapping HumanML3D ~92%, NOT the exact HumanML3D set.
  **⭐ CORPUS DECISION (for full caption alignment — user goal): use the 12,972 fully-captioned clips as the
  hybrid corpus (drop the 1,171 absences) ⇒ contrastive applies to EVERY clip, no partial-supervision split;
  ~13k = same order as TMR.** For clean comparison, RETRAIN the MAMP+pose base on the SAME 12,972 (apples-to-apples;
  the current cut1 base .462/.302/.481 was on all 14,143 — re-derive on 12,972). TMR already on HumanML3D.
  Build steps: (i) freeze the 12,972 caption-aligned clip list + captions; (ii) precompute frozen MPNet text embs;
  (iii) add contrastive head to MAMP_dualdec; (iv) retrain base + smoke 1-clip fwd; (v) train {contrastive×KL} grid.

## NEXT (chimera back): re-verify tmr/models + tmr/repo + guofeats + env; confirm cut1 composition + its
val_recon curve (for PLATEAU); smoke a 1-clip contrastive-head forward pass BEFORE any full train.
Pointers: [[project_mechanism_teasing_jobs]], [[project_tmr_reconstruction_framing]], [[project_tmr_smpl_pipeline]],
[[reference_canonical_results_tree]]; ledger REPRO_TRUTH §12/§25; supplement `docs/overleaf/supplement.tex` §app:checkpoint.
