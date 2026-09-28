# Competing-Hypotheses Design — Model × Tier × Confound

**Scope note:** this is the design spec for the **unified-corpus "competing hypotheses" program** the paper is
reframing around — *not* the original bounded TMLR sprint. It is now a flagship-scale effort (retrain the sweep +
reimplement objectives + run modern externals, all on one corpus). **Venue TBD:** leading = TMLR (rolling,
rigor-rewarding) as the flagship, with JAIR = flagship-plus (this + the new better-powered study + mechanism).
Companions: `00_REVISION_BRIEF.md`, `01_REVIEWER_MAP.md`, `03_VARIATIONAL_MAMP_SPEC.md`.

## The frame in one line
**Every model is a hypothesis about what human motion-similarity is grounded in.** The paper is not a model
sweep — it is a controlled test of these hypotheses, with the rating dataset + nested-CV + class-bootstrap as
the measuring instrument. Each design choice justifies itself as an *experiment answering a perception question.*

## Two axes (this is the crux)
- **PRIMARY = the hypothesis = the training OBJECTIVE / inductive bias.** This is what the rungs vary.
- **CONTROLS (held FIXED in Tier 1): corpus (cut1), input representation, masking regime.** Varied deliberately
  only where noted — the one deliberate architecture contrast is **Rung 3 (MAMP+pose vs MotionBERT)**.

## Tiers = internal vs external validity
- **Tier 1 — controlled (we train on cut1).** One framework, one corpus, one representation, one masking
  regime; only the *objective* (and, in Rung 3, *architecture*) varies → clean causal attribution.
- **Tier 2 — external (published, frozen).** Real named models on their *native* representations, run on our
  stimuli via conversion → external validity, but corpus + representation + conversion are **bundled** confounds.
- **Tier 0 — geometric.** Parameter-free; no training; the null learning must beat.

## The table

| # | Hypothesis: perception = … | Model(s) | Tier | What varies | Confirms if… | Current evidence (bootstrap) | Confounds / caveats |
|---|---|---|---|---|---|---|---|
| 0 | raw kinematic distance | DTW, rotation-geodesic | **0** | nothing learned | learned models fail to beat these | learned beats **geodesic** (robust ×3); **DTW ties** walk/point → timing-aware geometry is strong | DTW sign-invariance/length-norm (rev. M3); needs common eval representation |
| 1 | faithful reconstruction / fidelity | plain-AE, MLD-AE, MLD-VAE | **1** | objective = full-visibility reconstruction | recon encoders align best | **capacity dissociation** survives (more fidelity ≠ more alignment); recon **ties** masked | architecture varies within (plain/skip/VAE) but ties in bootstrap |
| 2 | movement **dynamics** | MAMP (masked, motion-only) | **1** | objective = masked delta-prediction (removes pose-copy shortcut) | masked > reconstruction | **ties** reconstruction → dynamics not separable from fidelity here | masking regime (~80%) |
| 3 | **dynamics + configuration** | **MAMP+pose** (spatiotemporal-patch masked) **＋ MotionBERT** (dual-stream DSTformer) — **FIXED-OBJECTIVE / VARIABLE-ARCHITECTURE cell** | MAMP+pose **1**; MotionBERT **1** (clean arch test) *and* **2** (frozen) | **objective SAME** (config + velocity); **ARCHITECTURE varies** | this rung aligns best AND the two architectures agree | pose head helps **walking** only; **prediction: MAMP+pose ≈ MotionBERT** (author) → architecture *under-determines* → *supports* the thesis | ⚠ Tier-2 MotionBERT bundles **representation** (3D positions vs our 6D rotations) + masking + corpus; a clean architecture claim needs the **Tier-1 reimplemented DSTformer** (match corpus + representation + masking, position+velocity objective) |
| 4 | **semantic** / action structure | TMR (text-contrastive) | **1** (reimpl. objective on cut1, 34-joint, using our HumanML3D captions + ACTOR head) *and* **2** (frozen SMPL-263) | objective = text-motion contrastive | semantic aligns best (esp. pointing) | TMR **mixed** (wins pointing, mixed elsewhere) | Tier-2 = ~4 cm BVH→SMPL + HumanML3D corpus + frozen; Tier-1 clean but bounded by **caption coarseness** (see [[project_contrastive_corpus_mismatch]]) |
| 5 | **perceptual quality** | MotionCritic | **2 ONLY** | objective = human *quality-preference* (external, unavailable to us) | a perception-trained model aligns best → perceptual training transfers to *similarity* | TBD | ⚠ **cannot be unified** — no preference labels on cut1; frozen only; **not a native similarity model** → repurpose its internal features for distance; state as adaptation + irreducible Tier-2 exception |

## Orthogonal treatment (NOT a rung)
The **perceptual fine-tune** (anchored ranking loss) is applied *across* encoders — it tests whether *light
perceptual supervision* lifts alignment, and is reported as a **treatment**, not a hypothesis. ⚠ fine-tuned models
**cannot be class-bootstrapped** (each fold's model would retrain per resample) → report with **Tier-A**
(repeat-mean, n=5) only, and say so.

## MotionBERT — why it lands on Rung 3, not its own
Author (domain knowledge): MotionBERT's pretraining loss = **3D position + 3D velocity + normalized-position** —
i.e. **configuration + dynamics**, the *same* objective family as MAMP+pose. So it is **not** a new objective
hypothesis; it is the **architecture variant** (dual spatial+temporal streams per block) on the shared rung.
Comparing it to MAMP+pose isolates **architecture at fixed objective** — cleanly *only* in Tier 1. The predicted
**tie** is the informative result: a very different architecture matching MAMP+pose reinforces the central honest
finding — *perceptual alignment is under-determined by these design choices; the objective family (learned
dynamics+configuration), not the specific architecture or product, is what matters.*

## MotionCritic — the irreducible exception to "unified across all"
Restated plainly for the record: **you cannot put MotionCritic on the unified corpus.** Its supervision is human
*preference* annotations that don't exist for cut1; making it Tier 1 would require a **new preference-annotation
study** (distinct from the similarity study). Until then it is Tier-2-frozen, features repurposed for distance. So
"unified across *all* models" is achievable modulo this one exception — state it, don't paper over it.

## Open build items (feed the run DAG)
1. **Canonical Tier-1 representation:** keep 34-joint 6D rotations as the common input; for the DSTformer, feed
   positions *derived from the same clips* (or adapt) — decide whether representation is held fixed or bundled.
2. **Reimplement vs run-frozen:** which objectives become Tier-1 reimplementations (recon, masked, masked+pose,
   dual-stream, text-contrastive) vs which stay Tier-2 externals (published TMR/MotionBERT/MotionCritic).
3. **Run DAG + parallelism** across pomplun + AICORE nodes (next deliverable).
4. Confirm the venue/scope decision (flagship TMLR vs JAIR) before the other session pivots off current-data stats.
