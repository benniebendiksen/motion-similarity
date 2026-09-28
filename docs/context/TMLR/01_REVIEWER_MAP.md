# TMLR Reviewer → Fix Map

Every AAAI-27 critique → severity for TMLR → the fix → data/analysis needed → do we have it → TMLR-required?
→ Paper 1 (TMLR) or Paper 2 (JAIR). Raw reviews in `02_REVIEWER_FEEDBACK_RAW.md`. Sources = `main_2.tex`,
canonical `triplets/cvn_results/`, memory (see `00_REVISION_BRIEF.md §7`).

Legend — **TMLR bar:** REQ = required (claims-support) · IMP = important (clarity/positioning) · SCOPE =
scope-honestly-no-new-data · OPT = optional/strengthening · ERR = reviewer error to clarify.

| # | Critique (reviewer) | TMLR bar | Fix | Data/analysis | Have it? | Paper |
|---|---|---|---|---|---|---|
| S1 | 25 CV folds treated as independent for SE/Wilcoxon (R2-Q2, AI-d) | **REQ** | Fold-aware inference: bootstrap over 56 Effort classes AND/OR average folds within repeat (n=5); re-Holm; report surviving claims | Re-analysis of fold-level scores | **Yes** (fold JSONs in bundle / `cvn_results/`) | 1 |
| S2 | Mirroring inflates n; leakage risk (R2-a, R2-Q1) | **REQ** | Report headline on **171 unmirrored** clips; confirm & state orig+mirror share Effort key → same outer fold | Re-eval on 171; grep fold-assignment code | **Yes** | 1 |
| S3 | No human-reliability / noise ceiling (R2, AI-e) | **REQ** | Inter-rater reliability + noise ceiling from raw responses; report ρ against ceiling in main text | Analysis of raw MTurk responses | **Yes** (de-identified responses, [[reference_perceptual_data_provenance]]) | 1 |
| S4 | MLD-VAE beats MAMP+pose on walking (.597 vs .552), excluded "as context" (AI-c, R2-b) | **REQ** (framing) | Reframe claim → masked-prediction-vs-reconstruction; state MLD-VAE not uniformly beaten; +optionally the VAE run (R1) | Rewrite; optional new run | Yes (numbers) / run TBD | 1 |
| S5 | Target = neutral-conditioned choice prob, not pairwise dissimilarity; ordinal used as cardinal margins (AI-a) | **REQ** (framing) | Frame target as neutral-anchored ordinal similarity; recast anchored FT loss as **rank/ordering** (margins weight, don't assert distance); drop cardinal language | Rewrite only | Yes | 1 |
| P1 | Novelty vs MotionCritic (Wang'25), MotionBERT (Zhu'23), SkeletonMAE (Yan'23) missing (AI-b) | IMP | Add related-work; distinguish (none has our perceptual benchmark) | Lit + writing | — | 1 |
| P2 | "Combination of existing models reduces contribution" (R1-b, R1-c) | IMP | Reframe contribution = **dataset + dissociations**, not architecture; TMLR ignores novelty per se | Writing | — | 1 |
| P3 | Hard to read / "heavy AI use"; abbrevs (MAMP, TMR) defined late (R1-a) | IMP | Define terms on first use; **method+losses+protocol before ablation tables** (AI suggestion); tighten prose | Restructure | — | 1 |
| P4 | "Effort" unclear, and how it "influences pose detection" (R1-d) | IMP | Clarify LMA Effort + stimulus pipeline; correct the misframing (Effort parameterizes *stimuli*, not "pose detection") | Writing | Yes ([[reference_dperc_lma_effort_space]]) | 1 |
| P6 | "Very little explanation justifying selection of models/configs" (R1-c) — the design walkthrough isn't principled | **IMP** (+1 run) | (a) Reframe design section as motivated **hypothesis-tests** (each rung: question → change one thing → alternative ruled out → outcome → motivates next). (b) Add a **design-ladder figure** vanilla→MAMP+pose. (c) **Own the REGIME-DEPENDENCE of the skips** (⚠ corrected 2026-09-25 — NOT "skips win reconstruction"; that conflated the backbone comparison with the skip ablation): under **reconstruction** the encoder–decoder skips are perceptually **near-inert** (plain+skips pointing .213→.221; MLD−skips −.001) — the MLD recon advantage (.213→.331) is its **overall config, not the skips**. Under **masking (motion-only)** skips become consequential + **action-dependent**: help walking (.351→.470, +.119) & picking (+.076) but HURT pointing (.299→.275). With the **pose head** added they turn **redundant+harmful across all 3** (MAMP+pose plain→U-Net: pointing .434→.260). This justifies the final choice = **plain encoder + pose head, skips dropped**: skips harm pointing under masking and duplicate the pose head's signal (they leak frame-level pose to the decoder). State as a result. **⚠ GATED ON S1/S2 (author, 2026-09-25): these skip deltas are on the OLD stats (25-fold-as-independent, mirrored) — RE-VERIFY under fold-aware + no-mirror before stating as fact.** Triage: ROBUST = the reconstruction nulls (+.008/−.001), the large masking help (walk +.119, pick +.076), the pose-head pointing collapse (.434→.260, −.174). FRAGILE = the −.024 pointing-hurt under masking-motion-only, and "harmful across all 3" (walk −.028, pick −.058) — may soften/vanish. General rule: the stats fix touches EVERY small-delta claim in the design ladder, not just the headline. (d) Clarify **TMR is the supervised yardstick, not a rung**; ladder culminates in MAMP+pose. (e) Complete the factorial's one real hole = **VAE×masked** (see R1run). | Writing + figure + 1 run | Yes (numbers/tables exist) | 1 |
| P5 | Wants with/without **CLIP** ablation (R1-b, rebuttal) | **ERR** | Clarify: **we don't use CLIP**; TMR = frozen DistilBERT→ACTOR ([[reference_tmr_architecture]]); our encoder is caption-free | One sentence | Yes | 1 |
| C1 | "Limited supervision rivals large-scale semantic" overstated (R2-c) | SCOPE | Soften → mixed result: post-FT beats TMR only on walking (trend, p_adj=.078), ties picking, loses pointing | Writing (numbers exist) | Yes ([[project_tmr_finetune_result]]) | 1 |
| C2 | Headline (.552/.434/.543) vs matched (.462/.302/.481) gap (R2-b, R2-Q3) | SCOPE/IMP | Explain: pretraining **corpus/domain** (full vs AMASS-matched) + TMR **BVH→SMPL ~4cm** conversion; label Table 8 rows AMASS-pretrained | Writing | Yes ([[project_legacy_vs_cut1_provenance]]) | 1 |
| C3 | Narrow generalization: synthetic, 1 skeleton, 1 protocol, 3 actions, 1 base motion/action (R2-d, AI-f) | SCOPE | State limitation plainly; real fix (natural motion / more bases / **2nd study**) = Paper 2 | Writing | — | 1 (scope) / 2 (fix) |
| M1 | Sec 3.2 feature-dim inconsistency (204 vs +3-D root) (AI-minor) | IMP | State exact layout (34×6=204; root slot usage) | Writing | Yes | 1 |
| M2 | Sec 4.3: delta offset `s` undefined; masked-vs-all patches per loss unclear (AI-minor) | IMP | Define `s`; state loss is masked-only (per code) | Writing | Yes | 1 |
| M3 | DTW quaternion sign-invariance / path-length normalization (AI-minor) | IMP | State DTW handling of quaternion double-cover + normalization | Writing/verify | Yes | 1 |
| B1 | Baseline coverage — stronger learned / perceptual-metric baselines (R2-b) | OPT | Optionally add a learned baseline (MotionBERT); else defer | New run | No | 2 (likely) |
| R1run | ⭐ "Why not a MAMP-VAE+pose?" (author, from S4) — **TRIPLE-DUTY** | **likely REQ for P6** | Variational MAMP+pose (VAE bottleneck + pose), 1–2 cells {VAE-masked, VAE-masked+pose}. Simultaneously: (i) completes the design factorial's one hole (P6), (ii) answers MLD-VAE-honesty (S4), (iii) = variational-MAMP lever (Paper 2). Elevated from optional → likely include in Paper 1. | New chimera run (1–2 cells, not a sweep) | Run TBD | **1** (+2) |
| F1 | Factor-level error analysis (Space/Weight/Time/Flow) (AI suggestion) | OPT | Break down alignment by Effort factor / State-Drive / active region | Analysis | Partly | 1 supp or 2 |
| F2 | Body-aware / localized pooling (AI suggestion) | OPT | Localized/directional pooling for pointing | New runs | mechanism thread | **2** |

## Execution order (sprint)
1. **S1 + S2** (fold-aware stats + no-mirror) — required, and sets how strongly everything else can be claimed.
2. **S3** (reliability/noise ceiling) — required calibration.
3. **S4 + S5 + C1 + C2** (honesty/scoping reframes) — writing on known numbers.
4. **P1–P5, M1–M3** (positioning, readability, minors) — writing.
5. **R1run** (variational MAMP+pose) — decide include-in-Paper-1 vs hold-for-Paper-2 after S1 outcome.
6. Template swap AAAI→TMLR; supplement carry-over.
