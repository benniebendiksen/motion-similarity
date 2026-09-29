# Reproducibility — Single Source of Truth

**Purpose.** ONE place holding: (1) the official eval protocol, (2) each model's reconstruction-loss
plateau cap to apply under it, (3) which existing evals are protocol-compliant vs must be re-run, and
(4) the locked findings. If a number is claimed in the paper, it must trace to a row here.
Cross-refs: `provenance_ledger.md` §0.5 (training) / §0.6 (eval); memory `project_methodology_decisions.md` (D-decisions).

Last updated: 2026-07-25 (full claim-by-claim audit vs ledger+memory; see §13).

## 13. CLAIM-BY-CLAIM AUDIT (2026-07-25) — paper vs ledger/memory, post-TMR-reconciliation
Systematic pass matching every quantitative/comparative claim + implication in main.tex against ground truth.
VERDICT: paper is CONSISTENT. Checks passed: (a) all "all three" claims correctly scoped to geometric/vanilla
baselines NOT TMR; (b) every TMR verb (matches/exceeds/behind/tied/retains) matches the two-sided Wilcoxon
(§7.3) exactly; (c) dissociation-2 table (.434/.509…​.213/.727) + nonlinear (.509→.645, .727→.829) verbatim-correct;
(d) capacity numbers (.221/.213/.00305/.00563/3.9×) correct; (e) headline table .552/.434/.543 correct; (f) MAMP
= "prediction" (no "inpainter" mischaracterization); (g) velocity-only-lifts-pointing on PLAIN transformer (not MLD)
correct. FIXES APPLIED during audit: (1) stale Limitations claim "raw edged on pointing, wins only after fine-tune"
was FALSE (ft MAMP loses pointing to ft TMR) → corrected to "behind on pointing both raw and after matched fine-tune";
(2) added Limitations bullet that pointing-interpretation is not a tested causal claim; (3) §6.1 headline caption now
notes .552 = full-corpus vs .462 = corpus-matched (the legacy-vs-cut1 reader-confusion point); (4) structure reorder
Results→Discussion→Limitations→Reproducibility→Appendix (was Repro before Limitations); Discussion consolidated
(2 sections → 1 section + subsections; Dissociation 1/2 → subsubsections). All refs resolve, tables 12/12.

## 19. FLOW / HUMANIZATION PASS (2026-07-26, in progress) — main_2.tex, matching Funda's cohesive voice
Guardrails: numbers/claims/citations FROZEN (all verified §16-18); prose/bridges only; clinical register (no flowery); bridging clauses at EVERY conceptual seam (user's central pattern); consolidate over-fragmented subsections (Funda style).
DONE this pass:
- Intro: broad opener (health/rehab FIRST then animation) w/ verified cites (yurtman2014rehab, Wang2015/Aristidou2018/Kovar2002/holden2017fast/Aberman2020-yl); DTW-baseline cites (muller2007ir book, kovar2004extraction) — ALL web-verified real, added to references.bib. Filled the %TODO gap (geometric-metrics-deviate-from-perception → learned-model motivation). SO(3) glossed. "orders" not "classifies" (we rank, not classify).
- §4 Per-Action: killed "(1)/(2)" inline enumeration → prose forecasting pointing-as-discriminating-action.
- §5 CONSOLIDATED 5→3 subsections: (Backbone) + (Objective: reconstruction/velocity/full-visibility limits — merged old 5.2+5.3) + (From masked prediction to MAMP+pose — merged old 5.4+5.5). Bridges threaded Step2→5. MAMP described faithfully (predicts TEMPORAL MOTION/deltas of MASKED joints; input=masked absolute pose patches NOT velocities; guardrail: MAMP "not superiority but complementarity" per memory retraction). Added MAE encoder/decoder explanation (decoder=training scaffold, discarded at inference, encoder=the product) — closed conceptual gap; de-duped vs §6.1 mean-pool.
- §7 CONSOLIDATED 6→4 (Headline / Ablations[merged pose-ablation+sweep+arch×obj] / TMR / Fine-tuning) + REORDERED to Headline→Ablations→TMR→Fine-tuning (establish→justify→compare→extend). Ablations→TMR bridge added ("advantage is objective not architecture" — ledger-faithful → "far larger supervisory signal"). Labels sec:poseablation/sec:skip-ablation/tab:sweep preserved; word-count identical pre/post reorder (pure move); refs resolve; tables 12/12; 5 resizeboxes intact.
§8 DISCUSSION FLOW PASS DONE: opener bridges Mechanisms→"is alignment just faithful representation? no"; §8.2 retitled "Perceptual alignment is not faithful representation" + framing (capacity vs linear-accessibility); capacity→decodability bridge; ADDED the "why linear" rationale (L2 metric → only linear info visible, non-circular); scope guardrail restored ("pointing, the one action on which encoders separate enough" — NO walking-mechanism claim, "abstract" not used); synthesis close unifies BOTH dissociations + TMR into thesis ("learnable but not explained by fidelity/linear-kinematics/semantic-supervision alone; open question; benchmark helps answer"). Numbers intact, guardrails honored, tables 12/12.
BIB HYGIENE (post-compile): removed stray \bibliographystyle (sty auto-issues); removed dup mahmood2019amass from references.bib (kept Funda's w/ pages); added cmumocap year=2003; all [h]→[htbp] (float warning). fundaRef.bib (215 entries) wired via \bibliography{references, fundaRef}. Funda added Related Work prose + §3 Data Corpus (Stimuli/Study Design/Results/Corpus) + Intro edits — her voice = the target.
✅ TWO VERIFICATION SWEEPS DONE (2026-07-26):
(1) HARD-NUMBER/CLAIM FIDELITY: re-extracted every PROSE numeric claim post-flow-edits, verified each vs ledger — all deltas arithmetically exact (backbone +.118/+.056, pose-head +.201/+.135, U-Net +.119/+.076, fine-tune +.237/+.161/+.124, skip .213→.221/−.001, capacity .00563→.00305 etc, nonlinear .509→.645), all Wilcoxon p unchanged. NO flow edit corrupted a value.
(2) FLOW-SEAM SWEEP: checked every (sub)section opener; ~12 "cold" flagged → triaged to 2 real narrative cold-starts FIXED (§5.1 Backbone "The first decision is the backbone..."; §7 Results section-opener mapping headline→ablations→TMR→fine-tune arc). Rest legitimately declarative (methods/repro/appendix) or already bridged (TMR bridged by Ablations close). tables 12/12, braces 0, all refs resolve.
FLOW/HUMANIZATION PASS ESSENTIALLY COMPLETE across all assistant-written body. Remainder = Funda's domain (external Related Work cites, final bib field verify).

## 18. ABSTRACT-CHANGE RISK MANAGED (2026-07-26) — AAAI "no substantial abstract change" rule
AAAI-27: 9pg total, pp8-9 references-only (=7pg content; we're AT 7 ✓). RULE: "reject without review if abstracts
change SUBSTANTIALLY between abstract & final deadline." Our submitted (Jul 21) abstract claimed "surpasses TMR on
all three actions" + "stronger than TMR on two of three" — BOTH now FALSE (TMR fine-tune experiment ran POST-deadline;
matched comparison: TMR wins pointing, raw MAMP+pose sig-beats TMR on only 1 [picking], ties walking).
DECISION (user, minimize-drift): reverted abstract to SUBMITTED text VERBATIM, changed ONLY the 2 false TMR clauses.
Word-diff confirms: ONLY those 2 sentences differ; opening ("We show that a single self-supervised..."), MAMP+pose
desc, capacity+decodability dissociations, conclusion = IDENTICAL to submitted. 2067 chars. Defense if questioned:
change confined to 2 sentences re ONE baseline (TMR), corrected TOWARD honesty after a post-deadline symmetric-fine-tune
control; thesis/method/other results unchanged — opposite of bait-and-switch. main.tex (old) abstract also holds this.

## 17. main_2.tex HYPOTHESIS-TEST-GROUNDED FACTUAL SWEEP (2026-07-26)
Audited every comparative claim against Wilcoxon verdicts (checklist-rigor). 4 real errors FIXED:
- (A) Limitations "unsupervised MAMP+pose trails TMR on pointing" = FALSE (headline .434 > raw TMR .425; conflated full-corpus w/ corpus-matched). → scoped to corpus-matched, both regimes.
- (B) Abstract+contrib "outperforms reconstruction ENCODERS (plural) across all three" = OVERCLAIM at significance: MAMP+pose sig-beats VANILLA all 3 (p=.004/9e-6/.005) but MLD-AE walking is a TIE (p=.129, +.020 n.s.); MLD-VAE set aside (not a comparison target per [[project_mld_ae_vs_vae_justification]]). → SINGULAR "a reconstruction-based autoencoder" (=vanilla). Body verified: never claims to beat MLD-AE on all-3; sweep "clears GEOMETRIC baseline thresholds" (not "beats all encoders") is correct (MLD-AE fails pointing bar .331<.370).
- (C) "MLD achieves superior alignment across ALL actions" overstated picking (.498 vs .491 ~tie). → per-action (point +.118, walk +.056, pick ~unchanged).
- (D) Fine-tune "plateaus epochs 100–120 (≈0.60)" = INVENTED (not in logs) + CONTRADICTS fixed-200-epoch protocol (§Reproducibility). → replaced w/ sourced gains +.237/+.161/+.124.
VERIFIED clean: k=16 PCA/ridge, capacity 35M/hidden384/heads6/depth7, nonlinear .509→.645, all table numbers, TMR paragraph (every verdict = Wilcoxon), headline p-values. KEY LENS: point-estimate "outperforms/superior" claims that fail significance were the weak spots — condensation asserted comparisons from means without the test.

## 16. main_2.tex VERIFICATION SWEEP (2026-07-26) — the voice-rewrite audit
main_2.tex = user's condensed voice-rewrite (~3540 words vs main.tex ~9500; solves 12pg problem). FULL factual audit vs ledger:
- ✅ ALL numbers exact: headline table, TMR two-regime table (.452/.425/.408, .462/.302/.481, .649/.573/.611, .699/.463/.605), all Wilcoxon p-values, dissociation-2 table (.434/.509…​.213/.727) + direction (winner decodes pose LEAST — preserved), nonlinear .509→.645, capacity (.221/.213/.00305/.00563/35M), pose-ablation gains (+.201/+.135), skip-ablation gains (+.119/+.076), corpus (433+3077+2285=5795, −342=5453), velocity (.213→.362), masking 80%, seed 42+1000·rep. Tables 12/12, refs resolve.
- ⚠️ FIXED BUG (user caught): Limitations "pointing margin" bullet said "Unsupervised MAMP+pose... trails text-supervised TMR on pointing" — FALSE for headline model (.434 > raw TMR .425); conflated full-corpus headline (.434) with corpus-matched TMR-comparison model (.302). CORRECTED to scope: "corpus-matched MAMP+pose trails on pointing both before+after fine-tune (§tmr)".
- ✅ FIXED earlier this session: 5× [cite: N] tool-artifacts removed; false "static poses/models dynamic progression" contribution bullet → complementarity (ledger-retraction-compliant); SO(3) gloss + "temporally aligned" clause; hand-crafted→predefined.
- RESIDUAL: ✅ (a) SKIP-EVIDENCE SEAM FIXED 2026-07-26 — restored the direct toggle proof into §skip-ablation ("adding skips to plain recon .213→.221; removing skips from MLD −0.001; skips near-inert; MLD advantage = configuration not skip links"). Numbers match ledger §4.1. Backbone claim (line ~294) now points to a section that PROVES it. Section logic now: skips-inert → port U-Net into MAMP → pose objective recovers config. ⏳ (b) bib %VERIFY page numbers deferred to camera-ready. ⏳ LENGTH still needs final Overleaf page count.

## 15. OVERLEAF COMPILE FIXES (2026-07-25) + AI-TROPE SWEEP
- BibTeX: `.bst` must be named `aaai2027.bst` (user's Overleaf had `aai2027.bst` — renamed). references.bib had in-entry `%` comments (ILLEGAL in BibTeX — `%` is not a .bib comment inside entries) → REMOVED; VERIFY notes now live here/ledger only. Page numbers + author lists (MAMP/Bandai/dtaidistance/Laban) STILL need verification before camera-ready.
- main.tex: (1) REMOVED my redundant `\bibliographystyle{aaai2027}` — aaai2027.sty auto-issues it under natbib (line 354); a 2nd triggers "Illegal, another \bibstyle". (2) ADDED `\affiliations{}` after `\author{}` — AAAI `\maketitle` needs `\aaai@affiliations` defined; `[submission]` mode suppresses content + prints "Anonymous submission" anyway, so empty is correct.
- Compile needs 2 passes (latex→bibtex→latex) for cites to resolve.
- AI-TROPE SWEEP DONE: prose em-dashes ~40→0 (1 kept by user choice + legit item/title/table uses); colon-antithesis removed from abstract+titles; filler adverbs (Crucially×2)→0; "robust" cluster 4-in-para→varied; punchy antithesis titles softened ("modeling vs copying"→"Why MAMP+pose succeeds"; "capacity buys reconstruction, not perception"→"capacity improves reconstruction without improving perception"). KEPT substantive "rather than"/negations (10, all precise contrasts — not tropes). Zero delve/leverage/underscore/seamless. Integrity: parens/braces/tables balanced.
- ⚠️ LENGTH: PDF = 12 pages, likely ~5 OVER (AAAI 7pg). Length-reduction pass PENDING user decision (Tier1 relocate Repro/Pearson/ablation-tables to supplement = lossless ~2-2.5pg; then Tier2 compress).

## 14. FINAL LARGE-SCALE SWEEP (2026-07-25) — validity/soundness/over-reach/consistency/completeness
- ⭐ FIXED CRITICAL BUG: ~13 HARDCODED section cross-refs ("Section 6.3/6.4/6.8/4.1/4.4/2.2/2.3/3/5.2") were
  ALL WRONG after Related-Work insertion (§ shifted +1) + Discussion consolidation + reorder. Converted every
  one to \label/\ref (added 14 labels: sec:data/stimulus/motionrep/corpus/peraction/step1/step4/budget/pipeline/
  baselines/tmr/finetune/poseablation/sweep). Now self-healing. Verified: ZERO hardcoded section numbers remain,
  ALL \ref resolve, no duplicate labels.
- VERIFIED CLEAN: MAMP masking 80% hidden/20% visible consistent; invariants (n=25 ×15, 1540/action, 14143 corpus,
  57 classes, 171 stimuli, 92% AMASS-overlap, 4cm fit, 28 joints) all consistent; TMR bars .478/.370/.431 consistent;
  no over-reach verbs (no "proves/demonstrates that/causes"); "first...to our knowledge" properly scoped+Wilcoxon-backed;
  abstract claims all supported in body (incl "reconstruction capacity" = Dissociation 1 concept); tables/tabular/itemize
  all balanced; removed main.tex.bak_labels from overleaf dir.
- ⚠️ FLAG (not error, plan-dependent): checklist item 1.3 "pedagogical references = yes" is AHEAD of reality until
  Related Work (Funda) is completed; downgrade to "partial" ONLY IF related work stays thin at submission.
- Core bib refs = ORIGINAL papers confirmed: MLD (Chen CVPR23 "Executing Your Commands..."), MAMP (Mao ICCV23
  "Masked Motion Predictors..."), TMR (Petrovich ICCV23). Only page-number %VERIFY remains (deferred by user).

---

## 0. PAPER IDENTITY (center of gravity, as of 2026-07-18)

**TWO contributions, in tension-free order:**
1. **Systems result (mature, months-hardened):** a single self-supervised encoder — masked-motion
   INPAINTER + auxiliary pose head (MAMP+pose) — whose raw, unsupervised embedding distances predict
   human perceptual similarity better than geometric baselines (DTW, quaternion-geodesic) AND a generic
   learned encoder, on ALL THREE actions, under leakage-clean repeated nested CV; a perceptual rank
   fine-tune improves it further. Official-protocol numbers: §3.0. This is the paper's spine + title.
2. **Conceptual finding (hardened this week):** perceptual alignment DISSOCIATES from faithful motion
   representation. Two independent controls: (a) CAPACITY — a plain transformer scaled to the winner's
   size reconstructs better, perceives no better; (b) DECODABILITY — on the discriminating action,
   perceptual alignment is ANTI-CORRELATED with linear pose-decodability (the winner encodes pose LEAST
   linearly-recoverably). ACTION-CONDITIONAL: holds on pointing (localized), absent on walking/picking.

ABSTRACT FRAMING: MAMP = "masked INPAINTING (infer dense motion from sparse ~20% sample)", NOT future
prediction. Do NOT use "abstract" (retired — circular); say "anti-correlated with LINEAR pose-decodability".
Primary-vs-twist framing decision pending final hardening (Attack-2 job 934537); leaning dissociation
promotable to primary now that it's action-conditional + metric-consistent + Attack-4-refuted.

---

## 1. OFFICIAL EVAL PROTOCOL (decision D2-OFFICIAL)

Nested CV: `cv_nested.py --k 5 --repeats 5 --n-triplet-seeds 5 --triplet-epochs 200`, 3-way disjoint
folds per outer fold (TEST / SELECT / TRAIN); report raw Spearman on TEST (n=25 = 5 folds × 5 repeats).

**Checkpoint-epoch selection:** `--epoch-grid 100 --loss-plateau <PLATEAU>` where **PLATEAU = the model's
own reconstruction-loss plateau epoch** — the epoch at which its SSL *validation* reconstruction
(`val_recon`, F.mse_loss in normalized space, 5% held-out split, seed 0) stops improving.
- Candidates = every 100th epoch in `[100, PLATEAU]`. SELECT only CHOOSES within this window.
- PLATEAU is **model-intrinsic + reconstruction-side** — NOT perceptual (avoids SELECT double-use),
  NOT the arbitrary full-trajectory end (training length is a scheduling artifact).
- **REJECTED alternatives:** PSP-bounded (perceptual plateau on SELECT = circular, D3); full-trajectory
  / `--loss-plateau 10000` (arbitrary length). PSP kept ONLY as a §4.4.1 over-training diagnostic.
- **Plateau operational def (uniform):** 3-pt moving-avg smooth of val_recon; asymptote = min over last
  20% of trajectory; PLATEAU = first grid-100 epoch where smoothed ≤ 1.02·asymptote and stays
  ≤ 1.03·asymptote thereafter. Defense = ceiling-invariance (headline ~unchanged across [0,PLATEAU],
  [0,2·PLATEAU], [0,last]).
- **MAMP family** diverges if over-trained → PLATEAU = pre-divergence epoch (same recon/divergence rule).

Training side: each model trained under its OWN SSL objective (recon / masked-pred / +pose) to a fixed
budget, full trajectory saved every 100 ep. Stopping/checkpointing on val_recon ONLY — never perceptual.
(Leakage audit: [[reference_select_leakage_audit]] — no SELECT double-use, no TEST leakage.)

---

## 2. PER-MODEL RECONSTRUCTION-PLATEAU CAPS

val_recon plateau (uniform rule above). MLD family = native val_recon from training logs. Vanilla family
= post-hoc held-out val_recon (job 932960; vanilla trainer logs no val). MAMP = train_loss plateau
(MAMP logs train_loss/epoch, no val; it DOES plateau — last-200ep <2.5% of total drop).

| model | val signal | asymptote | **PLATEAU cap** | source |
|---|---|---|---|---|
| MLD-AE native (query+skips) | val_recon | 0.1263 | **1625** | log aeh_3ds_893876 |
| MLD-AE no-skip | val_recon | 0.1154 | **1770** | log ae_mld_noskip_931697 |
| MLD-AE attn-pool | val_recon | 0.1009 | **2575** | log ae_mld_attnpool_932654 |
| vanilla rot1vel0 (canonical) | post-hoc val_recon | 0.00563 | **7600** | job 932960 |
| vanilla +skips | post-hoc val_recon | 0.00578 | **7500** | job 932960 |
| vanilla querypool | post-hoc val_recon | 0.00415 | **7500** | job 932960 |
| vanilla bigcap (capacity) | post-hoc val_recon | 0.00305 | **1300** | job 932960 (best recon of ANY vanilla, yet no perceptual gain §4.3) |
| MAMP (motion-only) | train_loss | 0.318 | **1000** | log lma_mamp_holdout_1200 |
| MAMP+pose (HEADLINE) | train_loss | 0.225 | **1000** | log lma_mamp_pose_holdout |
| MAMP-skip | train_loss | — | **1000** | log lma_mamp_skip_holdout_1200 |
| MAMP-skippose | train_loss | — | **1100** | log lma_mamp_skippose_holdout_1200 |

NOTE: perceptual peak (~ep240 for MAMP+pose) is EARLIER than recon plateau (~1000) — the NORMAL case
for ALL models (recon keeps improving after perceptual saturates; that gap is WHY we CV-SELECT).
SELECT picks the early peak within [100, cap]; a generous cap that contains the peak is correct.

---

## 3. EVAL COMPLIANCE STATUS (which numbers are official)

### 3.0 ✅ OFFICIAL-PROTOCOL §A SPINE (job 934430 + 934428, 2026-07-18) — grid-100 @ per-model plateau cap
These SUPERSEDE the ad-hoc fine-grid paper numbers AND the ⚠️ full-trajectory rows below. Committed
sbatch: `spineA_rerun.sbatch`, `cvn_bigcap_full.sbatch`. W / **P** / K, n=25:

| model | cap | walking | **pointing** | picking | note |
|---|---|---|---|---|---|
| **MAMP+pose (HEADLINE)** | 1000 | .554±.025 | **.435±.030** | .543±.033 | reproduces paper .552/.434/.543 ✓ |
| MAMP (motion-only) | 1000 | .353±.031 | **.300±.032** | .488±.039 | |
| vanilla-rot1vel0 | 7600 | .476±.030 | **.212±.034** | .492±.029 | |
| MLD-AE (query) | 1625 | .521±.022 | **.325±.040** | .490±.029 | |
| vanilla-bigcap (35M, capacity) | 1300 | .349±.037 | **.221±.037** | .301±.032 | = interim; CAPACITY DISSOCIATION confirmed full-traj |

Still TODO for full compliance: re-run MAMP-skip/skippose, MLD-AE-vel/prov, MLD-VAE, vanilla-rot1vel1/rot0vel1,
+ the mechanism-ablation pairs (skip/pool) at their §2 caps (ceiling-invariant → numbers ≈ current).

⚠️ = ran on `--loss-plateau 10000` (full-trajectory, REJECTED). Numbers ≈ correct (ceiling-invariant)
but NOT official until re-run/re-verified at the §2 plateau cap.

| model | cvn file | job | protocol used | W / **P** / K | status |
|---|---|---|---|---|---|
| MLD native | cvn_mldae_full_results.json | 932421 | plateau-10000 | .532/**.331**/.498 | ⚠️ re-verify @1625 |
| MLD no-skip | cvn_mldnoskip_full_results.json | 932422 | plateau-10000 | .570/**.330**/.513 | ⚠️ re-verify @1770 |
| MLD attn-pool | cvn_mldattnpool_results.json | 932759 | plateau-10000 | .501/**.424**/.497 | ⚠️ re-verify @2575 |
| vanilla (no skip) | cvn_vrot1vel0.json | (old) | ad-hoc fine-grid | .476/**.213**/.491 | ⚠️ re-run @TBD (also §0.6.2 provenance) |
| vanilla +skips | cvn_vanskips_full_results.json | 932504 | plateau-10000 | .471/**.221**/.456 | ⚠️ re-verify @TBD |
| vanilla querypool | cvn_vanquerypool_results.json | 932934 | plateau-10000 | .493/**.216**/.455 | ⚠️ re-verify @7500 |
| vanilla bigcap (35M) | cvn_bigcap_interim_results.json | 932935 | plateau-2400 (interim) | .349/**.221**/.301 | interim (≤2400); re-run @1300 + after training |

**Core §A paper models** (MAMP, MAMP+pose, MLD-VAE, vanilla-rot1vel1/rot0vel1, TMR baselines): all on
uncommitted ad-hoc fine-grid runs (§0.6 audit) → must be re-run under §1 protocol at §2 caps. See §0.6.2.

---

## 4. LOCKED FINDINGS (mechanism ablations)

Numbers above; ceiling-invariance means re-verification at plateau caps will not change these
qualitative findings (pointing especially). Full narrative: [[project_skip_ablation_4corner]],
[[project_swap_pooling]].

1. **Skips perceptually INERT, both architectures.** vanilla P +.008 (skips on), MLD P −.001 (skips off).
   (MLD toggle = encoder+decoder skips together; vanilla = encoder skips; combined ⇒ no skip component helps.)
2. **POOLING × ARCHITECTURE INTERACTION (2×2 COMPLETE).** Pointing:
   |         | attention-pool | latent-query pool |
   |---------|----------------|-------------------|
   | vanilla | .213           | .216 (932934)     |
   | MLD     | **.424** (932759) | .331           |
   - Query-pooling is **INERT in vanilla** (.213→.216) but **HARMFUL in MLD** (.424→.331).
   - Attention-pooling is a **large win ONLY inside MLD** (+.093).
   - ⇒ Pooling matters only when the encoder produces something LOCALIZED worth selecting from.
     Vanilla's encoder apparently does not, regardless of pooling. Contradicts "richer un-averaged z"
     (latent-query was the hypothesized MLD advantage; it is a liability).
   - Grounded in kinematics: pointing frac-active 0.41 / tstd .048 = LOCALIZED; walking 0.88 / .089 =
     DISTRIBUTED → selection helps localized, mildly hurts distributed (MLD walking −.031 fits).
3. **⭐ CAPACITY CONTROL: MLD's edge is NOT degrees-of-freedom (932935 interim).**
   Big vanilla (35.0M ≈ MLD's 31.2M budget; hidden384/heads6/depth7, same rot1_vel0 objective+corpus):
   **pointing .221 ± .037** — IDENTICAL to small vanilla's .213 (9M). 3.9× the parameters bought
   **ZERO** perceptual gain. Walking/picking actually DROPPED (.349/.301 vs .476/.491 — overfitting).
   **Yet it reconstructs BEST of any vanilla** (val_recon asymptote .00305 vs canonical .00563) and
   plateaus fastest (ep1300 vs 7600).
   ⇒ **STRIKING DISSOCIATION: capacity buys better RECONSTRUCTION but ZERO PERCEPTUAL gain.**
   ⇒ The MLD>vanilla pointing gap (~.11) IS architectural/objective-driven, NOT parameter count.
   (Interim: ep≤2400 window; confirm with full-trajectory eval after 932900 finishes.)
4. **⭐ LOCALIZATION PROBE (job 933328, n=25 per encoder×action) — tests the "selectivity" thesis DIRECTLY.**
   Ridge probe z→pose(t) per frame (PCA k=16/m=2, GCV ridge, relaxed n=46 split, padding masked, R²>0 gate).
   - **PRIMARY (pre-registered): DISPERSION (SD of R²-profile = selectivity) FAILED.** Pointing SD flat across
     all 4 encoders (.244/.235/.267/.286, within ~.03 SEM), if anything runs BACKWARDS (worst encoder highest SD).
     **Selectivity-as-dispersion does NOT explain perceptual performance.** ⇒ RETRACT the "localization/selection"
     framing of finding-2's interpretation (the +.093 attn-pool win is NOT because attn-pool is "more selective").
   - **SECONDARY (exploratory): mean R² (RETENTION) separates by ARCHITECTURE, not pooling.** MLD retains more
     linearly-decodable pose info than vanilla (walk +.12, point +.06, pick +.05); within-MLD attn≈query,
     within-vanilla attn≈query. ⇒ retention is an ENCODER property — matches "gap is architectural" (#3).
   - **⚠️ RESIDUAL UNEXPLAINED:** mean R² does NOT track the within-MLD pooling gap (MLD-attn .620 ≈ MLD-query .617
     on pointing despite +.09 perceptual gap). So the probe explains MLD-vs-vanilla but NOT attn-vs-query-within-MLD.
   - Interpretive: R²-profile correlates w/ active_joints+motion_var (~.69/.67) on WALKING across all encoders;
     collapses on POINTING → decodable frames are action-dependent (pointing = sparser regime).
5. **⭐⭐ THE HEADLINE CONCEPTUAL FINDING (hardened 2026-07-18, jobs 934536/934537): ACTION-CONDITIONAL DISSOCIATION.**
   On POINTING (the localized/sparse DISCRIMINATING action), perceptual alignment is ANTI-CORRELATED with
   linear pose-decodability — nearly MONOTONIC:
   | encoder | pointing perceptual P | pointing pose-decodability R² |
   |---|---|---|
   | MAMP+pose (winner) | .434 | **.509** (2nd lowest) |
   | MLD-attn | .424 | .548 |
   | MLD-query | .331 | .630 |
   | vanilla-query | .216 | .657 |
   | vanilla-attn (worst) | .213 | **.727** (highest) |
   The BEST perceiver decodes pose LEAST; the WORST decodes it MOST. On WALKING/PICKING (distributed/easy
   actions) pose-decodability is FLAT across encoders (~.23-.27 / ~.43-.53) and does NOT track perception —
   NO dissociation. So the effect is SPECIFIC to the discriminating action (grounded in kinematics: pointing
   frac-active .41 localized vs walking .88 distributed). Metric = standardized ridge probe (per-dim target
   standardization, PCA k16/m2, relaxed n=46, padding masked, per-fold SELECT epoch). Attack-4 REFUTED:
   vel/acc decodability negative for ALL encoders (nobody linearly encodes per-frame dynamics; Attack-4 refuted).
   Attack-2 (nonlinear MLP probe, job 934537 DONE): ALL encoders are ROW B — nonlinear probe lifts pose-R2 by
   ~.10-.15 (MAMP+pose .509→.645; vanilla .727→.829) AND PRESERVES ORDERING (MAMP+pose still lowest, vanilla
   highest). ⇒ (i) anti-correlation is NOT a linear-probe artifact; (ii) pose IS present in all (winner did NOT
   discard pose) — just LESS LINEARLY-ACCESSIBLE. Since perceptual metric = L2 (linear), only linearly-accessible
   pose is metric-visible. FINAL CLAIM: "the winning encoder makes per-frame pose the LEAST LINEARLY-ACCESSIBLE
   (though comparably present nonlinearly); faithful pose, where present, is arranged out of the L2 metric's reach."
   ⚠️ WORD "abstract" RETIRED (undefinable/circular). ⚠️ WALKING-flatness has NO defensible mechanism
   (periodicity/padding/cross-clip-variance ALL FALSIFIED, action_structure.py) — frame ONLY as "pointing is the
   sole action with cross-encoder decodability SPREAD; there the anti-correlation appears; walk/pick uniform hence
   untestable." NOTE: absolute R2 LEVEL per action ≠ the DISSOCIATION (= between-encoder SPREAD within an action);
   different measurements. Don't claim WHY walk/pick are uniform.
6. **SYNTHESIS STATE:** perceptual gap is ARCHITECTURAL (not skips #1, not capacity #3, not pooling-per-se #2).
   Two DISSOCIATIONS establish perception ≠ faithful representation: (a) capacity buys reconstruction not perception;
   (b) on pointing, perception ANTI-correlates with linear pose-decodability. Retention "bridge" MLD→MAMP+pose
   FAILED (#4). The within-MLD attn>query gap remains UNEXPLAINED — report honestly, do NOT force a mechanism.

---

## 5. OPEN ACTIONS before July 28 submission

- [x] Vanilla plateau caps (932960) → §2 filled (7600/7500/7500/1300).
- [x] Capacity result (932935 interim) → §3/§4.3. Big-vanilla training 932900 TIMEOUT@ep9091 (converged) → full-traj eval PENDING.
- [x] Localization probe (933328) → §4.4. Primary FAILED, secondary=retention-by-architecture.
- [x] MAMP caps: NO special-casing needed (I was wrong about a "tension"). MAMP train-loss DOES plateau
      (+pose: last 200ep = 0.3% of total drop; motion-only: 2.2%) → cap at train-loss plateau ~ep1000 like every
      model. The perceptual peak (~240) being EARLIER than the recon plateau is the NORMAL case (true for all
      models; it's WHY we CV-SELECT); SELECT just picks the early peak within [100,plateau]. No divergence-cap,
      no SELECT-circularity. MAMP fits D2 identically.
- [ ] Re-run/re-verify all §3 ⚠️ rows at §2 caps; update numbers + mark compliant.
- [ ] Re-run §A core spine (§0.6.2) under §1 protocol; commit sbatch each.
- [ ] Full-trajectory bigcap eval (training now done) to confirm interim .221.
- [ ] main.tex §6.8 rewrite: skips inert + retention-architectural + NO localization claim.
- [ ] Rewrite `repro_nested_cv/` README to §1 protocol (drop PSP-as-selector); sync frozen code to
      current (use_skips/use_pool/capacity-aware).
- [ ] Reconcile main.tex §6.8 (pre-ablation skip framing) with §4 findings (see ledger 0.6.1 flag).

---

## 12. ⭐⭐ TMR PERCEPTUAL FINE-TUNE RESULT (job 938824 COMPLETED 2026-07-24, exit 0, 4h10m, n=25)

**SYMMETRIC control for the "beats TMR" claim — CHANGES THE STORY. Handle honestly.**
Fine-tuned TMR's ACTOR motion encoder with the IDENTICAL rank objective + nested-CV as MAMP+pose.
Validated: 25/25 folds, 0 NaN, early-stop ep 15–119 (real training), per-fold spreads normal.
Code: `tmr_perc_finetune.py` + `dperc_folds.py`; results `tmr_perc_finetune_results.json`.

| method | walking | pointing | picking |
|---|---|---|---|
| raw TMR | .452 | **.425** | .408 |
| **fine-tuned TMR (NEW)** | .649±.025 | **.573±.037** | .611±.035 |
| MAMP+pose raw | .462 | .302 | .481 |
| MAMP+pose + fine-tune (rankdiag_headline) | **.699** | .463 | .605 |

**SYMMETRIC (fine-tuned vs fine-tuned) verdict:**
- Walking: MAMP+pose .699 > TMR .649 (MAMP+pose wins)
- Picking: MAMP+pose .605 ≈ TMR .611 (TIE, TMR marginally higher)
- **Pointing: TMR .573 > MAMP+pose .463 — TMR WINS pointing decisively (~3 SE).**

⇒ The paper's "fine-tuned MAMP+pose surpasses TMR on ALL THREE" was fine-tuned-MAMP+pose vs **RAW** TMR (unfair, as user suspected). Under MATCHED fine-tune, **TMR is NOT surpassed on all three**: it beats MAMP+pose on pointing and ties picking. Semantic pretraining, given equal perceptual supervision, makes TMR the stronger POINTING perceiver — CONSISTENT with the dissociation (pointing = semantic/config-sparse action where abundant supervision helps; §6.10 already says "pointing = the one action where abundant semantic supervision helps most").
⚠️ AFFECTS: §6 TMR subsection claim, abstract sentence "surpasses TMR on all three actions" (SUBMITTED abstract — can't change submitted, but BODY must not repeat it; body already editable to Jul 28), TL;DR. The raw-vs-raw comparison + "alignment over abundance" framing NEEDS revisiting — the honest read is now "matched perceptual supervision: MAMP+pose wins walking, ties picking, loses pointing to TMR." STILL A STRONG PAPER: MAMP+pose is a single SELF-SUPERVISED encoder competitive with (winning 1, tying 1 vs) a large SEMANTICALLY-SUPERVISED model even after both get perceptual fine-tune, and wins outright in the RAW/unsupervised regime on walking+picking.
✅ RESOLVED 2026-07-24 (Option B, user-approved): PAPER UPDATED. (1) ABSTRACT (main.tex, now 2101 chars) — replaced "surpasses TMR on all three" with honest "when TMR receives the same perceptual fine-tuning, our self-supervised encoder still matches or exceeds it on two of three actions, while TMR wins the third (pointing, where semantic supervision helps most)". (2) §6 TMR table — added "after identical perceptual fine-tuning" block with fine-tuned-TMR row (.649/.573/.611); bolding = within-regime winner. (3) §6 narrative — rewrote to two regimes (raw: MAMP+pose wins 2/3; matched fine-tune: wins walking, ties picking, loses pointing) tied to dissociation. (4) Removed "alignment over abundance" overclaim + the "future work" limitation para (experiment now DONE). Contributions/summary "all three" claims verified SAFE (they're about geometric+SSL baselines, not TMR). Tables balanced 12/12, refs resolve.

---

## 10. COMPUTE INFRASTRUCTURE (added 2026-07-23, for checklist 4.7 + Method) — chimera (UMass Boston)

VERIFIED from chimera 2026-07-23:
- **Cluster:** chimera.umb.edu, Slurm. **OS:** Rocky Linux 8.10 (Green Obsidian). Head/login node CPU: Intel Xeon Silver 4210R @2.40GHz, 40 CPUs, 187 GiB.
- **GPU partitions (sinfo):** DGXH200 (gpu:h200:8, 2063936 MB node) — the H200 partition used to dodge the impact billing cap; DGXA100 (gpu:A100:8, 2000000 MB); A30 (gpu:a30:5); plus MIG slices (2g.35gb, 1g.35gb) on DGXH200. CPU partitions: Intel6126/6240/2650/6248/6326.
- Paper training/eval ran on **H200 (impact)** via pomplun/Funda account (see [[reference_chimera_encode_recipe.md]]).
- **Training env = `torch_gpu_cu12`** (VERIFIED 2026-07-23; NOT gpu_env, which lacks torch). Sbatch files `conda activate torch_gpu_cu12`.
  Python 3.10.16; **torch 2.4.1+cu121, CUDA 12.1, cuDNN 9.1.0 (90100)**; torchvision 0.19.1+cu121; numpy 2.2.6; scipy 1.15.3; scikit-learn 1.7.2; pandas 2.3.3; einops 0.8.2; timm 1.0.22; dtaidistance 2.4.0 (DTW baseline); PyYAML 6.0.3; matplotlib 3.10.8. Kernel Linux 4.18 (Rocky 8.10).
- **TMR fine-tune env = `tmr`** (separate): Python 3.10.20, torch 2.5.1+cu121, cuda12.1, cudnn 9.1, hydra-core 1.3.3, pytorch-lightning 2.6.5, transformers 5.14.1 (+ pandas/sklearn/matplotlib installed 2026-07-23).
- 4.7 = yes once the Method §Infrastructure paragraph lands (drafted 2026-07-23).

---

## 11. TMR CHIMERA LAYOUT (added 2026-07-23, for full TMR fine-tune feasibility)

At `$TR/learned_baselines/tmr/` ($TR=/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets):
- `repo/` = FULL official TMR (Petrovich): encode_motion.py, encode_dataset.py, demo/{model,load}.py, src/{load,config,rifke,geometry,joints}.py, train.py, retrieval.py. Encoder code IS present ⇒ full fine-tune is architecturally possible.
- `encoded/` = the 171 frozen 256-d .npy (what tmr_nested.py evaluates).
- `models/` = (contents NOT yet listed — likely ckpt home).
- SMPL pipeline: `learned_baselines/bvh2tmr_pipeline/joints2smpl` + `joints2smpl_probe`.
RESOLVED 2026-07-23 (2nd probe):
- **Encoder ckpt:** `models/models/tmr_humanml3d_guoh3dfeats/last_weights/motion_encoder.pt` (19M) — already extracted from the Lightning ckpt (src/load.py:extract_ckpt splits state_dict into motion_encoder/text_encoder/motion_decoder .pt). ⇒ loadable as plain nn.Module, NO Lightning loop needed.
- **Input format:** HumanML3D **Guo 263-dim feats** (`guoh3dfeats`); normalizers `repo/stats/humanml3d/guoh3dfeats/{mean,std}.pt`. motion_encoder = transformer [T,263]→256-d (demo/model.py:48 forward). Model built via Hydra `instantiate(cfg.model)` from config.json (3.8K).
- **BVH→263 pipeline (deterministic, non-diff front-end):** `bvh2tmr_pipeline/scripts/{bvh_to_smpl22.py→fitted_to_263.py}` + `joints2smpl/fit_seq.py`; guofeats code in `bvh2tmr_pipeline/tmr_guofeats/motion_representation.py`.
- **FINE-TUNE PLAN — FULLY RESOLVED (2026-07-23, all probes done):**
  - Fixed 263-feat inputs EXIST: `bvh2tmr_pipeline/scaled/feats263/<Action_e0_e1_e2_e3>.npy` [T-1,263], 171 files, guofeats pre-norm. SMPL fits frozen upstream (`scaled/fit_out/*/*.pkl`). ⇒ differentiable from 263-feats; NO re-fit.
  - Encoder = `ACTORStyleEncoder.forward({"x":[B,T,263],"mask":[B,T]}) -> [B,nbtokens,256]`; latent via `model.encode(collate_x_dict([{"x":normalizer(motion),"length":len}]), sample_mean=True)[0]`. Non-VAE branch → deterministic `(latent,)=encoded.unbind(1)`. Fully differentiable transformer.
  - Load: `load_model_from_cfg(read_config(run_dir), ckpt_name="last")`, `run_dir=.../tmr/models/models/tmr_humanml3d_guoh3dfeats`; normalizer via `instantiate(cfg.data.motion_loader.normalizer)`.
  - sbatch conventions: account `pi_funda.durupinarbabur`, partition A30 (or DGXH200), gres gpu:1, slurm_logs/. TMR repo needs its own conda env (hydra/pytorch-lightning/transformers) — env name TBD (probe B).
  - Script mirrors `perc_finetune.py`: same triplet rank loss + margin=d_perc gap, SAME nested-CV folds (make_folds seed 42+1000·rep), low-LR + early-stop on SELECT, eval TEST. Fine-tune the ACTOR motion_encoder (freeze text branch). Output = tmr_perc_finetune_results.json, compare to MAMP+pose rankdiag_headline.
  - ✅ **END-TO-END SMOKE PASS 2026-07-23** (`tmr_perc_finetune.py` + `dperc_folds.py`, chimera `tmr` env, MOTION_IS_REMOTE=1): load→encode→rank-loss(0.20)→BACKPROP into motion_encoder(grad 0.0047)→spearman_eval all work. Integration hurdles SOLVED: (1) `src` namespace collision between MS and TMR repos — fixed by evicting `src*` from sys.modules + removing MS paths during TMR load in load_tmr_encoder, then restoring; (2) MS d_perc/fold helpers extracted to dep-light `dperc_folds.py` (no src/pymo/tf chain); (3) `tmr` env needed pip: pandas, scikit-learn, matplotlib, transformers (installed); (4) read_config from src.config not demo.model (avoids eager transformers); (5) Config needs MOTION_IS_REMOTE=1 for chimera paths. REMAINING: write sbatch (tmr env, A30/H200, MOTION_IS_REMOTE=1), launch 5×5 nested-CV, compare fine-tuned-TMR vs fine-tuned-MAMP+pose.

---

## 9. PAIRED WILCOXON SIGNIFICANCE (added 2026-07-23, for checklist 4.11)

One-sided paired Wilcoxon signed-rank over n=25 matched folds (pair by (repeat,fold)); H1: MAMP+pose > comparator.
Code: `repro_bundle/05_code/wilcoxon_paired.py` (pure-stdlib impl — no scipy locally; validated on rankdata-ties + constant-shift + symmetric-null cases). Results: `repro_bundle/03_results/wilcoxon_paired_results.json`.
Pairing VALIDITY: baselines_nested.py L84-96 emits rows rep-major/fold/action via SAME `seed+1000*rep` make_folds as models ⇒ positional pair == (repeat,fold) pair.

| comparator | walking p | pointing p | picking p |
|---|---|---|---|
| DTW (geometric)   | .022 *   | .0020 **  | 3.1e-5 *** |
| geodesic (geom)   | 4.8e-5 ***| 1.2e-5 ***| 1.9e-5 *** |
| vanilla (learned) | .0039 **  | 9.4e-6 ***| .0053 **   |
| MLD-AE            | **.13 ns**| .0018 **  | .0026 **   |
| MAMP              | 2.7e-5 ***| 6.2e-4 ***| .0067 **   |

- ✅ HEADLINE INTACT: MAMP+pose beats BOTH geometric baselines AND the learned/vanilla baseline (the headline table's comparators, main.tex L484-490) on ALL 3 actions, ALL significant. Walking-vs-DTW p=.022 (was hand-waved "≈1.6σ near-marginal, independent-SEM"; Wilcoxon is STRONGER + proper). main.tex L494-501 updated to cite Wilcoxon.
- ⚠️ HONEST TIE (NOT a claimed win): **MAMP+pose vs MLD-AE on WALKING = p=.13 ns (Δ=+.012)**. MLD-AE is the reconstruction CONTROL rung (carried on deterministic-parity grounds, NOT performance — [[project_mld_ae_vs_vae_justification]]); MLD-VAE actually BEATS MAMP+pose on walking (.597 vs .552, already the paper's openly-stated position). So the abstract's "reconstruction-based transformer encoders across three actions" MUST NOT be read as "beats MLD-AE/VAE on walking" — the paper does not claim that. Body headline "learned baseline" = VANILLA (significant all 3), not MLD. Do NOT let abstract/body imply an all-3 win over the MLD rung.
- Bonferroni (6 geometric cells, .05/6=.008): all pass EXCEPT walking-vs-DTW (.022). Pointing/picking (decisive axes) all p≤.002.

---

## 8. PERCEPTUAL-DATA PROVENANCE & DE-IDENTIFICATION (added 2026-07-23, for checklist 3.2/3.3)

Chain: **`../R/validAnswers.csv` (17,380 raw MTurk responses) → `../R/lmaPilot.R` → `aux/*_similarity_comparisons_ratios.csv`** (the aggregate d_perc that serves ALL supervision/eval). `../R/` is OUTSIDE the motion-similarity git repo.
- Raw cols: `timestamp,id,hitId,qInd,motionType,effortsLeft,effortsRight,selected0,selected1`. Identifiers = `timestamp`, `id` (424 distinct), `hitId` (110 distinct, MTurk) → IRB/consent-governed.
- **VERIFIED (2026-07-23): R output == committed aggregate.** `R/similarity_comparisons_ratios.csv` vs `aux/similarity_comparisons_ratios_11_17_23.csv`: 4620/4620 keys match, **0** count_normalized mismatches. ⇒ lmaPilot.R provably regenerates the supervision aggregate.
- **De-identified raw** built at `R/deidentified/responses_deid*.csv` (dropped `timestamp,id,hitId`; kept qInd/motionType/efforts*/selected*). **Round-trip VERIFIED (job bdktoe0vf, 2026-07-23): deid input → lmaPilot transform → IDENTICAL:TRUE, 4620/4620 rows.** Scrub is provably INERT ⇒ safe to ship. Checklist 3.2 flipped partial→YES.
- **Release plan (user decision 2026-07-23): ship de-identified raw + lmaPilot.R + aggregates** in Code/Data Supplement → checklist 3.2 → yes, 3.3 yes. Raw `validAnswers.csv` (with identifiers) NOT released.

---

## 7. PEARSON SUPPLEMENT (added 2026-07-20)

Spearman remains PRIMARY (ordinal ground truth); Pearson is the magnitude-alignment supplement
(paper §2.6 already promises it). Added eval-side ONLY — NO retraining (Pearson is a scoring choice
on the same embedding-distances vs d_perc).
- Code: `cv_preliminary.py:raw_both()` computes (spearman, pearson) from IDENTICAL pairs;
  `cv_nested.py` stores `pearson_test` per fold + `aggregate.pearson`. ADDITIVE — SELECT path
  (raw_spearman) untouched. `.bak_pearson` saved.
- GATE VERIFIED (job 936023, MLD-AE @1625): raw/Spearman = .5208/.3247/.4901 == spineA .521/.325/.490
  EXACT → patch confirmed additive. Pearson (MLD-AE) = .537/.342/.535.
- Tier-1 batch (job 936037, official caps + Pearson): MAMP+pose@1000, MAMP@1000, vanilla-rot1vel0@7600.
  MLD-AE Pearson from smoke (pearson_mldae_results.json).
- Tier-2 (DEFERRED to paper week — need per-model val_recon plateau caps derived properly):
  MLD-VAE, MLD-AE-vel/prov, vanilla-rot1vel1/0vel1, MAMP-skip/skippose. Their training val_recon logs
  not located tonight; derive via post-hoc val_recon pass (like job 932960 for vanilla) before reporting.
- NOTE: Pearson is NOT an abstract number (abstract = Spearman). Not deadline-critical for Jul 21.

### 7.1 TIER-1 PEARSON RESULTS (job 936037 DONE 2026-07-21, official caps) — Spearman == spineA (patch additive, re-confirmed)
| model | Spearman (W/P/K) | Pearson (W/P/K) |
|---|---|---|
| MAMP+pose (headline) | .554/.435/.543 | .569/.472/.604 |
| MAMP | .353/.300/.488 | .341/.336/.543 |
| vanilla-rot1vel0 | .476/.212/.492 | .494/.206/.548 |
| MLD-AE (from smoke 936023) | .521/.325/.490 | .537/.342/.535 |
Pearson TRACKS Spearman (same encoder ordering; close magnitudes) → corroboration, not a new story.

### 7.3 ⭐ MAMP+pose vs TMR — TESTED (two-sided paired Wilcoxon, n=25, 2026-07-24)
Code `repro_bundle/05_code/wilcoxon_mamp_vs_tmr.py`; per-fold sources = cvn_mamppose_cut1 (raw MAMP),
tmr_raw_perfold.json (raw TMR, from patched tmr_nested.py), rankdiag_headline['B'] (ft MAMP),
tmr_perc_finetune_results (ft TMR). Results `wilcoxon_mamp_vs_tmr_results.json`:
| regime | walking | pointing | picking |
|---|---|---|---|
| RAW | .462 vs .452 **TIE p=.85** | .302 vs .425 TMR p=.007 | .481 vs .408 MAMP p=.011 |
| FINE-TUNED | .699 vs .649 MAMP p=.026 | .463 vs .573 TMR p=.003 | .605 vs .611 **TIE p=.98** |
⚠️ CORRECTED OVERCLAIM: raw "better on TWO of three" was FALSE — raw = picking WIN, walking TIE, pointing LOSS.
HONEST UNIFIED FRAME (now in abstract + §6): "matches or exceeds TMR on the two kinematically rich actions
(walking, picking), trailing only on pointing" — TRUE in BOTH regimes (raw: picking exceeds/walking ties;
ft: walking exceeds/picking ties). Abstract 2107 chars. §6 states per-cell p-values. Both artifacts in archive.

### 7.2 ⭐ REPORTING DECISION (d_perc-aware, user-raised 2026-07-23):
d_perc IS bounded-continuous ([0,1]) so Pearson is meaningful on the TARGET side. BUT the embedding-L2↔d_perc
relationship need not be LINEAR (could be monotone-but-curved/saturating); Spearman needs only MONOTONICITY =
the weaker, safer assumption matching our actual claim ("closer embedding ⟹ judged more similar").
⇒ **Spearman stays PRIMARY in main tables; Pearson goes in an APPENDIX/supplementary table (or one sentence:
"Pearson yields the same encoder ordering; see Appendix"), NOT a co-headline.** Value of including it:
(a) fulfills the §2.6 promise (currently unmet), (b) robustness — Pearson agreeing shows result isn't a
rank-transform artifact. Honest caveat to state: Pearson assumes linearity of the distance↔d_perc relation,
which we don't claim; hence Spearman-primary, Pearson-confirmatory. Do NOT let Pearson become a co-headline.

---

## 20. PAGE-8 OVERFLOW FIX + TECHNICAL APPENDIX (2026-07-26)

**Problem:** flow-pass prose additions pushed body §10 (Reproducibility) onto page 8 =
references-only zone (AAAI-2027: pp8-9 refs only). Compounded by a STRUCTURAL bug: Pearson
`\appendix` was placed AFTER `\bibliography` (body content in the ref zone).

**Fix (main_2.tex):**
- Pearson appendix REMOVED from body -> moved to standalone `supplement.tex`.
- §10 Reproducibility collapsed 2 subsections -> 1 tight paragraph (recovers ~4-5 lines);
  every fact preserved (5x5 CV, seeds, checkpoint sel, 200-ep, Rocky8.10, A100/H200, versions).
- `\bibliography` is now the FINAL content before `\end{document}` (refs correctly begin p8).
- Body line 291 `Appendix~\ref{app:pearson}` -> "supplementary material" (label now cross-doc; would ??).

**Technical appendix (`docs/overleaf/supplement.tex`) — standalone, compilable, anonymized:**
Full AAAI preamble + class. Purpose = reproducibility detail that doesn't fit 7pg; NOTHING
load-bearing (reviewers not required to read). Sections, all ledger-verified:
  §1 Checkpoint-selection protocol (plateau rule §§1-2 above verbatim: val_recon, 1.02/1.03x,
     grid-100, ceiling-invariance, SELECT-not-double-used).
  §2 Capacity control (35M hidden384/heads6/depth7; .00563->.00305; pt .221 vs .213; wk .476->.349;
     pk .491->.301) — numbers = §4-item-3 above + body §8.2.1, cross-checked identical.
  §3 Decodability probes (ridge k=16 PCA/GCV, MLP control .509->.645 & .727->.829 ordering preserved,
     vel/acc R2<0, pointing-only) — = §4-item-5 above + body §8.2.2, cross-checked identical.
  §4 Pearson table (unchanged from the ex-body appendix).

**Compile status:** cannot compile locally (missing newtxtext.sty, AAAI dep; Overleaf has it).
User compiling on Overleaf to confirm p-count. If §10 still nudges p8, next lever = trim flow prose
(Intro bridge / §8 synthesis most compressible w/o losing bridges or numbers).

---

## 21. §7.4 TRIPLET-LOSS NOTATION FIX + SPACE TRIM (2026-07-26)

**Loss definition (§7.4) corrected against code** (`repro_bundle/05_code/perc_finetune.py`
`_triplet_rank`, lines 165-200). Prior body used bare i/near/far/j/k/p_j/p_k with NO definitions.
KEY SEMANTIC POINT (user caught): pj=pmap[(i,j)]=d_perc(i,j) — the dissimilarities ARE
anchor-relative. So near/far are defined by d_perc(i,j) vs d_perc(i,k), NOT free properties of j,k.
Rewrote: anchor i; near/far = perceptually closer/farther of j,k TO THE ANCHOR
(d_perc(i,j) vs d_perc(i,k)); margin m=|d_perc(i,k)-d_perc(i,j)| = the anchor-relative gap.
Now matches code exactly + hyperparameter-free framing preserved.

**Space trims (user: over 7pg via Repro tail clause):** all pure-redundancy, zero numbers/claims/bridges:
- Repro tail "which track the Spearman rankings throughout... supplementary material" -> "track the
  Spearman rankings, appear in the supplement".
- §3 eval-metrics "provided in the supplementary material as supplementary magnitude metrics" (dup
  word + dup of §10 pointer) -> "reported in the supplement as magnitude metrics".
- §7.2 filler "Placing MAMP+pose against... situates this result within the wider design space" ->
  "Table 8 situates MAMP+pose against the full set of encoders and objectives considered above".
- §7.5 "substantially increases correlation scores across all action types" -> "increases correlation
  across all actions".
Net: ~5-6 lines recovered vs ~4 added by the loss rewrite. Awaiting user Overleaf p-count confirm.

---

## 22. §7 TMR/FINE-TUNE CONSOLIDATION + TABLE 10 DROP + L-EQ NARROWED (2026-07-26)

Space recovery, no numbers/claims changed:
- **Merged §7.3 (TMR) + §7.4 (fine-tune) -> one subsection** "Perceptual supervision and comparison
  to TMR". Loss DEFINITION now PRECEDES the results (was after). Both labels kept
  (sec:tmr + sec:finetune) so all cross-refs (lines 182/286/739) still resolve.
- **Dropped Table 10** ("Fine-tuning impact on MAMP+pose", raw .462/.302/.481 -> FT .699/.463/.605):
  EVERY value already in Table 9 (TMR table, MAMP+pose Raw + FT rows). The +.237/+.161/+.124 deltas
  now stated inline in prose instead. Zero information lost.
- **L equation narrowed** (was bleeding right past column): factored the two embedding distances into
  d_near/d_far shorthand so the displayed line is just relu(d_near - d_far + m); the near/far and m
  definitions moved to the following clause. Same math.
Awaiting user Overleaf p-count confirm.

---

## 23. α-BLOCK ALIGNED TO RANK LOSS + L/m RE-EXPRESSED IN i,j,k (2026-07-26)

Paper hit exactly 7 pages (user confirmed). Two consistency fixes:
- **§3 α block removed from main text.** It promised "fine-tuning uses ordinal ranking losses derived
  from d_perc" then defined neutral-referenced alphas α(0→2),α(2→0) — but §7 rank loss uses ONLY
  d_perc gaps, NOT alphas (alphas = separate α-arm, not in main text). Replaced with 1-line pointer:
  "same d_perc targets drive §7.x fine-tuning, adaptive margins = gaps between d_perc values."
  (also a minor space save). Live alphas now gone; only commented-out dead blocks remain.
- **§7 loss L,m re-expressed in i,j,k** (user found near/far hid the indices). Now: anchor i, order
  j,k so d_perc(i,j)<d_perc(i,k) [j nearer, k farther]; L=relu(d̂(i,j)-d̂(i,k)+m),
  m=d_perc(i,k)-d_perc(i,j) (≥0 by ordering, abs-bars dropped — matches code). Every symbol traceable
  to i,j,k. Same math as before, just index-explicit + shorter (no overflow).

---

## 24. REPRODUCIBILITY-CHECKLIST HONESTY AUDIT (2026-07-28, submission day)

Audited docs/ReproducibilityChecklist.tex for truthfulness vs actual artifacts. Fixes:
- Template class `aaai2026` → `aaai2027` (line 15; was a copy-paste bug).
- **Q4.1** (number/range of hyperparam values tried + selection criterion): yes → **partial**. We document
  FINAL settings + checkpoint-selection criterion (100-ep grid, plateau rule) + arch/objective sweep, but
  NOT a per-hyperparameter range-tried log. "partial" is the honest, unimpeachable answer.
- **Q4.12** (all final hyperparameters listed) = KEPT "yes" (user decision) — but only honest because the
  deferred code archive ships them. VERIFIED 2026-07-28: `code_data_supplement_aaai2027.tar.gz` contains
  `02_encoders/configs/{lma_mamp,lma_mamp_pose,lma_mamp_uencfull,lma_mamp_uencfullpose}_pretrain.yaml`
  each with full set (depth8/heads8/mlp4/dim256/tpatch4/pose_weight1.0/epochs600/warmup20/bs64/blr1e-3/
  min_lr5e-4/wd0.05/mask0.80/tau0.75) + tmr_perc_finetune.sbatch. Memory [[project_checklist_yes_commitments]]
  holds the pre-final-upload gate.

Verified HONEST as-is (no change): theoretical=no; 1.1-1.3 yes; 3.x dataset (de-id round-trip inert,
external sets public+cited); 4.3 partial (GB .pth referenced-not-copied); 4.4-4.11 (seeds 42+1000·rep,
A100/H200+PyTorch vers, Spearman metric+rationale, n=25, ±SE/CIs, Wilcoxon). Today's upload = paper +
checklist; code/data archive within AAAI +2-3 day supp window.

---

## 25. SUPPLEMENT PEARSON/SPEARMAN MISMATCH FIXED (2026-07-28, Funda catch)

Funda flagged: supplement's Spearman column ≠ main paper's Spearman. CAUSE: supplement's
Pearson/Spearman table was built from the Pearson-companion run (spineA / job 936037), whose Spearman
(MAMP+pose .554/.435/.543, MLD-AE .521/.325/.490, MAMP .353/.300/.488, vanilla .476/.212/.492) differs
by run-to-run/checkpoint variation from the CANONICAL cvn tree the MAIN PAPER tables use (MAMP+pose
.552/.434/.543, MLD-AE .532/.331/.498, MAMP .351/.299/.490, vanilla .476/.213/.491). MLD-AE was the
largest gap (.521 vs .532; Pearson run was MLD-AE@1625 plateau ckpt).

FIX (user decision): DROP the Spearman column from supplement — show Pearson r ONLY. Avoids
duplicated-but-mismatched ρ. Pearson r kept (validly paired w/ its own run's Spearman via raw_both
IDENTICAL pairs, ledger §7). Caption reframed: "Pearson r ... as a magnitude counterpart to the primary
Spearman results in the main paper; preserves the same encoder ordering." Body already promises exactly
this ("Pearson coefficients ... in the supplement as magnitude metrics"). No main-paper number touched.

## 26. ⭐⭐ POOLING TWO-FAMILY MAP — VERIFIED ON CHIMERA (2026-07-28, GPT catch)

GPT flagged: eval-setup says clip embeddings are mean-pooled, but §5.5/probe calls the .213 baseline
"attention pooling" — contradiction. RESOLVED by direct chimera verification (NOT inference — three
successive inferences were each disproven mid-audit; only torch state-dict inspection settled it).

**Chimera repo root:** `/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets`
(SLURM WorkDir of jobs 932654/932655; login `ssh -tt p.bendiksen001@chimera.umb.edu`; venv
`conda activate torch_gpu_cu12`).

**VERIFIED POOLING MAP (each encoder's EVAL readout):**
| Encoder (paper) | Eval pooling | Verified by |
|---|---|---|
| MAMP, MAMP+pose, MAMP-U-Net | **MEAN** | `MAMP*/encode_mamp_lma.py` `feats.mean(dim=1)`, `--pool default="mean"` (all 3 dirs: MAMP/, MAMP_dualdec/, MAMP_uencfull/). Matches Fig 1B "↓ mean pool" ✓ |
| Vanilla plain transformer (.213) | **ATTENTION** | ckpt `checkpoints/v3_ablation_rot1_vel0/cp_rot1_vel0_8400.pth` — **torch-loaded `model_state_dict` CONTAINS `attention_weights.weight`** (the `AETransformer` attn-pool head: `Linear(hidden,1)→softmax over T→weighted avg`), NO `latent_query`. |
| MLD-AE / MLD-VAE recon | **ATTENTION** | same `autoencoders/AETransformer.py` class, trainer `--pool default="attn"` (`train_AE_ablation_with_velocity_curriculum.py`); ckpts under `checkpoints_ae_mld/v0_ae_3ds_seed0` |
| querypool ablation (.216) | latent-query | `training/van_querypool.sbatch` `--pool query`, job **932655** (`van_querypool`), `v3_ablation_rot1_vel0_querypool` |
| attnpool ablation (MLD .424) | attn | `training/ae_mld_attnpool.sbatch` `--pool attn`, job **932654** (`mld_attnpool`), `checkpoints_ae_mld/v0_ae_3ds_seed0_attnpool_traj` |

**Pooling head lives in `autoencoders/AETransformer.py`:** `use_pool` arg (attn|query); `attn` builds
`self.attention_weights = nn.Linear(hidden_dim,1,bias=False)`, `encode()` L119-124 = softmax over time
× frame vectors summed. `query` builds `self.latent_query` param. Trainer default=`attn`; vanilla
baseline sbatch `training/van_rot1_vel0.sbatch` passes NO `--pool` ⇒ default attn. Encode dispatch infers
head from ckpt keys (`probes/encode_aetransformer.py` L49, `probes/encode_sweep.py` L73).

**⚠️ TRAP that cost time:** a `.pth` zip-member listing does NOT expose pickle key names — it showed
zero pooling keys and falsely implied "no head." MUST `torch.load(...)["model_state_dict"].keys()` in
the venv. Also the head attribute is `attention_weights` NOT `attn_pool` (that name is only the MLD
`AEMotionMLD` head); grepping `attn_pool` on the AETransformer ckpt gives a false negative.

**PAPER FIX (main_2.tex, both provably true):** (1) L331 universal "mean-pooling" → "each encoder's
native readout — mean pooling for the MAMP masked-autoencoder family and the trained attention-pool head
for the reconstruction autoencoders." (2) L434 scoped to "the MAMP family" + "reconstruction autoencoder
baselines are read out through their trained attention-pool head instead." **Probe's "attention pooling"
label on the .213 baseline is CORRECT and was LEFT UNCHANGED** (it really is attention-pooled).
Fig 1B unchanged (correct: depicts MAMP+pose = mean).

**NOT done (deferred to camera-ready):** exact pooling of the perceptual `+FT` rows. FT script =
`MAMP_dualdec/main_finetune.py` (+ cv_nested harness), NOT the repro-bundle `perc_finetune.py` `AttnPool`
variant. Could not confirm FT readout to this thread's standard under deadline; numbers are provenance-
tracked and no paper text asserts FT pooling, so omission is safe. Re-verify before any FT-pooling claim.

## 27. THREE GPT TABLE FIXES (2026-07-28): T7 factorial completed, MLD-VAE described, T6 ✓ dropped

**(1) T7 (tab:archObj) — 4th factorial cell ADDED.** MAMP+pose (U-Net Encoder) = **.524±.028 / .260±.036 /
.485±.029**, from `cvn_mampuencfullpose.json` (job **929970**, chimera `cvn_results/`, raw aggregate n=25:
walk mean .5235 sem .0282, point .2596 .0358, pick .4846 .0286). Now a true 2×2 Architecture×Pose-Head;
".260" prose claim now backed by a table row. Prose trimmed (dropped floating ±.036, kept ".260").

**(2) MLD VAE DESCRIBED.** One sentence added before Table 6: "identical to the MLD SkipTransformer AE
except its bottleneck is a Gaussian latent trained with an added KL term + reparameterized sampling;
reference point not comparison target." Justification = [[project_mld_ae_vs_vae_justification]]. Row value
.597/.324/.501 (`cvn_mldvae.json`) unchanged.

**(3) T6 (tab:sweep) ✓ COLUMN DROPPED — provenance failure.** The bars .478/.370/.431 are HARDCODED
constants (`BARS={}` in `baselines_nested.py:37`, `cv_nested.py:39`, `tmr_nested.py:18`, both repos
`.../triplets/` AND `.../motion-similarity/`) with NO derivation comment ("Bars to beat" per
cv_nested.py:24). TESTED against real baselines and they match NOTHING current: geometric means
(DTW .490/.273/.303, geo .371/.234/.264), geometric upper-CI (+1.96SE: DTW .544/.349/.378, geo
.425/.294/.323), vanilla (.476/.213/.491), TMR (.452/.425/.408) — pointing .370 exceeds EVERY current
pointing baseline ⇒ stale legacy constant. Also caption said "non-overlapping 95% CI" which is wrong
(it's lower-CI-bound vs a fixed bar, not two intervals). FIX: dropped ✓ column + 3 constants entirely;
table→`lccc`, caption now points to Table 4 (baselines) which makes the "only config clearing all
baselines on all 3" claim RIGOROUSLY via Holm-significant Wilcoxon (p=.004/9e-6/.005 vs vanilla, all
geometric p<1e-4). Prose §5.3 updated to cite Table 4 significance, not CI-vs-bar.

**ALL 9 T6 rows re-verified live vs chimera cvn JSONs 2026-07-28 — every value byte-exact** (vanilla
recon/+vel/vel-only, MLD-AE recon/+vel/vel-only, MLD-VAE, MAMP, MAMP+pose). Table body fully
reproducible; only the ✓ thresholds were the artifact.

## 28. TABLE HIGHLIGHT CONVENTION — bold-best + asterisk-significance (2026-07-28, Funda request)

Funda wanted best-performers marked even absent a significance test. NEW uniform convention across ALL
result tables (each caption states which it uses):
- **BOLD = best (highest mean) in each column** (T4/T8: within each regime-pair). Pure descriptive; no test.
- **$^{*}$ = difference is significant (Wilcoxon p<.05)** — ONLY where a test exists (T4, T8).
- **ITALIC = MAMP+pose (our method)** — unchanged.

**SE VERIFIED FIRST** (user asked): all 11 sweep rows' SE re-checked vs chimera JSON sem (=SD/√25) —
all match. Vanilla walking true sem=.0305→rounds .031 (paper correct; my display truncated to .030).

**Per-table (all bold cells = verified column/regime max via python):**
- T2 backbone: MLD-AE row bold all 3 (.532/.331/.498). T3 objective: bold SPLITS (Plain-vel-only point
  .362, MLD-recon+vel walk .537, MLD-recon-only pick .498) — honestly shows no single objective wins all.
- T5 posehead: MAMP+pose bold all 3. T7 arch: MAMP+pose(Plain) bold walk .552 & point .434; **MAMP(U-Net)
  bold pick .566** (>MAMP+pose .543 — honest, matches prose "U-Net increases picking").
- **T6 sweep: MLD-VAE bold WALKING .597** (>MAMP+pose .552), MAMP+pose bold point .434 & pick .543.
  DECISION (user): bold true-max EVEN when MLD-VAE wins walking — SAFE because (a) MLD-VAE framed as
  "reference rung not comparison target" in body L513 + T6 caption, per [[project_mld_ae_vs_vae_justification]]
  (deterministic-control choice, report-both-openly = anti-cherry-pick); (b) paper's actual claim is
  significance vs BASELINES (T4 Holm-Wilcoxon), not "highest mean every sweep row". MAMP+pose stays italic.
- **T4 baselines:** MAMP+pose bold + $^{*}$ all 3 (Holm-sig vs every baseline p=.004/9e-6/.005).
- **T8 TMR:** bold=higher-in-regime, $^{*}$ on the 4 SIGNIFICANT cells only (raw point TMR .425* p=.007,
  raw pick MAMP .481* p=.011, ft walk MAMP .699* p=.026, ft point TMR .573* p=.003); the 2 TIES bolded
  WITHOUT star (raw walk .462 p=.85, ft pick .611 p=.98). Verified against ledger §7.3.
- T1 kinematic: NO bold (descriptive stats, "best" undefined) — deliberate.

Audited: every bold = true max; every $^{*}$ = real Wilcoxon sig; every marked table's caption states its
convention. All numbers/SE untouched.

## 29. POST-SUBMISSION MECHANISM-TEASING JOB QUEUE (2026-07-29, NOT run — planning only)

Good-science follow-up (user: "not about the submission, about good science"). NO jobs launched; this is a
staged plan only. Full detail in memory `project_mechanism_teasing_jobs.md` + scratchpad
`mechanism_jobs/JOB1_mamppose_selection_pool_SPEC.md`.

Goal: definitively localize WHICH transformer mechanism carries perceptual signal. State of the mechanism
localization (from [[project_swap_pooling]]): MLD's ~.11 pointing edge is NOT skips (inert), NOT capacity
(35M control flat at .221≈.213), NOT pooling-as-main-effect (attn-pool wins ONLY inside MLD, +.093 → a
pooling×arch INTERACTION, and a liability elsewhere).

- **JOB 1 (CRUX):** MAMP+pose × selection pooling — the untested cell. MAMP+pose mean-pools yet wins pointing
  (.434); does a SELECTION readout help/hurt/leave-inert? Fills 2×2→2×3. 3 arms (user-chosen "both"):
  (1a) SSL-trained attn-pool = pure thesis test (raw rep pre-localized?); (1b) perceptual attn-pool =
  reuse `perc_finetune.py` variant A, ~zero code; (max) crude selection, `encode_mamp_lma.py --pool max`
  already supported. Verified impl: thread `--pool` at `cv_preliminary.py:242`; label-free target =
  `MAMP/model_mamp/transformer.py:269 latent_pose_pred`. Canonical ckpt `MAMP/output_dir/lma_mamp_pose_holdout`
  (up to ckpt-1199), config `config/lma_mamp_pretrain.yaml`, env `torch_gpu_cu12`.
- **JOB 2:** decoder swap (MLD cross-attn-query decoder vs vanilla std) — prime remaining suspect for the
  residual MLD edge; the "separate future corner" swap-pooling deferred.
- **JOB 3:** peakedness localization probe (per-fold, SELECT-chosen epoch, TEST clips, avg over 25) — designed,
  never run ([[project_localization_probe]]).

⚠️ HONESTY GUARDRAIL: until Job 1 runs, the "two routes to localization are redundant" claim is a HYPOTHESIS,
not established. Do not write it as fact.

### 29.1 JOB 1 LAUNCHED (2026-07-29 eve) — max-pool + perceptual-pool arms RUNNING
Harness patched: `cv_preliminary.py` (bak `cv_preliminary.py.bak_poolgrid`) — threaded `--pool {cfg.get("pool","mean")}`
into the mamp `encode()` (default "mean" ⇒ ALL existing results byte-identical) + added registry entry
`"MAMP+pose-max"` (pool="max"). Smoke-verified `--pool max` encodes 69 pointing clips dim=256 ✓.
- **JOB 952001 `cvn_mampposemax`** (arm=max-pool, crude selection): `cv_nested.py --encoders MAMP+pose-max --k 5
  --repeats 5 --epoch-grid 100 --loss-plateau 1199` → `cvn_mampposemax_results.json`. RUNNING chimera21.
- **JOB 952003 `percftA_mamppose`** (arm 1b, learnable perceptual attn-pool = perc_finetune variant A, rank loss,
  per-fold base-epoch SELECT over 200/400/600/800/1000/1199): `perc_finetune.py --variants A --loss rank
  --ckpt-subdir lma_mamp_pose_holdout` → `percftA_mamppose_rank_results.json`. Answers "selection+SUPERVISION?"
  (bonus row, NOT the pure thesis). PENDING.
- **JOB 952014 `sslpool_mamppose`** (arm 1a, THE CRUX): new script `ssl_pool_eval.py` (scratchpad-authored,
  smoke-verified k=2/ep3 on chimera → ran clean, pointing≈.39 on the tiny run). SSL-trained AttnPool on FROZEN
  MAMP+pose@1199, NO perceptual labels: pool trained on TRAIN with MSE to clip mean per-frame pose (label-free
  SSL, pose_dim=204), early-stop on SELECT SSL loss, eval raw Spearman on TEST. n=25 (5×5). Answers the PURE
  thesis (raw rep pre-localized?). PENDING. → `sslpool_mamppose_results.json`.
Compare all to canonical MAMP+pose mean-pool: walk .552 / point .434 / pick .543.

**Reusables verified (for future mechanism jobs):** `perc_finetune.py` exports `load_mamp`, `load_clip_tensor`,
`AttnPool`; `cv_preliminary.py` exports `load_action_keys(a, Config())→(module,keys)`, `make_folds`,
`get_inverse_direct_comparison_value`, `raw_spearman`. `load_action_keys` needs a `Config()` object NOT the model
yaml (smoke caught this). `load_mamp` returns full yaml (`cfg["model_args"]["dim_feat"]`). SLURM: account
`cs_funda.durupinarbabur`, partition `pomplun`, env `torch_gpu_cu12`. All sbatch + `ssl_pool_eval.py` + `patch_pool.py`
in scratchpad `mechanism_jobs/`.

**1a caveat (honest):** base epoch fixed at converged 1199 (encoder frozen so base matters less than for
fine-tuning); per-fold SELECT-based base-epoch selection is a future refinement, noted not-yet-done.

### 29.2 JOB 1 RESULTS (all COMPLETED exit 0, 2026-07-30) — MAMP+pose readout grid (raw Spearman n=25)
Canonical mean-pool: W .552 / P .434 / K .543.
- **952001 max-pool** (crude selection): raw W .530 / **P .453±.022** / K .522 (triplet W.434/P.285/K.453).
- **952014 SSL-attn-pool (1a)**: raw W .482±.033 / **P .265±.037** / K .485±.038.
- **952003 perc-attn-pool (1b, variant A rank loss)**: W .638 / **P .424** / K .608.

FINDINGS (verified from JSONs):
1. POINTING is READOUT-ROBUST: max-pool .453 ≈ mean .434 (within ~1SE); perceptual-pool .424 ≈ .434.
   No readout beats mean-pool on pointing ⇒ supports route (b) localization-in-REPRESENTATION (crux leans YES).
2. Localized/distributed axis reappears: max-pool HELPS sparse pointing (+.019) but HURTS distributed
   walking (.552→.530) — same pattern as MLD→attn-pool [[project_swap_pooling]].
3. ⭐ SSL-pool (1a) pointing COLLAPSED to .265 (−.169, ~5SE). CAUSE: its SSL target = reconstruct mean
   per-frame pose = a POSE-FAITHFULNESS objective, which §7 probe showed is ANTI-CORRELATED with pointing
   alignment. So training readout toward pose-faithfulness pushed AWAY from perceptual alignment = an
   INTERVENTIONAL confirmation of the correlational decodability dissociation. Elegant but NOT the pure
   localization test intended.
CAVEAT: 1a's pose-recon SSL target confounds the crux; clean localization answer = max-pool + 1b (both
≈mean on pointing). A pure 1a needs a localization-preserving SSL target OR the frozen-random control.
See [[project_mechanism_teasing_jobs]].

## 30. EFFORT-EXPOSURE CONFOUND: full-corpus vs cut1 (2026-08-27, user-raised) — for hybrid design + rebuttal
Full-corpus MAMP+pose (.552/.434/.543) pretrains on 5,453 clips = **91 non-eval LMA Effort recordings** (same
PERFORM/LMA generation as the eval set) + Bandai 3,077 + CMU 2,285. cut1/corpus-matched MAMP+pose
(.462/.302/.481) pretrains on 14,143 AMASS clips, NO LMA effort. KEY: cut1 has MORE clips yet WORSE pointing
(.302 < .434) ⇒ the full-corpus edge is NOT "more data" — it's a **pretraining↔eval distribution home-field
prior** (the 91 effort-stylized clips prime the effort-similarity SELECT/TEST). NOT leakage (the 91 are
strictly disjoint from the 342 held-out eval clips; held-out separation intact), but a distributional
familiarity confound. ⇒ full-corpus .434 pointing is "effort-warm", cut1 .302 is "cold".
IMPLICATIONS: (1) HYBRID must run at MATCHED corpus (cut1) to isolate the objective from this confound —
`docs/specs/HYBRID_mamppose_tmr_SPEC.md`. (2) REBUTTAL: if a reviewer probes "does effort-clip pretraining
inflate your pointing?", answer = held-out clean (91 ⊥ 342), disclosed in §Pretraining Corpus, AND the
corpus-matched cut1 comparison (where TMR/MAMP+pose are effort-cold and fair) is the honest cross-model
arena. No paper number changes; this is framing/design provenance.

## 31. SUPPLEMENT CLARITY FIX (2026-08-27): SELECT-perceptual checkpoint criterion made explicit
The submitted supplement §Checkpoint-Selection stated the [100,PLATEAU] window + reconstruction-bounded
PLATEAU + "keeps SELECT from being used twice" but was IMPLICIT on the *within-window selection criterion*.
The mechanism (verified in cv_nested.py SELECT step): among the grid, the reported checkpoint MAXIMIZES mean
raw-Spearman across the 3 actions on SELECT (action-agnostic) — perceptual, not reconstruction. FIXED in
docs/overleaf/supplement.tex (added the explicit sentence). Pure clarification — no number/claim change; the
submitted PDF is unchanged, this preps camera-ready. Reproduction spec docs/specs/HYBRID... also states it
explicitly so new trainings match the criterion, not just the window.
