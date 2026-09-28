# TMLR Resubmission — Revision Brief (READ FIRST)

**Purpose.** Seed doc for the TMLR resubmission sprint of *"Motion Encoding for Human Perceptual
Similarity."* This pack lets a fresh conversation pick up cold. Companions in this folder:
`01_REVIEWER_MAP.md` (every critique → fix → data → status), `02_REVIEWER_FEEDBACK_RAW.md` (the three
reviews + decision, verbatim).

_Created 2026-09-25 after the AAAI-27 rejection (Phase-1, did not advance)._

---

## 1. Decision & framing (settled)
- **Venue = TMLR (Transactions on Machine Learning Research).** Rolling, ML/AI-branded, and its acceptance
  bar is **"are the claims correct & supported" + "would some of the audience be interested"** — novelty is
  *explicitly not* a criterion. This neutralizes the #1 rejection driver (R1/AI: "combination of existing
  models"). TMLR is **not** rigor-agnostic, though: unsupported claims still get rejected.
- **Effort level = a bounded, high-quality SPRINT (~2–3 wks), not "our all."** The required work sits almost
  entirely on **data we already have** (re-analysis + writing), and it's work JAIR would need anyway → not
  wasted. Rationale the author agreed to: **the mechanism research is compute-bound, not author-bound**, so a
  TMLR writing sprint runs in parallel with GPU jobs (H1, variational-MAMP) without stalling them.
- **TWO-PAPER PROGRAM (dissolves the fragmentation worry):**
  - **Paper 1 → TMLR (now):** the *benchmark + encoder study* — dataset, leakage-clean nested-CV protocol,
    MAMP+pose, ablations, the capacity + decodability dissociations. Current paper, stats fixed, claims scoped.
  - **Paper 2 → JAIR (later):** *what a perceptually-aligned representation should encode* — the
    mechanism→objectives work (H1 grounding / pooling / variational-MAMP) **+ a second validated user study**
    (fine-tuning + validation context). Distinct, higher-concept contribution; novelty rests on the causal
    mechanism + new study, NOT on re-publishing the benchmark. JAIR is HELD until both mature.
- Paper source: `docs/overleaf/main_2.tex` (body, 7pg AAAI) + `docs/overleaf/supplement.tex`. Switch template
  to TMLR (JMLR/`tmlr.sty`); remove AAAI double-blind constraints.

## 2. The required TMLR work (claims-support bar — do these or TMLR bites)
All on EXISTING data unless noted. Full detail per point in `01_REVIEWER_MAP.md`.
1. **Fix the statistics (NON-NEGOTIABLE).** Both serious reviewers flag treating the **25 repeated-CV folds
   as independent** for SE/Wilcoxon. Redo inference: bootstrap over the **56 Effort classes**, and/or average
   folds **within each repeat (n=5)**; re-run Holm across the surviving comparisons. Report which headline
   claims survive. **⚠ This propagates to EVERY small-delta claim, not just the headline** — the design-ladder
   deltas (skip observations, velocity near-ties, etc.) are all on the old stats; re-verify each before stating
   it as fact. Large deltas and nulls are robust; small ones (e.g. −.024 skip-hurts-pointing) are the risk.
2. **No-mirror headline.** Report the primary result on the **171 unmirrored clips**; state explicitly that
   original+mirror share an Effort key → same outer fold (confirm in code) so it was grouped, not leaked.
3. **Reliability / noise ceiling.** From raw MTurk responses (10–18 judgments/triplet) compute inter-rater
   reliability + a noise ceiling, so "ρ=.55" reads against an attainable ceiling. Main-text, not buried.
4. **MLD-VAE honesty (framing).** MLD-VAE **beats MAMP+pose on walking (.597 vs .552)** yet is walled off as
   "context." A TMLR reviewer will call the "only config that beats every baseline on all three" claim
   engineered-by-exclusion. Reframe: the claim is about **masked-prediction vs reconstruction as objectives**;
   MLD-VAE is a strong reconstruction reference MAMP+pose does *not* uniformly beat — say so plainly. (See §4
   for the new run that turns this into a strength.)
5. **Ordinal-vs-cardinal (framing).** AI reviewer's sharpest point: the target is defined ordinal, but the
   fine-tune uses `d_perc` *gaps* as cardinal margins. Reframe the anchored loss as a **rank/ordering**
   objective (margins *weight*, they don't assert metric distance); drop cardinal-distance language. No
   numbers change.

## 3. Important-but-lighter (clarity / positioning — TMLR cares moderately)
- **Related work** must cite & distinguish **MotionCritic (Wang 2025), MotionBERT (Zhu 2023), SkeletonMAE
  (Yan 2023)** (AI reviewer). Cheap; strengthens positioning.
- **Reframe the contribution** as *dataset + dissociation science*, not "new architecture." Answers R1
  ("combination of existing models") and the AI novelty note.
- **Readability / restructure:** define MAMP, TMR, LMA on first use; move the full method + losses + protocol
  *before* the ablation tables (AI suggestion). Directly answers R1's "heavy AI use / hard to read."
- **⭐ Principled design walkthrough (R1-c, map P6)** — the ladder exists but reads as config-tables, not a
  motivated search. Fix: (a) reframe each rung as a **hypothesis test** (question → change one thing →
  alternative ruled out → outcome → motivates next); (b) add a **design-ladder figure** vanilla→MAMP+pose; (c)
  **own the REGIME-DEPENDENCE of the skips** (⚠ NOT "skips win reconstruction" — that conflated the backbone
  comparison with the skip ablation): skips are **near-inert under reconstruction** (+.008 / −.001; the MLD
  recon advantage is its overall config, not skips); under **masking** they help walking/picking but hurt
  pointing; with the **pose head** they turn redundant+harmful across all three (pointing .434→.260) — which is
  exactly why the final model is **plain encoder + pose head, skips dropped**; state as a *result*. **⚠ GATED on
  the stats fix (§2.1–2.2): these skip deltas are on the OLD stats — re-verify under fold-aware + no-mirror
  before stating as fact; the −.024 pointing-hurt under masking-motion-only is the fragile one, the nulls and
  the −.174 pose-head collapse are robust; the stats fix touches EVERY small-delta claim in the ladder, not
  just the headline;** (d) clarify **TMR is the supervised yardstick, not a rung** (ladder culminates in MAMP+pose);
  (e) the factorial's one real hole = **VAE×masked**, filled by the run in §4.1.
- **Clarify "Effort"** (R1d): what LMA Effort is and how it parameterizes the stimuli (it does not "influence
  pose detection" — reviewer misframed; clarify the pipeline).
- **Reviewer error to clarify:** R1 thinks we use **CLIP** and asks for a with/without-CLIP ablation. **We do
  not use CLIP.** Clarify TMR's text encoder (frozen DistilBERT→ACTOR; see [[reference_tmr_architecture]]) and
  that our encoder is caption-free. One sentence.
- **Minor technical fixes (AI):** (a) Sec 3.2 feature-dim inconsistency — 34×6=204 vs "+3-D root translation";
  state layout exactly. (b) Sec 4.3 define the delta offset `s` and say masked-vs-all-patches for each loss.
  (c) Table 8 rows: label as **AMASS-pretrained** MAMP+pose (differ from Table 4). (d) DTW: state quaternion
  sign-invariance / path-length normalization.

## 4. Candidate NEW chimera runs (may be needed; decide inclusion per claims-support)
1. **⭐ Variational MAMP+pose — FULLY SPECCED in `03_VARIATIONAL_MAMP_SPEC.md` (Option A LOCKED).** *Why didn't
   we try a VAE MAMP+pose, given MLD-VAE wins walking?* **Decision (locked 2026-09-25): Option A = per-patch
   variational** — make each of the 1020 per-patch latents stochastic (μ,σ + per-patch KL), decoder consumes the
   sampled per-patch latents as normal MAMP. **NOT pooled to one vector** (Option B = MLD-VAE/TMR style) — that
   fights masking, abandons the per-patch inductive bias, and collapses into "H1 + KL". Option A keeps masking
   faithful and gives KL footing per patch. **TRIPLE-DUTY → likely REQUIRED for Paper 1:** (i) completes the
   VAE×masked factorial hole (P6), (ii) answers MLD-VAE-honesty (§2.4), (iii) = Paper-2 variational lever. Build
   on `MAMP_hybrid/` (reuse the reserved `kl_weight` slot; contrastive=0, z_recon=0). ⚠ eval readout must use **μ**.
   A100-40GB → `--batch_size 32 --accum_iter 2`. Run after the stats land; report result either way. **See
   `03_VARIATIONAL_MAMP_SPEC.md` for the full build/eval/guardrails.**
2. **(Optional) A stronger learned baseline** (e.g., MotionBERT features) to answer R2b "baseline coverage."
   Bigger lift; only if a reviewer presses. Likely defer to Paper 2.
3. **(Analysis, not training) Factor-level error breakdown** (Space/Weight/Time/Flow, State vs Drive, active
   region) — AI reviewer suggestion; supports the pointing story. Could be Paper 1 supplement or Paper 2.
- **Chimera recipe:** pomplun/`cs_funda.durupinarbabur` dodges the billing cap; env `torch_gpu_cu12`; see
  [[reference_chimera_encode_recipe]] and `docs/context/PROTOCOLS.md`. ⚠ pomplun/chimera21 has shown repeated
  external node kills (H1 jobs 1023761, 1024128) — make any long run **self-resuming** (auto-restart from
  latest checkpoint) or use the idle AICORE_A100 node.

## 5. What is OUT of scope for TMLR (Paper 2 / honest scoping)
- **Generalization** (synthetic, single skeleton, single protocol, 3 actions; R2d/AIf): **scope claims
  honestly** in Paper 1 ("results are limited to this benchmark"); the fix (natural motion / more base
  motions / second study) is **Paper 2**. Do NOT try to fix with new data for TMLR.
- **Overstated "limited supervision rivals large-scale semantic"** (R2c): **soften** — after matched FT,
  MAMP+pose beats TMR only on walking (a *trend* post-Holm, p_adj=.078), ties picking, loses pointing. Report
  as a mixed result, not parity. Already partly fixed; see [[project_tmr_finetune_result]].
- Body-aware / localized pooling, causal mechanism → **Paper 2** (the mechanism thread already underway).

## 6. Key numbers & provenance (guardrails — pull only from canonical)
- **Headline (full-corpus)** MAMP+pose walk/point/pick = **.552 / .434 / .543**; **matched-corpus** raw =
  **.462 / .302 / .481** (TMR .452 / .425 / .408); **+FT** MAMP+pose .699 / .463 / .605 vs TMR .649 / .573 /
  .611. **MLD-VAE(recon) .597 / .324 / .501.** n=25 (5×5 nested CV). ⚠ **NEVER mix full-corpus & cut1/matched
  numbers** — see [[project_legacy_vs_cut1_provenance]].
- Canonical results tree = `triplets/cvn_results/` (Jun23+); Jun16 tree superseded — [[reference_canonical_results_tree]].
- Provenance ledger `docs/provenance_ledger.md` (§0.5 maps every encoder → job/config/ckpt).
- Repro truth / checklist: `docs/REPRO_TRUTH.md`, `docs/ReproducibilityChecklist.tex`,
  [[project_checklist_yes_commitments]].

## 7. Relevant memory (source material)
[[reference_canonical_results_tree]] · [[project_legacy_vs_cut1_provenance]] · [[project_narrative_table_final]]
· [[project_mld_ae_vs_vae_justification]] · [[project_architecture_loss_sweep]] · [[project_swap_pooling]] ·
[[project_localization_probe]] · [[reference_tmr_architecture]] · [[project_tmr_finetune_result]] ·
[[project_tmr_reconstruction_framing]] · [[reference_dperc_lma_effort_space]] ·
[[reference_perceptual_data_provenance]] · [[project_body_dissociation_plan]] · [[reference_final_abstract]] ·
[[project_contrastive_corpus_mismatch]] · [[project_hybrid_pilot_running]]

## 8. Suggested first move for the sprint conversation
Start with **the statistics fix (§2.1–2.2)** — it's the one genuinely *required*, claims-defining item, and its
outcome (which headline claims survive fold-aware inference + no-mirror) determines how strongly the rest of
the paper can be framed. Everything else (reliability, reframing, related work, minor fixes, the VAE run) is
additive once the stats footing is known.

## 9. Live jobs & cross-session state (do NOT relaunch)
- **Paper-2 mechanism work is already in flight on chimera** and should keep running in parallel — the TMLR
  sprint is author-bound (writing/analysis), the mechanism work is compute-bound.
  - **H1 grounding** (Paper 2): job **1025076** `hybrid_H1`, resumed from `checkpoint-580` (pomplun/chimera21),
    PENDING/RUNNING as of 2026-09-25. ⚠ two prior instances (1023761, 1024128) were **externally killed** ~ep180
    / ~ep580 on chimera21 with no traceback (node/scheduler event, not a bug) — make it **self-resuming** (§10).
    Output `MAMP_hybrid/output_dir/lma_hybrid_H1`. Do NOT start a fresh H1; resume from latest checkpoint.
- **What the TMLR stats fix does NOT need:** new *training* jobs. Fold-aware inference = re-analysis of existing
  fold-level scores (`triplets/cvn_results/`); no-mirror headline = cheap **re-evaluation** (encode + correlate
  via `cv_nested.py` / `cv_preliminary.py` on the 171-clip subset), NOT retraining. The **only** genuinely new
  training run is the **variational MAMP+pose** (§4.1). So parallel GPU load for TMLR is light.

## 10. Chimera node recipes — BOTH families (for the new runs / re-eval)
Env (both): `source $(conda info --base)/etc/profile.d/conda.sh; conda activate torch_gpu_cu12`. Repo root
`/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets`. SSH: `ssh -i ~/.ssh/cluster
p.bendiksen001@chimera.umb.edu` (VPN or campus-wired; `chimera` alias may not resolve — use `chimera.umb.edu`).

**A) pomplun — PREFERRED (dodges the impact billing cap):**
```
#SBATCH --account=cs_funda.durupinarbabur
#SBATCH --partition=pomplun
#SBATCH --gres=gpu:1            # chimera21, H200 143GB
#SBATCH --mem=80gb
#SBATCH --time=3-23:59:00
#SBATCH --requeue
#SBATCH --open-mode=append
```
Association confirmed 2026-09-25: `cs_funda.durupinarbabur → pomplun`. ⚠ chimera21 has shown repeated external
kills — pair with the self-resume snippet below.

**B) impact / AICORE — FALLBACK (billing-capped, 24-hour QOS):**
```
#SBATCH --account=impact
#SBATCH --partition=AICORE_A100   # chimera12/13, A100 80GB (chimera12 frequently IDLE → good for parallel)
#SBATCH --qos=aicore              # association: impact → QOS {24hr, aicore}
#SBATCH --gres=gpu:1
#SBATCH --mem=80gb
#SBATCH --time=23:59:00           # respect the 24hr cap → self-resume across it for 1200-epoch runs
#SBATCH --requeue
#SBATCH --open-mode=append
```
Notes: `AICORE_H200` (chimera24) is **MIG-sliced** (`2g.35gb`/`1g.35gb`) — small memory, only for light jobs;
prefer `AICORE_A100`. Other associations on this account: `haehn→normal`, `pi_*→scavenger` (preemptible; avoid
for long runs). A 1200-epoch MAMP-family run is ~10–20h, so the 24hr impact cap needs self-resume; pomplun's
4-day limit does not.

**Self-resume snippet (drop into any sbatch so a re-submission auto-continues):**
```
OUT=/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets/MAMP_hybrid/output_dir/<run>
CKPT=$(ls -t "$OUT"/checkpoint-*.pth 2>/dev/null | head -1)
RESUME=${CKPT:+--resume "$CKPT"}   # empty on first launch, else newest checkpoint
python -u main_pretrain.py --config <cfg> $RESUME --output_dir "$OUT" --log_dir "$OUT"
```
`misc.load_model` restores model+optimizer+scaler and sets `start_epoch = ckpt.epoch+1` (verified). For hands-off
recovery, resubmit on failure (`sbatch --dependency=afternotok:$JID ...`) or a small watcher; or just prefer the
idle AICORE_A100 node when pomplun/chimera21 is flaky. See [[reference_chimera_encode_recipe]], PROTOCOLS.md §4.
