# Mechanism Findings — why perception ≠ faithful representation

Consolidated empirical support for the program thesis: **the encoder that best predicts human
perceptual similarity is NOT the one that most faithfully reconstructs or decodes the motion.**
Companion to `PIPELINE_TRACKER.md` (status) and `SHAPLEY_ATTRIBUTION.md` (the tool's math).

Metric throughout: **ρ** = Spearman(embedding-distance, human `d_perc`), per action. Higher = more
perceptually aligned. MAMP+pose ≈ 0.46 / 0.30 / 0.48 (walk/point/pick) cut1; full-corpus pointing ρ=.434.

---

## The dissociations (each an independent knife, same conclusion)

### 1. Capacity dissociation
A 35M-param vanilla transformer reconstructs motion BEST (lowest MPJPE ≈ .00305) yet perceives
WORST. Capacity buys reconstruction, not perception. (ledger; `van_bigcap`.)

### 2. Decodability dissociation (A2) — ✅ COMPLETE & PUBLISHED (main §8 + supplement app:probe)
On POINTING, perceptual alignment runs inversely (almost monotonically) to LINEAR pose-decodability.
**AUTHORITATIVE = v1 probe** (`localization_probe.py` → `probe_*.json`): ridge from mean-pooled
embedding → per-frame pose, k=16 PCA pose targets, GCV ridge, SAME folds as perceptual eval.
FULL-CORPUS encoders (ρ already paired/matched — no cut1/full mixing):

| encoder | pointing ρ | linear R² | nonlinear-MLP R² |
|---|---|---|---|
| **MAMP+pose** (best aligned) | **.434** | **.509** (lowest) | .645 |
| vanilla + attention pool (worst) | .213 | .727 (highest) | .829 |
| other encoders ordered in between (full table in supplement) | | | |

**Nonlinear MLP control IS DONE (published):** recovers more pose from EVERY encoder (.509→.645,
.727→.829) while PRESERVING the ordering ⇒ pose is PRESENT everywhere (not destroyed) but, in the
better-aligned encoders, arranged to be less *linearly* accessible → contributes less to the L2 metric.
Velocity & acceleration NOT linearly decodable from any encoder (R²<0). Action-specific: measurable
only on pointing (flat on walk/pick → no mechanistic claim there).

SECONDARY (exploratory, NOT the paper): `localization_probe_v2.py` → `probe2_*.json` uses a DIFFERENT
config (embedding-PCA k=16, pose-PCA m=2, SD(R²) as primary metric); its mean_r2 (MAMP+pose pointing
0.424) is a different parameterization and must NOT be quoted as THE decodability result. Its 6-encoder
spread did hint the inverse relation isn't a perfectly clean monotone across ALL encoders (MLD both
highly decodable and moderately aligned) — consistent with the paper's hedged "almost monotonically /
others in between," nothing to change.

### 3. U-Net skip "relief valve"
Skips help reconstruction but NEUTRALIZE the auxiliary pose head — they let the model be faithful
WITHOUT being perceptually organized. (4-corner skip×arch ablation; ledger.)

### 4. Peakedness (A1) — perceptual signal is TEMPORALLY localizable but JOINT-distributed
Probe: `rho(top-k salient slices ONLY)` vs `rho(bottom-k ONLY)`, subset-only design (the naive
ablate-from-mean design is washout-confounded — mean-pool is robust to frame drop; its null is an
artifact; use subset-only at aggressive fraction, ~10%). MAMP+pose cut1, ckpt-1199, 2026-08-27:

| axis | finding |
|---|---|
| **TEMPORAL** | concentration REAL, action-graded **picking > pointing > walking**. 10% subset: picking top=0.473 (≈ full 0.472) vs bot=0.336; pointing top=0.223 vs bot=0.166; walking top=0.302 < full=0.406 (distributed). Picking sharpest (reach-grasp = discrete event); walking distributed (periodic). |
| **JOINT** | NO concentration. Joint selectivity ~0/negative all actions. Joint-saliency profile FLAT (max/mean ≈ 1.4) and ACTION-INVARIANT — same core/root joints (idx 2,8,19,7,3) top every action. No action-specific "perceptual joint." |

**Reading:** perception lives in *time-where* (a few apex frames), not *body-part-where*. Consistent
with MAMP+pose's known readout-robustness (swap-pooling). CAVEAT: absolute ρ (pointing 0.216) below
nested-CV 0.302 → ckpt-1199 may be post-divergence; concentration is RELATIVE so expected robust,
re-confirm at pre-divergence plateau ckpt.

---

## The intervention (Phase 2) — does ADDING semantic signal raise perceptual relevance? [🔵 IN PROGRESS]
Everything above DIAGNOSES perceptual signal. The hybrid is the first attempt to INJECT it: MAMP masked
backbone (kept — Dissociation 2 says full-recon hurts) + a semantic contrastive scaffold (TMR recipe:
frozen distilbert TOKEN embeddings → trainable light non-VAE ACTOR head; motion pooled → proj; symmetric
InfoNCE τ=0.7 + false-neg filter via frozen mpnet caption-caption cos>0.6; MPNet=filter only).
See webpage §09, [[reference_tmr_architecture]], docs/specs/HYBRID_mamppose_tmr_SPEC.md.

**PILOT (contrastive-only, C1K0):** job 1020752, pomplun/chimera21 H200, launched 2026-09-19; ~ep1060/1200,
loss 2.15→~0.49, clean, no NaN. Baseline = existing cut1 MAMP+pose (.462/.302/.481; contrastive_weight=0≡base).

**EVAL SUITE (COMPLETE 2026-09-22 — pilot ckpt-1199):**
| eval | what it measures | status | RESULT |
|---|---|---|---|
| LMA ρ (nested-CV, `MAMP-hybrid`; both under identical [100,1199]) | perceptual guardrail + weak proxy | ✅ | hybrid **.487 / .212 / .485** vs base-matched .462/.302/.481 → walk +.025, **point −.090 (~1.9σ REAL DROP)**, pick +.004 |
| **SNP@k** (`probes/snp_eval.py`, 3000-clip, ckpt-1199) | motion-manifold semantic organization | ✅ | dist-Spearman .250→**.419 (+.169)**; SNP@5 +.017, SNP@10 +.035 → contrastive DID organize manifold |
| A′ re-probe (peakedness `probes/peakedness_v2_param.py`) | HOW the manifold reorganized | ✅ | **global saliency reshaping** — see below |

**A′ RESULT (base→hybrid, rho_all): walk .406→.385, point .216→.122 (−.094), pick .472→.447.** The contrastive
GLOBALLY reshaped internal saliency geometry, not a pointing-only effect: (1) picking's temporal localization —
base's STRONGEST signal, Tsel@10 **+.138 → −.115** (FLIPPED) — yet picking rho_all barely moved ⇒ kinematically-rich
picking found a REDUNDANT route absorbing the reshaping; (2) pointing JOINT anti-selectivity BLEW UP (Jsel@50
−.016 → **−.160**); (3) walking Tsel@10 +.022 → −.046. **Only POINTING's overall alignment collapsed — it lacked
kinematic redundancy to absorb the semantic pull (sparse action).** Signature is consistent with dual-pass
INTERFERENCE (H2) reshaping the shared representation, but A′ ALONE does not adjudicate H1 vs H2; the H1/H2
trainings must be judged by whether A′ RESTORES picking's +temporal-localization & pointing's joint-selectivity
toward base, not merely recovers rho.

**VERDICT = SCENARIO B (over-collapse on pointing).** objective = general perceptual manifold quality for Phase-3
sample-selection, NOT the LMA-effort ranking. SNP@k gain confirms the manifold organized semantically; the
pointing regression confirms semantic ⊥ effort on the sparse action. Honest caveat: SNP@k is in-sample for the
pilot (contrastive saw all 10,174) and semantic (coarse proxy); Phase-3's new study is the true verdict. Next
levers: H1 (ground pooled z via reconstruction), H2 (unify to single forward pass), lower contrastive weight
0.1→0.05, variational-MAMP (orthogonal motion-side lever).

## Through-line
Every dissociation says the winning encoder **compresses and reorganizes** the motion rather than
faithfully reproducing it. This motivates:
- **Phase 2 (hybrid):** add the RIGHT abstraction via caption supervision (contrastive), plus KL,
  on top of masking+pose — hypothesis: contrastive→pointing, KL→walking, masking+pose→picking.
- **Phase 4 (tool):** since there is NO intrinsic per-joint perceptual specialization, the per-part
  color MUST be a per-comparison contribution share — see `SHAPLEY_ATTRIBUTION.md`.
