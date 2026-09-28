# Variational MAMP+pose — Run Spec (Option A: per-patch VAE) — LOCKED

**Purpose (triple-duty):** completes the design factorial's one real hole (**VAE × masked**, map P6), answers the
**MLD-VAE-honesty** point (S4 — MLD-VAE beats MAMP+pose on walking .597 vs .552), and is the **Paper-2
variational-MAMP lever**. Launch **after** the statistics fix lands (it decides framing strength); the code can be
built + smoke-tested in parallel now. Companion docs: `00_REVISION_BRIEF.md`, `01_REVIEWER_MAP.md`.

## Decision (locked 2026-09-25): Option A — per-patch variational, NOT pooled
Make **each per-patch latent** stochastic (μ, σ) with a per-patch KL; the decoder consumes the sampled per-patch
latents exactly as MAMP does. **Do NOT pool the 1020 patches to a single Gaussian** — that is Option B
(MLD-VAE/TMR-style), which *fights masking* (inpaint 80% from one vector), abandons the per-patch inductive bias
that carries the effort signal, and collapses into "H1 + KL". Option A is chosen because it is the minimal,
masking-faithful variational bottleneck and **KL has footing per patch** (it regularizes the very latents the
decoder inpaints from). Rationale in full: `00_REVISION_BRIEF.md §4.1` + the session discussion.

## Architecture change (guarded by `kl_weight > 0`)
Build on **`MAMP_hybrid/`** (its `__init__` already has a **RESERVED `self.kl_weight`** slot; with
`contrastive_weight=0` and `z_recon_weight=0` the hybrid reduces to base MAMP+pose — **verify this equivalence
first**, see Guardrails). Changes to `model_mamp/transformer.py`:
1. **New heads** (only if `kl_weight>0`): `self.vae_mu = nn.Linear(dim_feat, dim_feat)`,
   `self.vae_logvar = nn.Linear(dim_feat, dim_feat)`.
2. **In `forward()`**, between `forward_encoder` and `forward_decoder`:
   ```
   feat, mask, ids_restore, ids_keep = self.forward_encoder(x, mask_ratio, tau)   # per-patch [NM, len_keep, D]
   if self.kl_weight > 0:
       mu = self.vae_mu(feat); logvar = self.vae_logvar(feat)
       logvar = logvar.clamp(-6, 6)                       # numerical safety
       z = mu + torch.randn_like(mu) * torch.exp(0.5*logvar)   # reparam (train)
       kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()   # per-element mean over patches
       feat = z                                            # decoder consumes sampled per-patch latents
   pred_motion, pred_pose = self.forward_decoder(feat, ids_restore)
   ...
   if self.kl_weight > 0: loss = loss + self.kl_weight * kl
   ```
3. **KL footing:** during the masked pass `feat` = only the ~204 *visible* patch latents → KL regularizes exactly
   the masked-prediction bottleneck. Correct and intended.

## ⚠ Eval / encode readout MUST use μ (the biggest gotcha)
The clip embedding is `mean-pool(encoder per-patch output)`. For the VAE it must be `mean-pool(μ)`, **not** the raw
pre-μ features and **not** a stochastic sample. So the encode path (`encode_mamp_lma.py` and any `_pooled_full`)
must, when `kl_weight>0`, apply `vae_mu` to `forward_encoder(mask_ratio=0)` output and mean-pool **that**.
Otherwise the eval embedding is inconsistent with what was trained. Add a `--variational`/config-driven branch.

## Config
`MAMP_hybrid/config/lma_mamp_vae_pose_cut1.yaml` = the base MAMP+pose config with:
- `kl_weight: 1.0e-4`  (small β; the ONE hyperparameter — sweep {1e-4, 1e-3} if needed; see collapse guardrail)
- `contrastive_weight: 0.0`, `z_recon_weight: 0.0`  (pure MAMP+pose + per-patch VAE; no contrastive/no grounding)
- everything else identical to `lma_hybrid_C1K0`'s masking/pose block (dim_in 6, dim_feat 256, depth 8,
  num_frames 120, num_joints 34, t_patch_size 4, pose_weight 1.0, norm_skes_loss true, mask_ratio 0.80,
  epochs 1200, batch 64, blr 1e-3). Optional **KL warmup** (linearly ramp kl_weight over first ~100 ep) to avoid
  early posterior collapse.

## Training (sbatch)
Reuse the **AICORE_A100 self-resume** pattern (`training/hybrid_H1_aicore.sbatch` as template): account=impact,
partition=AICORE_A100 (chimera12/13, non-preemptible aicore QOS — verified 2026-09-25), self-resume from latest
checkpoint. **A100-40GB → use `--batch_size 32 --accum_iter 2`** (= effective batch 64, identical LR; the trainer
scales LR off *effective* batch). Output `MAMP_hybrid/output_dir/lma_mamp_vae_pose`. ~10–20h for 1200 ep (aicore
QOS has no wall cap). pomplun/H200 is the alternative if a 40GB card is too tight even at batch 32.

## Eval (same trio as the ladder)
- **LMA-ρ nested-CV** — register `MAMP-vae-pose` in `cv_preliminary.ENCODERS` (family=mamp, code_dir=MAMP_hybrid,
  config=lma_mamp_vae_pose_cut1.yaml, norm_stats_cut1, **variational readout=μ**). Run `cv_nested.py`.
  **Key question:** does the VAE lift **walking** (where MLD-VAE won .597) without hurting pointing/picking?
- SNP@k optional; **A′** (peakedness) to see how the stochastic bottleneck reshapes saliency.
- Compare against MAMP+pose (full-corpus .552/.434/.543; matched .462/.302/.481 — NEVER mix, see
  [[project_legacy_vs_cut1_provenance]]) and MLD-VAE (.597/.324/.501).

## Guardrails
1. **Base-equivalence check FIRST:** confirm `MAMP_hybrid` with `contrastive_weight=0, z_recon_weight=0,
   kl_weight=0` reproduces the paper's MAMP+pose alignment before trusting any VAE delta. (Memory:
   "contrastive_weight=0 ≡ base"; verify empirically on a few checkpoints.)
2. **Posterior collapse:** watch the KL term — if it →0 the latent is deterministic (β too small / warmup too
   long); if recon blows up, β too large. Report the KL value alongside recon.
3. **μ at eval** (not a sample) — see the readout section; the single most likely silent bug.
4. **Report the result either way** — help / neutral / hurt all complete the factorial and answer R1/AI honestly.
   Decide Paper-1 inclusion after seeing it; hurt-or-neutral still earns a sentence ("a variational bottleneck
   does not transfer MLD-VAE's advantage to the masked regime").

## Scope
Paper 1 = report as a factorial cell (design-ladder completeness). The same run doubles as the Paper-2
variational-MAMP lever (mechanism). Orthogonal to H1 (grounding); see [[project_hybrid_pilot_running]].
