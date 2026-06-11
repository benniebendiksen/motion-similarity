# Soft-DTW in Embedding Space — Design

## Motivation

The DTW-fusion λ-ablation established two facts:

1. **On pooled embeddings** (one 512/256-D vector per clip), the DTW-fusion loss
   is inert — the embedding has no time axis for DTW's ordering to align to.
2. **On windowed-concat embeddings** (N sub-window latents concatenated), the
   DTW-fusion term unlocks a large Pearson gain (pointing +0.212 → +0.345) but
   only a modest Spearman gain (+0.220 → +0.256).

The Spearman shortfall is the diagnostic: concatenation imposes a **rigid**
window-to-window correspondence (clip-A window i compared only to clip-B
window i). DTW's rank superiority comes from **warping** — aligning A's window i
to B's window j where the poses actually match, regardless of timing. Soft-DTW
in embedding space adds exactly that warping, end-to-end differentiable.

## Core idea

Replace the single pooled embedding with a **sequence of per-window embeddings**,
and replace L2 distance between clips with **soft-DTW distance between their
embedding sequences**.

```
clip  --AE encoder per window-->  E = [e_1, e_2, ..., e_N]   (N x d)
d(A, B) = soft_dtw_gamma( E_A, E_B )    # warped alignment, differentiable
```

- `e_t` ∈ R^d is the latent of sub-window t (already produced by
  `encode_pointing_ae_windowed.py`, just kept as a sequence instead of concat).
- `soft_dtw_gamma` is the differentiable DTW of Cuturi & Blondel (2017): a
  smooth min (log-sum-exp with temperature γ) over alignment paths through the
  pairwise cost matrix `C[i,j] = ||e_A_i - e_B_j||^2`.

## Where it plugs in

The triplet machinery is metric-agnostic: it consumes a pairwise distance
matrix `classes_distances` over the batch's classes
(`TripletMining.calculate_distances`). Today that's L2 on pooled vectors.

**Change:** make `calculate_distances` compute soft-DTW between class embedding
*sequences* instead of L2 between class embedding *vectors*. Everything
downstream (alpha matrices, SEMI_HARD/ALL masks, the DTW-fusion regularizer)
is unchanged — it still just sees an N_class × N_class distance matrix.

This is the key architectural economy: **only the distance function changes.**
The loss, the human-triplet alphas, the clustering neutral, and the DTW-fusion
term all operate on the resulting matrix exactly as now.

## Components to build

1. **Sequence-valued embeddings.**
   - Encoder already supports it. Save per-window latents as a `[N, d]` tensor
     per clip (`_seq.pt`) rather than the concatenated `[N*d]` `_emb.pt`.
   - `EmbeddingDataset`: add a `sequence_mode` alongside `single_embedding_mode`
     that loads `[N, d]` and keeps the window axis.
   - Data loader yields `[B, N, d]` batches; the refining MLP applies per-window
     (shared weights across the N axis) → refined `[B, N, d]` sequences.

2. **Differentiable soft-DTW module** (`networks/soft_dtw.py`).
   - Input: two `[N, d]` sequences (or batched `[B, N, d]`).
   - Cost matrix `C[i,j] = ||x_i - y_j||^2`.
   - Forward recursion with soft-min:
     `R[i,j] = C[i,j] + softmin_gamma(R[i-1,j], R[i,j-1], R[i-1,j-1])`.
   - Return `R[N,N]` as the distance. γ ~ 0.1 (tunable).
   - Use the standard differentiable formulation; for N=4 windows the matrix is
     tiny (4×4), so a plain PyTorch recursion is fast enough — no CUDA kernel
     needed. Batch the pairwise computation over the N_class × N_class pairs.

3. **`calculate_distances` variant** that builds the class×class soft-DTW matrix
   from the batch of `[N, d]` sequences, replacing the L2 matmul path. Keep the
   class-neutral distance path analogous (soft-DTW to the neutral sequence).

4. **Config knobs:** `use_soft_dtw_distance` (bool), `soft_dtw_gamma` (float).
   Default off → exact current behavior.

## Fairness (unchanged argument)

Soft-DTW here operates **in the learned embedding space**, not on raw features.
It is the model's own representation being warped-aligned — we are NOT importing
DTW's raw-feature distances. The DTW-fusion regularizer (which DOES use raw-DTW
as a frozen prior) remains optional and separately ablatable. The headline claim
stays: correlation with **human** ratings, with raw DTW/geodesic as baselines
evaluated against the same human ground truth.

## Expected outcome / kill criterion

- **Green:** soft-DTW lifts pointing Spearman toward/past DTW's +0.370 while
  holding walking/picking, confirming warped alignment is the missing piece.
- **Red:** if soft-DTW Spearman plateaus near the windowed-concat result
  (~+0.256), then pointing's ceiling is the star-shaped human data itself, and
  DTW should simply be reported as the best perceptual proxy for pointing.

## Cost estimate

- soft_dtw.py + sequence_mode loader + calculate_distances variant: ~1 day.
- Per-window MLP weight-sharing: small change to the embedding network forward.
- N=4 keeps soft-DTW matrices 4×4 — negligible compute overhead vs L2.
