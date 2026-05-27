# Motion Similarity Pipelines

This directory contains every training and inference pipeline as a canonical, self-contained
script. All scripts resolve the project root via `__file__` and can be invoked from any working
directory.

---

## Running all experiments (quick start)

```bash
# Train every variant and evaluate — results go to experiments/<name>/
python pipelines/run_all_experiments.py

# Skip training and re-evaluate existing checkpoints:
python pipelines/run_all_experiments.py --skip-training

# Run specific variants only (comma-separated):
python pipelines/run_all_experiments.py --only raw_motion,emb_vec_rots_only

# List all available experiment names:
python pipelines/run_all_experiments.py --list

# View and compare saved results at any time:
python pipelines/compare_results.py              # all experiments, sorted by Pearson r
python pipelines/compare_results.py --sort r2    # sort by R²
python pipelines/compare_results.py --anim walking  # filter to one animation
```

### How isolation works

Each experiment gets its own checkpoint directory:

```
experiments/
  raw_motion/
    checkpoints/          ← all .pt files for this run
    train.log
    infer.log
    results.json          ← parsed metrics (Pearson / Spearman / R²)
  emb_vec_rots_only/
    ...
  comparison.csv          ← tidy flat table, written after every run
```

Isolation is achieved through two environment variables set by the orchestrator before
each subprocess call:
- `MOTION_CHECKPOINT_DIR` — overrides `config.checkpoint_root_dir` in `Config.__init__`
- `MOTION_CHECKPOINT_FILE` — tells each inference script which `.pt` file to load

These env vars can also be set manually to point inference scripts at any checkpoint.

---

## Running a single script directly

```bash
# From the project root
python pipelines/train_raw_motion.py --animation walking

# Or from inside pipelines/
cd pipelines
python train_raw_motion.py --animation walking
```

---

## Naming Convention

```
{verb}_{input_type}[_{variant}].py
```

| Segment | Values | Meaning |
|---------|--------|---------|
| `verb` | `train` / `infer` | Training pipeline or evaluation/inference script |
| `input_type` | `raw_motion` | Raw BVH motion sequences fed as 2-D arrays (frames × joint features) |
| | `emb_vec` | Pre-computed autoencoder (AE) embedding **vectors** (flat 1-D, one per clip) |
| | `emb_baseline` | No trained model — raw AE embedding distances only (baseline) |
| `variant` | *(omitted)* | Standard neutral (taken directly from the dataset) |
| | `_clustered` | Clustering-based neutral initialisation via `NeutralRepresentationLearner` |
| | `_percloss` | Uses the full `SimilarityNetwork` trainer, which supports perception-aligned loss and an adaptive distance module |

---

## How `SimilarityNetwork` Handles Two Input Types

`SimilarityNetwork` is a **trainer wrapper** class (not a `nn.Module`).  
It contains two separate build methods that install different backbone networks:

| Method | Backbone | Expects |
|--------|----------|---------|
| `build_model()` *(called by `__init__`)* | `SimilarityNetworkV0` — a 3-layer 2-D CNN | Raw motion arrays shaped `(frames, joint_features)` |
| `build_embedding_model()` *(must be called explicitly)* | `EmbeddingSimilarityNetworkV0` — a 4-layer MLP | Flat AE embedding vectors shaped `(embedding_dim,)` |

`EmbeddingRefiningSimilarityNetwork` is a **separate, lighter** trainer class that always
builds `EmbeddingSimilarityNetworkV0` (MLP) and hard-codes
`use_perception_loss = False` / `use_adaptive_distance = False`.

---

## Training Pipelines

### `train_raw_motion.py`
*Renamed from `run_motion_triplet_training.py`*

- **Input:** Raw BVH motion sequences loaded via `osd.load_similarity_data`
- **Network:** `SimilarityNetwork` → `SimilarityNetworkV0` (CNN)
- **Neutral:** Standard dataset neutral (`bool_drop_neutral_exemplar = False`)
- **Clustering:** None
- **Key flags:**
  - `--animation {all,walking,pointing,picking,walking_pointing}` (default `all`)
  - `--use-perception-loss` — enable perception-aligned loss (batch-integrated triplet + rank correlation)
- **Produces checkpoints for:** `infer_raw_motion.py`

---

### `train_raw_motion_clustered.py`
*Renamed from `run_enhanced_motion_triplet_training.py`*

Same as `train_raw_motion.py` with one addition:

- **Neutral:** Replaced before training by a representative sequence learned via
  `MotionNeutralRepresentationLearner` (k-means on raw motion, from `neutral_clustering_motion.py`)
- **Key flags:**
  - `--use-clustering` (default `True`) — disable with `--use-clustering False`
  - `--n-clusters` — number of k-means clusters (default `5`)
  - `--selection-strategy {centroid,medoid}` (default `centroid`)
  - `--animation {all,walking,pointing,picking,walking_pointing}` (default `all`)
  - `--use-perception-loss`
- **Produces checkpoints for:** `infer_raw_motion.py` (same inference script as above)

---

### `train_emb_vec.py`
*Renamed from `run_embedding_triplet_training.py`*

- **Input:** Pre-computed AE embedding vectors from `*_encoded` directories
  (loaded via `EmbeddingDataset` / `EmbeddingSimilarityDataLoader`)
- **Network:** `EmbeddingRefiningSimilarityNetwork` → `EmbeddingSimilarityNetworkV0` (MLP)
- **Neutral:** Standard dataset neutral
- **Clustering:** None
- **Perception loss / adaptive distance:** Hard-coded `False` (base triplet loss only)
- **Key flags:**
  - `--embedding-dir`, `--embedding-dir-2`, `--embedding-dir-3` — paths to walking / pointing / picking encoded dirs
  - `--combination-method {concat,weighted,root_only,rots_only}` (default `rots_only`) — how root + rotation embeddings are combined into one vector
- **Produces checkpoints for:** `infer_emb_vec.py`

---

### `train_emb_vec_clustered.py`
*Renamed from `run_enhanced_embedding_triplet_training.py`*

Same as `train_emb_vec.py` with one addition:

- **Neutral:** Replaced before training by a representative embedding vector learned via
  `NeutralRepresentationLearner` (k-means on AE embedding vectors, from `neutral_clustering.py`)
- **Key flags:** same as `train_emb_vec.py`, plus:
  - `--n-clusters` (default `5`)
  - `--selection-strategy {centroid,medoid}` (default `centroid`)
- **Produces checkpoints for:** `infer_emb_vec.py` (same inference script as above)

---

### `train_emb_vec_percloss.py`
*Renamed from `main_embeddings_input.py`*

- **Input:** Pre-computed AE embedding vectors (same as `train_emb_vec.py`)
- **Network:** `SimilarityNetwork` → `EmbeddingSimilarityNetworkV0` (MLP)
  — uses the *full* trainer that supports optional perception-aligned loss and adaptive distance.
  Calls `build_embedding_model()` right after construction to swap the CNN backbone for an MLP.
- **Neutral:** Dropped (`bool_drop_neutral_exemplar = True`, `bool_fixed_neutral_embedding = True`)
- **Clustering:** None
- **Key flags:**
  - `--animation {walking,pointing,picking}` (default `walking`) — **single** animation only
  - `--embedding-dir`
  - `--combination-method {concat,weighted,root_only,rots_only}` (default `concat`)
  - `--use-perception-loss` — enable perception-aligned loss
  - `--use-adaptive-distance` — enable adaptive distance module (requires `--use-perception-loss`)
  - `--scheduler {plateau,cosine,step}` (default `plateau`)
- **Produces checkpoints for:** `infer_emb_vec_percloss.py`

> **Note:** This pipeline is the only one that exposes `--use-perception-loss` for embedding
> inputs.

---

## Inference / Evaluation Scripts

All inference scripts reproduce the output format shown below and report Pearson / Spearman /
R² against human perceptual comparison data on the **validation (out-of-sample) subset**.

```
======================================================================
WALKING (VALIDATION SUBSET): EMBEDDING L2 VS RAW FEATURE GEODESIC DISTANCE
======================================================================
...
```

---

### `infer_raw_motion.py`
*Renamed from `embedding_inference.py`*

Evaluates a checkpoint from **`train_raw_motion.py`** or **`train_raw_motion_clustered.py`**.

Loads a `SimilarityNetwork` checkpoint and computes:
- Embedding L2 (from the trained network)
- Raw feature geodesic distance (quaternion-aware)
- Raw feature DTW distance

---

### `infer_emb_vec.py`
*Renamed from `embedding_inference_autoencoder.py`*

Evaluates a checkpoint from **`train_emb_vec.py`** or **`train_emb_vec_clustered.py`**.

Loads an `EmbeddingRefiningSimilarityNetwork` checkpoint and computes the same three metrics,
using the refined embedding vectors in place of raw CNN embeddings.

---

### `infer_emb_vec_percloss.py`
*Renamed from `embedding_inference_embeddings_inputs.py`*

Evaluates a checkpoint from **`train_emb_vec_percloss.py`**.

Same metric set as above; loads the `SimilarityNetwork`-with-MLP checkpoint produced by
the full-trainer pipeline.

---

### `infer_emb_baseline.py`
*Renamed from `embedding_only_inference.py`*

**No trained model is loaded.** Uses raw AE embedding distances directly as a baseline:
- AE Embedding L2
- Raw feature geodesic distance
- Raw feature DTW distance

Useful to establish the floor — how well the AE encoder alone captures human similarity
judgements before any triplet fine-tuning.

---

## Pipeline × Inference Matrix

| Train script | Inference script | Network | Input | Clustering |
|---|---|---|---|---|
| `train_raw_motion.py` | `infer_raw_motion.py` | CNN | Raw BVH | – |
| `train_raw_motion_clustered.py` | `infer_raw_motion.py` | CNN | Raw BVH | Motion k-means |
| `train_emb_vec.py` | `infer_emb_vec.py` | MLP | AE vectors | – |
| `train_emb_vec_clustered.py` | `infer_emb_vec.py` | MLP | AE vectors | Embedding k-means |
| `train_emb_vec_percloss.py` | `infer_emb_vec_percloss.py` | MLP | AE vectors | – |
| *(no training)* | `infer_emb_baseline.py` | – | AE vectors | – |

---

## Original Filenames (for reference)

| New name | Original name |
|---|---|
| `train_raw_motion.py` | `run_motion_triplet_training.py` |
| `train_raw_motion_clustered.py` | `run_enhanced_motion_triplet_training.py` |
| `train_emb_vec.py` | `run_embedding_triplet_training.py` |
| `train_emb_vec_clustered.py` | `run_enhanced_embedding_triplet_training.py` |
| `train_emb_vec_percloss.py` | `main_embeddings_input.py` |
| `infer_raw_motion.py` | `embedding_inference.py` |
| `infer_emb_vec.py` | `embedding_inference_autoencoder.py` |
| `infer_emb_vec_percloss.py` | `embedding_inference_embeddings_inputs.py` |
| `infer_emb_baseline.py` | `embedding_only_inference.py` |
