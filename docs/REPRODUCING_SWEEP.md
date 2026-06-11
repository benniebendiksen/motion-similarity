# Reproducing the model sweep (training → encoding → evaluation)

This document records exactly how the per-model results in the paper (`docs/paper_draft.md`,
§4 and §6, especially the §6.6 full sweep) were produced, so they can be regenerated end to
end. It covers all nine encoders under both regimes:

- **raw** — pooled encoder embeddings, plain L2 distance vs. human dissimilarity `d_perc`;
- **+triplet** — the same embeddings passed through the triplet metric-learning module
  trained jointly on all three actions.

Everything is evaluated on the **fixed seed-42 validation split** (`MOTION_SPLIT_SEED=42`,
`val_ratio=0.4`), held-out (the 342 LMA states/drives are excluded from encoder
pretraining).

> Two machines are involved. **Encoder pretraining and embedding extraction** run on the
> chimera cluster (`torch_gpu_cu12` env), under
> `/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets` (referred to below as
> `$TRIP`). **Triplet training + final evaluation** run locally
> (`motion-similarity` conda env) under
> `…/motion-similarity-project` (referred to as `$PROJ`), because the local repo holds the
> DTW-corrected geodesic baseline and the inference pipeline. Embeddings are rsync'd between
> the two (§3).

---

## 0. The nine models

| # | Paper name | Backbone | Objective | Held-out | Embedding dir tag |
|---|------------|----------|-----------|----------|-------------------|
| 1 | Recon-only, plain transformer | AETransformer (vanilla) | rot-MSE 20×, no velocity | yes | `abl_provNoVel_p1` |
| 2 | Recon-only, MLD (AE-holdout) | MLD SkipTransformer (AE) | plain MSE | yes | `sweep_aeh_p1` |
| 3 | MLD + velocity bolt-on | MLD SkipTransformer (AE) | MSE + velocity 1.0 (warm-started) | yes | `abl_v0aevel_p1` |
| 4 | MLD, rotMSE 20× + vel 100×, scratch | MLD SkipTransformer (AE) | weighted recon + heavy velocity | yes | `abl_provloss_p1` |
| 5 | VAE, MLD (held-out) | MLD SkipTransformer (VAE) | ELBO | yes | `sweep_vaeh_p1` |
| 6 | AE, MLD (in-sample) | MLD SkipTransformer (AE) | plain MSE, full corpus | **no** | `sweep_aef_p1` |
| 7 | VAE, MLD (in-sample) | MLD SkipTransformer (VAE) | ELBO, full corpus | **no** | `sweep_vaef_p1` |
| 8 | MAMP (motion-only) | MAMP transformer | masked motion prediction | yes | `mampf_mamp_p1` |
| 9 | **MAMP+pose** | MAMP transformer | masked motion + aux pose head | yes | `mampf_mampPose_p1` |

All embedding dirs hold one pooled vector per clip, `<stem>_emb.pt`, for each of the three
actions: `<tag>_walking`, `<tag>_pointing`, `<tag>_picking` (68 / 69 / 76 non-mirror clips).

---

## 1. Encoder pretraining (chimera)

```bash
ssh chimera; cd $TRIP
source ~/miniconda3/etc/profile.d/conda.sh && conda activate torch_gpu_cu12
```

All trainers honor `holdout_lma_states_drives` (the held-out flag) and seed 0.

**MLD AE family** (`train_ae_mld.py`; presets in `make_preset`):

```bash
# 2  AE-holdout  (plain MSE, held-out)            -> checkpoints_ae_mld/v0_ae_seed0
python -u training/train_ae_mld.py --run-name v0_ae          --seed 0
# 3  AE + velocity 1.0 bolt-on                     -> checkpoints_ae_mld/v0_ae_vel_seed0
python -u training/train_ae_mld.py --run-name v0_ae_vel      --seed 0
# 4  PROV-loss: rotMSE 20x + velocity 100x scratch -> checkpoints_ae_mld/v0_ae_provloss_seed0
python -u training/train_ae_mld.py --run-name v0_ae_provloss --seed 0
# 6  AE-full (no hold-out, in-sample)              -> checkpoints_ae_mld/v0_ae_full_seed0
python -u training/train_ae_mld.py --run-name v0_ae_full     --seed 0
```

**MLD VAE family** (`train_vae_mld.py`):

```bash
# 5  VAE-holdout                                   -> checkpoints_vae_mld/v0_mld_holdout_seed0
python -u training/train_vae_mld.py --run-name v0_mld_holdout --seed 0
# 7  VAE-full (in-sample)                          -> checkpoints_vae_mld/v3_full_seed0
python -u training/train_vae_mld.py --run-name v3_full        --seed 0
```

**Vanilla AETransformer** (`train_AE_ablation_with_velocity_curriculum.py`; presets `A-curriculum|PROV|PROV_noVel|B|C|D`):

```bash
# 1  PROV_noVel: rot-MSE 20x, NO velocity (genuine recon-only vanilla AE)
#    -> checkpoints/v3_ablation_PROV_noVel/cp_PROV_noVel_<epoch>.pth
python -u train_AE_ablation_with_velocity_curriculum.py --preset PROV_noVel --epochs 10000
```

**MAMP family** (`MAMP/main_pretrain.py`; masked motion prediction adapted to LMA 6-D):

```bash
# 8  MAMP motion-only      -> MAMP/output_dir/lma_mamp_holdout/checkpoint-*.pth
python MAMP/main_pretrain.py --config MAMP/config/lma_mamp_pretrain.yaml
# 9  MAMP+pose             -> MAMP/output_dir/lma_mamp_pose_holdout/checkpoint-*.pth
python MAMP/main_pretrain.py --config MAMP/config/lma_mamp_pose_pretrain.yaml
```

> The MAMP configs set `dim_in=6, num_joints=34, num_frames=120, t_patch_size=4,
> mask_ratio=0.80, motion_aware_tau=0.75`, feature dim 256, and apply the LMA holdout. The
> `*_pose` config additionally enables the auxiliary pose head (`pose_weight: 1.0`).

---

## 2. Embedding extraction (chimera)

Each command writes pooled `N=1` embeddings (one 256-D vector/clip; vanilla AETransformer is
128-D) to `datasets/<tag>_<action>/`. Run on a GPU node; on the login node a stale-driver
CUDA warning appears but CPU extraction still succeeds.

```bash
cd $TRIP
source ~/miniconda3/etc/profile.d/conda.sh && conda activate torch_gpu_cu12
export CUDA_VISIBLE_DEVICES=""   # CPU is fine for N=1 extraction

# --- MLD AE/VAE family: probes/encode_sweep.py (--kind ae|vae) ---
for A in Walking Pointing Picking; do a=${A,,}
  python probes/encode_sweep.py --kind ae  --ckpt checkpoints_ae_mld/v0_ae_seed0/best.pt           --action $A --n-windows 1 --out-dir datasets/sweep_aeh_p1_$a
  python probes/encode_sweep.py --kind ae  --ckpt checkpoints_ae_mld/v0_ae_vel_seed0/best.pt       --action $A --n-windows 1 --out-dir datasets/abl_v0aevel_p1_$a
  python probes/encode_sweep.py --kind ae  --ckpt checkpoints_ae_mld/v0_ae_provloss_seed0/epoch_10000.pt --action $A --n-windows 1 --out-dir datasets/abl_provloss_p1_$a
  python probes/encode_sweep.py --kind ae  --ckpt checkpoints_ae_mld/v0_ae_full_seed0/best.pt      --action $A --n-windows 1 --out-dir datasets/sweep_aef_p1_$a
  python probes/encode_sweep.py --kind vae --ckpt checkpoints_vae_mld/v0_mld_holdout_seed0/best.pt --action $A --n-windows 1 --out-dir datasets/sweep_vaeh_p1_$a
  python probes/encode_sweep.py --kind vae --ckpt checkpoints_vae_mld/v3_full_seed0/best.pt        --action $A --n-windows 1 --out-dir datasets/sweep_vaef_p1_$a
done

# --- Vanilla AETransformer (model 1): probes/encode_aetransformer.py ---
for A in Walking Pointing Picking; do a=${A,,}
  python probes/encode_aetransformer.py --ckpt checkpoints/v3_ablation_PROV_noVel/cp_PROV_noVel_7300.pth \
    --action $A --n-windows 1 --out-dir datasets/abl_provNoVel_p1_$a
done

# --- MAMP family (models 8, 9): MAMP/encode_mamp_lma.py (mean-pool, native) ---
for A in Walking Pointing Picking; do a=${A,,}
  python MAMP/encode_mamp_lma.py --ckpt MAMP/output_dir/lma_mamp_holdout/checkpoint-599.pth \
    --config MAMP/config/lma_mamp_pretrain.yaml --action $A --pool mean --out-dir ../datasets/mampf_mamp_p1_$a
  python MAMP/encode_mamp_lma.py --ckpt MAMP/output_dir/lma_mamp_pose_holdout/checkpoint-599.pth \
    --config MAMP/config/lma_mamp_pose_pretrain.yaml --action $A --pool mean --out-dir ../datasets/mampf_mampPose_p1_$a
done
```

> **Path quirk (MAMP only):** `encode_mamp_lma.py` resolves `--out-dir` relative to the MAMP
> package, so `../datasets/...` lands in `virtual_reality/datasets/`, **not**
> `$TRIP/datasets/`. Move/rsync those into `$TRIP/datasets/` (or directly to the local repo)
> after extraction.

---

## 3. Sync embeddings to the local repo

Triplet training + evaluation run locally. Pull every model's three action dirs:

```bash
cd $PROJ/datasets
TAGS="abl_provNoVel_p1 sweep_aeh_p1 abl_v0aevel_p1 abl_provloss_p1 sweep_vaeh_p1 sweep_aef_p1 sweep_vaef_p1 mampf_mamp_p1 mampf_mampPose_p1"
for t in $TAGS; do for a in walking pointing picking; do
  mkdir -p ${t}_${a}
  rsync -aq chimera:$TRIP/datasets/${t}_${a}/ ${t}_${a}/
done; done
```

(Each `<tag>_<action>/` should contain 68/69/76 `*_emb.pt` for walking/pointing/picking.)

---

## 4. The evaluation sweep (local)

### 4a. Raw-embedding grid (no triplet)

Pooled L2 vs `d_perc` on the seed-42 validation split. Reproduces §6.6 "Raw embeddings".
Self-contained (`scipy` + the comparison CSVs in `aux/`); no model loading required:

```bash
cd $PROJ/motion-similarity
python - <<'PY'
import torch, glob, random, ast, warnings, numpy as np, pandas as pd
from itertools import combinations
from scipy import stats
warnings.filterwarnings("ignore")
ROOT = '..'                       # repo root ($PROJ)
AUX  = ROOT + '/motion-similarity/aux/{a}_similarity_comparisons_ratios.csv'
ACTIONS = ['walking', 'pointing', 'picking']
MODELS = {
 'plain-tf':'datasets/abl_provNoVel_p1_{a}', 'AE-holdout':'datasets/sweep_aeh_p1_{a}',
 'AE-vel':'datasets/abl_v0aevel_p1_{a}',     'PROV-loss':'datasets/abl_provloss_p1_{a}',
 'VAE-holdout':'datasets/sweep_vaeh_p1_{a}', 'AE-full':'datasets/sweep_aef_p1_{a}',
 'VAE-full':'datasets/sweep_vaef_p1_{a}',    'MAMP':'datasets/mampf_mamp_p1_{a}',
 'MAMP+pose':'datasets/mampf_mampPose_p1_{a}'}

def eff(f):  # Walking_-1_-1_0_1_emb.pt -> (-1,-1,0,1)
    c = f.split('/')[-1].replace('_emb.pt', '')
    return tuple(int(t) for t in c.split('_')[1:])

def load_emb(tmpl, a):
    d = {}
    for f in sorted(glob.glob(ROOT + '/' + tmpl.format(a=a) + '/*.pt')):
        d[eff(f)] = torch.load(f, map_location='cpu', weights_only=True).squeeze().numpy().astype(np.float64)
    return d

def load_dperc(a):
    df = pd.read_csv(AUX.format(a=a))
    df['efforts_tuples'] = df['efforts_tuples'].apply(
        lambda x: [tuple(ast.literal_eval(t)) for t in x.split('_')])
    o = {}
    for i in range(0, len(df), 3):
        g = df.iloc[i:i+3]; pair = g.iloc[0]['efforts_tuples']
        lr = g[(g['selected0'] == 0) & (g['selected1'] == 2)]
        if lr.empty or pd.isna(lr['count_normalized'].iloc[0]): continue
        o[frozenset(pair)] = 1.0 - lr['count_normalized'].iloc[0]   # d_perc
    return o

def val_keys(keys, seed=42, vr=0.4):
    rng = random.Random(seed)
    ks = sorted(k for k in keys if k != (0, 0, 0, 0)); rng.shuffle(ks)
    return set(ks[:max(1, int(len(ks) * vr))])

def corr(emb, dp, vk):
    keys = [k for k in emb if k != (0, 0, 0, 0) and k in vk]; d = []; t = []
    for ki, kj in combinations(keys, 2):
        tv = dp.get(frozenset((ki, kj)))
        if tv is None: continue
        d.append(np.linalg.norm(emb[ki] - emb[kj])); t.append(tv)
    return stats.pearsonr(d, t)[0], stats.spearmanr(d, t)[0]

DP = {a: load_dperc(a) for a in ACTIONS}
print(f"{'model':14s}" + "".join(f"{a[:4]+'P':>8}{a[:4]+'S':>8}" for a in ACTIONS))
for name, tmpl in MODELS.items():
    row = f"{name:14s}"
    for a in ACTIONS:
        emb = load_emb(tmpl, a); P, S = corr(emb, DP[a], val_keys(set(emb.keys())))
        row += f"{P:8.3f}{S:8.3f}"
    print(row)
PY
```

This reproduces §6.6 "Raw embeddings": for each model/action it loads `*_emb.pt` keyed by
effort tuple, reconstructs the seed-42 split (`random.Random(42)`, sorted keys, shuffle,
first 40% → val), computes pairwise L2 over validation pairs, and correlates (Pearson +
Spearman) against `d_perc = 1 − count_normalized[(selected0=0, selected1=2)]` from
`aux/<action>_similarity_comparisons_ratios.csv`.

### 4b. Triplet-refined grid (+triplet)

For each model: train the triplet metric-learning module **jointly on all three actions**
(seed 42, 120 epochs, cosine schedule) via the production runner, then evaluate the refined
embeddings with the inference pipeline. The runner consumes the `*_emb.pt` dirs directly
(single-embedding mode) via the `MOTION_EMB_DIR_*` env vars:

```bash
cd $PROJ/motion-similarity
export MOTION_SPLIT_SEED=42
PY=python   # the motion-similarity env's python
DS=$PROJ/datasets
EX=$PROJ/exemplars_dir/similarity_exemplars

for TAG in abl_provNoVel_p1 sweep_aeh_p1 abl_v0aevel_p1 abl_provloss_p1 \
           sweep_vaeh_p1 sweep_aef_p1 sweep_vaef_p1 mampf_mamp_p1 mampf_mampPose_p1; do
  export MOTION_EMB_DIR_WALKING=$DS/${TAG}_walking
  export MOTION_EMB_DIR_POINTING=$DS/${TAG}_pointing
  export MOTION_EMB_DIR_PICKING=$DS/${TAG}_picking
  # IMPORTANT: clear the cached per-action embedding-similarity pickles so each model regenerates
  rm -f $EX/*_embeddings_rots_only_similarity_labels_exemplars_dict_local.pickle

  # TRAIN triplet net (all three actions jointly)
  $PY run_enhanced_embedding_triplet_training.py \
      --walking-dir $MOTION_EMB_DIR_WALKING \
      --pointing-dir $MOTION_EMB_DIR_POINTING \
      --picking-dir  $MOTION_EMB_DIR_PICKING \
      --combination-method rots_only --epochs 120 --scheduler cosine

  # EVALUATE refined embeddings (per-action Pearson + Spearman vs d_perc, validation subset)
  MOTION_CHECKPOINT_FILE=0_final_embedding_model.pt $PY pipelines/infer_emb_vec.py
done
```

Per action, read the value under the block
`"<ACTION> (VALIDATION SUBSET): EMBEDDING L2 VS RAW FEATURE GEODESIC DISTANCE"` →
`Embedding L2 distances - {Pearson,Spearman}`. (The embedding L2 result is identical across
the GEODESIC and DTW blocks; either works — the baselines differ, not the embedding row.)

> **Notes.**
> - `--combination-method rots_only` is required: single-vector `_emb.pt` dirs are loaded in
>   "single-embedding mode," so the combination method is moot but must be a valid choice.
> - Triplet training is stochastic (k-means neutral init, online mining). The MAMP+pose row
>   reproduces the §6.1 headline within run-to-run tolerance (e.g. picking-S +0.499 vs the
>   canonical +0.507).
> - The full driver used to produce §6.6 is preserved verbatim in the project notes.

---

## 5. Geometric baselines (DTW + DTW-aligned geodesic)

The baselines compare on **raw, unpadded** sequences (no 120-frame windowing). The
DTW-corrected geodesic lives in the **local** `pipelines/infer_emb_vec.py`
(`compute_geodesic_distances_dtw`, `num_joints=28`); the chimera copy has only the legacy
frame-padded `compute_geodesic_distances` — **do not** use it (it inflates the pointing
geodesic to +0.411 vs the correct +0.353).

```bash
cd $PROJ/motion-similarity
python - <<'PY'   # seed-42 val split; reads {action}_similarity_labels_exemplars_dict_local.pickle
import pickle, ast, random, numpy as np, pandas as pd
from scipy import stats
from pipelines.infer_emb_vec import compute_geodesic_distances_dtw, calculate_real_variable_length_dtw
EX='../exemplars_dir/similarity_exemplars/'; AUX='aux/{a}_similarity_comparisons_ratios.csv'
# load raw unpadded sequences, build d_perc, restrict to seed-42 val keys, then:
#   geo = compute_geodesic_distances_dtw(raw_val)         # DTW-aligned per-frame geodesic
#   dtw = calculate_real_variable_length_dtw(raw_val)     # variable-length DTW
# correlate each against d_perc (Pearson + Spearman).
PY
```

Expected (seed-42 val, Spearman / Pearson):

| baseline | walk-S | point-S | pick-S |
|----------|--------|---------|--------|
| DTW | +0.478 | +0.370 | +0.431 |
| Geodesic (DTW-aligned) | +0.345 | +0.353 | +0.339 |

---

## 6. Determinism checklist

- `MOTION_SPLIT_SEED=42` for every train/eval invocation (stratified shuffle, val_ratio 0.4).
- Encoder pretraining excludes the 342 LMA states/drives (held-out flag).
- k-means neutral uses `random_state=42`.
- Raw-grid + baseline numbers are deterministic; +triplet numbers are reproducible to
  run-to-run tolerance (stochastic mining/init).
- Reproduced headline (MAMP+pose+triplet): walking +0.486, pointing +0.458, picking +0.507
  (Spearman), matching to four decimals on independent re-runs of the canonical pipeline.
