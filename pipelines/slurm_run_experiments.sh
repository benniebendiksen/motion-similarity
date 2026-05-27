#!/bin/bash
# =============================================================================
# SLURM array job — runs all 8 motion-similarity experiments on DGXH200.
# All 8 tasks are submitted simultaneously; SLURM schedules them as GPUs free.
#
# Submit from the project root:
#   sbatch pipelines/slurm_run_experiments.sh
#
# Each array task maps to one experiment name via EXPERIMENT_NAMES below.
# Per-experiment logs go to:
#   experiments/<name>/run.log      (combined stdout+stderr from run_all_experiments.py)
#   experiments/slurm_<jobid>_<taskid>.{log,err}  (SLURM header + exit code)
#
# To re-run a single task (e.g. index 3):
#   sbatch --array=3 pipelines/slurm_run_experiments.sh
# =============================================================================

#SBATCH --job-name=motion_sim
#SBATCH --partition=DGXH200
#SBATCH --account=cs_funda.durupinarbabur
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/motion-similarity/experiments/slurm_%A_%a.log
#SBATCH --error=/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/motion-similarity/experiments/slurm_%A_%a.err

# ---------------------------------------------------------------------------
# Experiment index → name mapping
# Must stay in sync with EXPERIMENTS list order in run_all_experiments.py.
# ---------------------------------------------------------------------------
EXPERIMENT_NAMES=(
    "raw_motion"           # 0  Raw BVH → CNN, standard neutral, base triplet loss
    "raw_motion_clustered" # 1  Raw BVH → CNN, k-means neutral, base triplet loss
    "raw_motion_percloss"  # 2  Raw BVH → CNN, perception-aligned loss
    "emb_vec_rots_only"    # 3  AE emb → MLP, rots_only, base triplet loss
    "emb_vec_concat"       # 4  AE emb → MLP, concat (root+rots), base triplet loss
    "emb_vec_clustered"    # 5  AE emb → MLP, rots_only, k-means neutral
    "emb_vec_percloss"     # 6  AE emb → MLP (full SimilarityNetwork), perception loss
    "emb_baseline"         # 7  No training — raw AE distances as floor baseline
)

EXP_NAME="${EXPERIMENT_NAMES[$SLURM_ARRAY_TASK_ID]}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT="/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/motion-similarity"
LOG_DIR="${PROJECT_ROOT}/experiments/${EXP_NAME}"

mkdir -p "${LOG_DIR}"

echo "========================================"
echo "SLURM job     : ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Experiment    : ${EXP_NAME}"
echo "Node          : $(hostname)"
echo "Started at    : $(date)"
echo "========================================"

# ---------------------------------------------------------------------------
# Activate conda environment
# ---------------------------------------------------------------------------
source /home/p.bendiksen001/miniconda3/etc/profile.d/conda.sh
conda activate torch_gpu

# Sanity checks
echo "Python        : $(which python)"
echo "Torch version : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU name      : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")')"

# ---------------------------------------------------------------------------
# Run the experiment (train + infer)
# ---------------------------------------------------------------------------
cd "${PROJECT_ROOT}"

python pipelines/run_all_experiments.py \
    --only "${EXP_NAME}" \
    2>&1 | tee "${LOG_DIR}/run.log"

EXIT_CODE=$?

echo "========================================"
echo "Finished at   : $(date)"
echo "Exit code     : ${EXIT_CODE}"
echo "========================================"

exit ${EXIT_CODE}
