#!/bin/bash
# =============================================================================
# SLURM array job — runs all 8 motion-similarity experiments on AICORE_H200.
# All 8 tasks are submitted simultaneously; SLURM schedules them as GPUs free.
#
# Partition notes (chimera):
#   AICORE_H200 + account=impact + qos=aicore → full H200 on chimera24,
#   non-preemptible by class jobs.  Max walltime ~4 days.
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
#SBATCH --partition=AICORE_H200
#SBATCH --account=impact
#SBATCH --qos=aicore
#SBATCH --nodelist=chimera24
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --mem=80gb
#SBATCH --time=3-23:59:00
#SBATCH --requeue
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
# torch_gpu_cu12 — CUDA 12.1 build, tested on chimera24 H200
# ---------------------------------------------------------------------------
source $(conda info --base)/etc/profile.d/conda.sh
conda activate torch_gpu_cu12

# Sanity checks
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "Python        : $(python -V)"
echo "Torch version : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# ---------------------------------------------------------------------------
# Run the experiment (train + infer)
# ---------------------------------------------------------------------------
cd "${PROJECT_ROOT}"

# Tell Config to use remote (chimera) data-directory paths even when no
# --task-index flag is passed.  MOTION_CHECKPOINT_DIR is set per-experiment
# by run_all_experiments.py; MOTION_IS_REMOTE handles the BVH / exemplar dirs.
export MOTION_IS_REMOTE=1

# Pass --skip-training when the caller sets SKIP_TRAINING=1 in the environment.
# Example: sbatch --export=ALL,SKIP_TRAINING=1 --array=0-7 pipelines/slurm_run_experiments.sh
SKIP_FLAG=""
if [ "${SKIP_TRAINING:-0}" = "1" ]; then
    SKIP_FLAG="--skip-training"
    echo "Training will be skipped (SKIP_TRAINING=1)."
fi

python pipelines/run_all_experiments.py \
    --only "${EXP_NAME}" \
    ${SKIP_FLAG} \
    2>&1 | tee "${LOG_DIR}/run.log"

EXIT_CODE=$?

echo "========================================"
echo "Finished at   : $(date)"
echo "Exit code     : ${EXIT_CODE}"
echo "========================================"

exit ${EXIT_CODE}
