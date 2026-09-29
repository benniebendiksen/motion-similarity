#!/bin/bash
#SBATCH --job-name=pairdist
#SBATCH --account=impact
#SBATCH --partition=AICORE_A100
#SBATCH --qos=aicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=48gb
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/motion-similarity/stats_fold_aware/pairdist_%A.out
# Encode-only (no training). Resumable: reruns skip matrices already in pairdist_<action>.npz, and the
# work dir lives on scratch (not node /tmp) so a requeue reuses finished encodes.
source $(conda info --base)/etc/profile.d/conda.sh; conda activate torch_gpu_cu12
cd /hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/motion-similarity
export MOTION_IS_REMOTE=1 PYTHONWARNINGS=ignore
echo "Job $SLURM_JOB_ID pairdist on $SLURM_NODELIST $(date)"
python -u stats_fold_aware/dump_pair_distances.py --baselines \
  --work-dir /hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/motion-similarity/stats_fold_aware/work
echo "DONE exit=$? $(date)"
