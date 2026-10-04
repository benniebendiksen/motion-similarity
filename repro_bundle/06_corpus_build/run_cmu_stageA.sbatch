#!/bin/bash
#SBATCH --job-name=cmuA
#SBATCH --account=impact
#SBATCH --partition=AICORE_H200
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH --array=0-11
#SBATCH --output=/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets/logs/cmuA_%A_%a.out
export MKL_THREADING_LAYER=GNU
TR=/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets
PY=/home/p.bendiksen001/miniconda3/envs/j2s_gpu/bin/python
cd $TR
mapfile -t SUBS < <(ls -d datasets/_amass_cmu_raw/CMU/*/ | sort)
N=${#SUBS[@]}; S=12; k=$SLURM_ARRAY_TASK_ID
for ((i=k; i<N; i+=S)); do
  sub="${SUBS[$i]}"
  $PY amass/amass_smplh_to_bvh_batch.py --amass_dir "$sub" \
    --out_npz_dir datasets/_amass_cmu_stageA_npz --out_bvh_dir datasets/_amass_cmu_stageA_bvh \
    --smpl2bvh_repo fit3d/third_party/smpl2bvh --target_fps 30 --python $PY
done
