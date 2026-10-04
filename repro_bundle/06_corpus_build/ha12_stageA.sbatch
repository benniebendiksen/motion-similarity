#!/bin/bash
#SBATCH --job-name=ha12A
#SBATCH --account=impact
#SBATCH --partition=AICORE_H200
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets/logs/ha12A_%A_%a.out
TR=/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets
PY=/home/p.bendiksen001/miniconda3/envs/j2s_gpu/bin/python
N=1191; S=6; C=$(( (N+S-1)/S )); ST=$(( SLURM_ARRAY_TASK_ID*C )); EN=$(( ST+C ))
cd $TR/learned_baselines/bvh2tmr_pipeline/joints2smpl && $PY ha12_params_to_bvh.py --start $ST --end $EN \
  --out_bvh_dir $TR/datasets/_humanact12_smpl_bvh --out_npz_dir $TR/datasets/_humanact12_bvhnpz
