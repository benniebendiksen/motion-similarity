#!/bin/bash
#SBATCH --job-name=ha12fit
#SBATCH --account=cs_funda.durupinarbabur
#SBATCH --partition=pomplun
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=05:00:00
#SBATCH --array=0-3
#SBATCH --output=/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets/logs/ha12fit_%A_%a.out
TR=/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets
J=$TR/learned_baselines/bvh2tmr_pipeline/joints2smpl
PY=/home/p.bendiksen001/miniconda3/envs/j2s_gpu/bin/python
N=1191; SHARDS=4; CHUNK=$(( (N + SHARDS - 1) / SHARDS ))
START=$(( SLURM_ARRAY_TASK_ID * CHUNK )); END=$(( START + CHUNK ))
echo "shard $SLURM_ARRAY_TASK_ID: clips [$START,$END)"
cd $J && $PY -u ha12_batch_fit.py --start $START --end $END --num_iters 150 --batch 128 --out_dir $TR/datasets/_humanact12_smpl_scaled
