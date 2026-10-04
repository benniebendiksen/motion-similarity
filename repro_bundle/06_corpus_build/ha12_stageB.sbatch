#!/bin/bash
#SBATCH --job-name=ha12B
#SBATCH --account=impact
#SBATCH --partition=AICORE_H200
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets/logs/ha12B_%j.out
TR=/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets
PY=/home/p.bendiksen001/miniconda3/envs/j2s_gpu/bin/python
cd $TR && $PY amass/retarget_amass_to_cmu_batch.py --src_dir $TR/datasets/_humanact12_smpl_bvh --dst_dir $TR/datasets/_humanact12_cmu
