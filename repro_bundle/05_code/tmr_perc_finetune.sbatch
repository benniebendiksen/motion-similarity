#!/bin/bash
# ============================================================================
# TMR perceptual fine-tune --- the SYMMETRIC control for "fine-tuned MAMP+pose beats TMR".
# Fine-tunes TMR's ACTOR motion encoder with the IDENTICAL triplet-rank objective + nested-CV
# protocol as our MAMP+pose fine-tune (perc_finetune.py), from the FIXED 263-feat guofeats.
# Smoke-verified end-to-end 2026-07-23 (load->encode->rank-loss->backprop->eval).
#
# Submit:  sbatch tmr_perc_finetune.sbatch
# ============================================================================
#SBATCH --job-name=tmr_perc_ft
#SBATCH --account=pi_funda.durupinarbabur
#SBATCH --partition=A30
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=01-00:00:00
#SBATCH --mem=100gb
#SBATCH --gres=gpu:1
#SBATCH --output=/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets/slurm_logs/tmr_perc_ft_%A.out

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tmr

# chimera data paths for Config (exemplars/comparisons) + faithful d_perc
export MOTION_IS_REMOTE=1
export TOKENIZERS_PARALLELISM=false

MS=/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/motion-similarity
cd "$MS"

# Primary arm = rank (the adaptive-margin ranking loss; the one that surpasses raw TMR for
# MAMP+pose). Full official protocol: 5 repeats x 5 folds = n=25, early-stop on SELECT.
python tmr_perc_finetune.py \
    --k 5 --repeats 5 --seed 42 \
    --epochs 120 --lr 1e-5 \
    --loss rank \
    --out "$MS/tmr_perc_finetune_results.json"

echo "DONE. Compare $MS/tmr_perc_finetune_results.json (aggregate W/P/K) vs MAMP+pose"
echo "rankdiag_headline (.699/.463/.605) and raw TMR (.452/.425/.408)."
