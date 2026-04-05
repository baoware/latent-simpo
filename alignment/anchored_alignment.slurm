#!/bin/bash
#SBATCH --job-name=vljepa_anchored_alignment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2         
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16         
#SBATCH --mem=160GB                
#SBATCH --time=48:00:00            
#SBATCH --output=anchored_%j.log

cd /scratch/rjr6zk/latent-simpo/
source .venv/bin/activate
pip install --quiet -r requirements.txt

accelerate launch --multi_gpu --num_processes=2 -m alignment.anchored_alignment \
    --loss_type infonce \
    --load_from checkpoints/datacomp_baseline_epoch_1.pt \
    --save_name rlhf_infonce_anchored.pt

accelerate launch --multi_gpu --num_processes=2 -m alignment.anchored_alignment \
    --loss_type latent-simpo \
    --load_from checkpoints/datacomp_baseline_epoch_1.pt \
    --save_name rlhf_latent_simpo_anchored.pt

accelerate launch --multi_gpu --num_processes=2 -m alignment.anchored_alignment \
    --loss_type triplet-margin \
    --load_from checkpoints/datacomp_baseline_epoch_1.pt \
    --save_name rlhf_triplet_margin_anchored.pt  
