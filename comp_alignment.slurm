#!/bin/bash
#SBATCH --job-name=vljepa_comprehensive_alignment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2         
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16         
#SBATCH --mem=160GB                
#SBATCH --time=24:00:00            
#SBATCH --output=comprehensive_%j.log

cd /scratch/rjr6zk/latent-simpo/
source .venv/bin/activate
pip install --quiet -r requirements.txt

accelerate launch --multi_gpu --num_processes=2 comp_alignment.py \
    --loss_type infonce \
    --load_from checkpoints/datacomp_baseline_epoch_1.pt \
    --save_name rlhf_infonce_comprehensive.pt

accelerate launch --multi_gpu --num_processes=2 comp_alignment.py \
    --loss_type latent-simpo \
    --load_from checkpoints/datacomp_baseline_epoch_1.pt \
    --save_name rlhf_latent_simpo_comprehensive.pt

accelerate launch --multi_gpu --num_processes=2 comp_alignment.py \
    --loss_type triplet-margin \
    --load_from checkpoints/datacomp_baseline_epoch_1.pt \
    --save_name rlhf_triplet_margin_comprehensive.pt  