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

BASE_CKPT=checkpoints/datacomp_baseline_epoch_1.pt

# InfoNCE baseline
accelerate launch --multi_gpu --num_processes=2 -m alignment.alignment \
    --loss_type infonce \
    --load_from ${BASE_CKPT} \
    --save_name infonce_clean.pt

# Latent-SimPO
accelerate launch --multi_gpu --num_processes=2 -m alignment.alignment \
    --loss_type latent-simpo \
    --load_from ${BASE_CKPT} \
    --save_name latent_simpo_clean.pt

# Triplet-margin ablation
accelerate launch --multi_gpu --num_processes=2 -m alignment.alignment \
    --loss_type triplet-margin \
    --load_from ${BASE_CKPT} \
    --save_name triplet_margin_clean.pt

# Latent-CPO ablation
accelerate launch --multi_gpu --num_processes=2 -m alignment.alignment \
    --loss_type latent-cpo \
    --load_from ${BASE_CKPT} \
    --save_name latent_cpo_clean.pt

# Preference-InfoNCE ablation
accelerate launch --multi_gpu --num_processes=2 -m alignment.alignment \
    --loss_type pref-infonce \
    --load_from ${BASE_CKPT} \
    --save_name pref_infonce_clean.pt