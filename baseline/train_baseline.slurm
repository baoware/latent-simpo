#!/bin/bash
#SBATCH --job-name=train_vljepa_baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2         
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16         
#SBATCH --mem=160GB                
#SBATCH --time=48:00:00            
#SBATCH --output=train_base_%j.log

cd /scratch/rjr6zk/latent-simpo/

echo "Job running on $(hostname)"
echo "Current directory: $(pwd)"
echo "----------"

echo "Activating virtual environment..."
source .venv/bin/activate
echo "----------"

echo "Checking and installing requirements..."
pip install --quiet -r requirements.txt

echo "Starting Training..."
accelerate launch --multi_gpu --num_processes=2 train_baseline.py
echo "----------"

echo "Job finished at $(date)"
echo "----------"