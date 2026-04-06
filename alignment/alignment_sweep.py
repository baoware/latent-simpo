import os
import subprocess

# define the hyperparameter grid you want to sweep
LOSS_TYPES = ["latent-simpo", "triplet-margin"]
GAMMAS = [0.1, 0.2, 0.3, 0.4, 0.5]               # decision boundary
LAMBDAS = [0.01, 0.05, 0.1, 0.2, 0.5]            # anti-collapse reg
BETAS =[5.0, 10.0, 15.0, 20.0, 25.0]             # simpo reward scale

base_checkpoint = "checkpoints/datacomp_baseline_epoch_1.pt"

if not os.path.exists("slurm_scripts"):
    os.makedirs("slurm_scripts")

for loss in LOSS_TYPES:
    for gamma in GAMMAS:
        for lam in LAMBDAS:
            # only sweep beta for simpo
            # for triplet margin, just run once per gamma/lam combo
            betas_to_run = BETAS if loss == "latent-simpo" else [10.0]
            
            for beta in betas_to_run:
                exp_name = f"{loss}_b{beta}_g{gamma}_l{lam}"
                save_name = f"{exp_name}.pt"
                
                slurm_content = f"""#!/bin/bash
#SBATCH --job-name={exp_name}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/{exp_name}_%j.log
#SBATCH --account=cs6770_sp26


cd /sfs/weka/scratch/rjr6zk/latent-simpo/
source .venv/bin/activate
pip install --quiet -r requirements.txt

mkdir -p logs/evals

echo "Running Sweep: {exp_name}"

python -m alignment.comp_alignment \\
    --loss_type {loss} \\
    --load_from {base_checkpoint} \\
    --save_name {save_name} \\
    --beta {beta} \\
    --gamma {gamma} \\
    --lambda_reg {lam}

python -m alignment.eval_safety --ckpt {save_name} --task all > logs/evals/{exp_name}_eval.txt

echo "Done"
"""

                # write and submit
                script_path = f"slurm_scripts/{exp_name}.slurm"
                with open(script_path, "w") as f:
                    f.write(slurm_content)
                
                print(f"Submitting {exp_name}...")
                print("----------")
                subprocess.run(["sbatch", script_path], check=True)
                
