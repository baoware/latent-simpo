import os
import subprocess

# define the hyperparameter grid you want to sweep
LOSS_TYPES = ["latent-simpo", "triplet-margin"]
GAMMAS = [0.1, 0.2, 0.3, 0.4, 0.5]               # decision boundary
LAMBDAS = [0.01, 0.05, 0.1, 0.2, 0.5]            # anti-collapse reg
BETA = 10.0                                      # simpo reward scale

base_checkpoint = "checkpoints/datacomp_baseline_epoch_1.pt"

os.makedirs("slurm_scripts", exist_ok=True)
os.makedirs("logs/evals", exist_ok=True)

job_count = 0

for loss in LOSS_TYPES:
    for gamma in GAMMAS:
        for lam in LAMBDAS:
            short_loss = "simpo" if loss == "latent-simpo" else "trip"
            exp_name = f"{short_loss}_g{gamma}_l{lam}"
            save_name = f"{exp_name}.pt"

            lines = [
                "#!/bin/bash",
                f"#SBATCH --job-name={exp_name}",
                "#SBATCH --partition=gpu",
                "#SBATCH --gres=gpu:a100:1",
                "#SBATCH --nodes=1",
                "#SBATCH --cpus-per-task=8",
                "#SBATCH --mem=80GB",
                "#SBATCH --time=08:00:00",
                f"#SBATCH --output=logs/{exp_name}_%j.log",
                "#SBATCH --account=cs6770_sp26",
                "",
                "cd /sfs/weka/scratch/rjr6zk/latent-simpo/",
                "source .venv/bin/activate",
                "pip install --quiet -r requirements.txt",
                "",
                "mkdir -p logs/evals",
                "",
                f"echo \"Running Sweep: {exp_name}\"",
                "",
                "# run the unified training script",
                "python -u -m alignment.comp_alignment \\",
                f"    --loss_type {loss} \\",
                f"    --load_from {base_checkpoint} \\",
                f"    --save_name {save_name} \\",
                f"    --beta {BETA} \\",
                f"    --gamma {gamma} \\",
                f"    --lambda_reg {lam}",
                "",
                "# run the safety evaluation automatically",
                f"python -u -m eval.eval_safety --ckpt {save_name} --task all > logs/evals/{exp_name}_eval.txt",
                "",
                "# run the vqa evaluation and append (>>) to the same log file",
                f"python -u -m eval.eval_vqa --ckpt {save_name} --task vqa >> logs/evals/{exp_name}_eval.txt",
                "",
                "echo \"Done.\""
            ]

            slurm_content = "\n".join(lines)
            script_path = f"slurm_scripts/{exp_name}.slurm"

            with open(script_path, "w", newline='\n') as f:
                f.write(slurm_content)

            print(f"Submitting {exp_name}...")
            print("----------")
            subprocess.run(["sbatch", script_path])
            job_count += 1

print(f"Successfully submitted {job_count} jobs to SLURM!")
