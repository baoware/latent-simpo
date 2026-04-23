import argparse
import os
from typing import List, Sequence

import torch
import torch.optim as optim
import torch.distributed.nn.functional as dist_nn
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from dotenv import load_dotenv
import huggingface_hub
from transformers import get_cosine_schedule_with_warmup

from src.config import Config
from src.model import VL_JEPA
from src.datasets import RLHFDataset, SafeVLDataset, VQADataset
from src.losses import (
    infonce_loss,
    latent_simpo_loss,
    triplet_margin_loss,
    latent_cpo_loss,          
    preference_infonce_loss,  
)

# less strict arg parsing lol
def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def alignment_collate(batch):
    out = {}
    keys = ["video", "q_ids", "win_ids", "win_mask", "lose_ids", "lose_mask"]

    for key in keys:
        exemplar = next(item[key] for item in batch if key in item)
        values = []
        for item in batch:
            if key in item:
                values.append(item[key])
            else:
                values.append(torch.zeros_like(exemplar))

        out[key] = torch.stack(values)

    return out


# proper masked mean for preference-only rows
def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


# differentiable gather for cross-GPU negatives
def gather_with_grad(tensor: torch.Tensor, accelerator: Accelerator) -> torch.Tensor:
    if accelerator.num_processes == 1:
        return tensor
    gathered = dist_nn.all_gather(tensor)
    return torch.cat(list(gathered), dim=0)


# modular sampling modes
def build_sampling_weights(lengths, sampling_mode):
    if sampling_mode == "equal-datasets":
        quotas = [1.0 / len(lengths)] * len(lengths)

    elif sampling_mode == "pref50-anchor50":
        # RLHF-V, SPA-VL, VQA
        quotas = [0.25, 0.25, 0.50]

    elif sampling_mode == "pref20-anchor80":
        quotas = [0.10, 0.10, 0.80]

    elif sampling_mode == "spavl-heavy":
        # RLHF-V, SPA-VL, VQA
        quotas = [0.20, 0.30, 0.50]

    else:
        raise ValueError(f"Unknown sampling_mode: {sampling_mode}")

    weights = []
    for quota, length in zip(quotas, lengths):
        weights.extend([quota / length] * length)
    return weights

# optional freezing for ablation
def maybe_freeze_y_encoder(model: VL_JEPA, freeze_y_encoder: bool) -> None:
    for param in model.y_encoder.parameters():
        param.requires_grad = not freeze_y_encoder
    for param in model.y_proj.parameters():
        param.requires_grad = not freeze_y_encoder


# only include Y-encoder params if unfrozen
def build_optimizer(model: VL_JEPA, cfg: Config, freeze_y_encoder: bool) -> optim.Optimizer:
    param_groups = [
        {
            "params": filter(lambda p: p.requires_grad, model.predictor_model.parameters()),
            "lr": cfg.lr_alignment,
        },
        {
            "params": model.predictor_head.parameters(),
            "lr": cfg.lr_alignment,
        },
    ]

    if not freeze_y_encoder:
        param_groups.extend(
            [
                {
                    "params": model.y_encoder.parameters(),
                    "lr": cfg.lr_y_encoder_alignment,
                },
                {
                    "params": model.y_proj.parameters(),
                    "lr": cfg.lr_y_encoder_alignment,
                },
            ]
        )

    return optim.AdamW(param_groups, weight_decay=0.01)


def model_forward_predictor(model, accelerator: Accelerator, video: torch.Tensor, q_ids: torch.Tensor) -> torch.Tensor:
    if accelerator.num_processes > 1:
        return model.module.forward_predictor(video, q_ids)
    return model.forward_predictor(video, q_ids)


def model_forward_y_encoder(model, accelerator: Accelerator, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if accelerator.num_processes > 1:
        return model.module.forward_y_encoder(ids, mask)
    return model.forward_y_encoder(ids, mask)


# much more modular CLI
def parse_args():
    cfg = Config()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--loss_type",
        type=str,
        required=True,
        choices=["infonce", "latent-simpo", "triplet-margin", "latent-cpo", "pref-infonce"],
    )
    parser.add_argument("--load_from", type=str, required=True, help="Path to Phase 1 baseline checkpoint.")
    parser.add_argument("--save_name", type=str, required=True, help="Output checkpoint filename.")

    parser.add_argument("--beta", type=float, default=cfg.beta, help="Preference reward scale.")
    parser.add_argument("--gamma", type=float, default=cfg.gamma, help="Preference target margin.")
    parser.add_argument("--lambda_reg", type=float, default=cfg.lambda_reg, help="InfoNCE regularization weight.")
    parser.add_argument("--cpo_bc_weight", type=float, default=cfg.cpo_bc_weight, help="Winner-anchor weight for latent CPO.")
    parser.add_argument("--pref_weight", type=float, default=cfg.pref_weight, help="Global weight on the preference term.")
    parser.add_argument("--temperature", type=float, default=cfg.temperature, help="InfoNCE temperature.")
    parser.add_argument("--max_grad_norm", type=float, default=cfg.max_grad_norm, help="Gradient clipping norm.")

    parser.add_argument(
        "--freeze_y_encoder",
        type=str2bool,
        default=False,
        help="Freeze Y-encoder during alignment. VL-JEPA-faithful default is False.",
    )

    parser.add_argument(
        "--sampling_mode",
        type=str,
        default="pref50-anchor50",
        choices=["equal-datasets", "pref50-anchor50", "pref20-anchor80"],
    )

    parser.add_argument(
        "--routing_mode",
        type=str,
        default="clean",
        choices=["clean", "legacy-scalar"],
        help="clean = masked per-sample preference reduction. legacy-scalar reproduces old scalar routing.",
    )

    parser.add_argument(
        "--simpo_variant",
        type=str,
        default="paper",
        choices=["paper", "legacy"],
        help="paper: softplus(gamma - beta*delta). legacy: -logsigmoid(beta*(delta-gamma)).",
    )

    parser.add_argument(
        "--pref_use_in_batch_negatives",
        type=str2bool,
        default=True,
        help="Use in-batch winner negatives for pref-infonce.",
    )

    parser.add_argument(
        "--use_cross_gpu_negatives",
        type=str2bool,
        default=True,
        help="Use differentiable cross-GPU gather for InfoNCE regularizer.",
    )

    parser.add_argument(
        "--pref_cross_gpu_negatives",
        type=str2bool,
        default=False,
        help="Use differentiable cross-GPU negatives for pref-infonce.",
    )

    parser.add_argument("--batch_size_alignment", type=int, default=cfg.batch_size_alignment)
    parser.add_argument("--epochs_alignment", type=int, default=cfg.epochs_alignment)
    parser.add_argument("--lr_alignment", type=float, default=cfg.lr_alignment)
    parser.add_argument("--lr_y_encoder_alignment", type=float, default=cfg.lr_y_encoder_alignment)

    parser.add_argument("--seed", type=int, default=cfg.seed)
    return parser.parse_args()


def main():
    args = parse_args()

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if token:
        huggingface_hub.login(token=token)

    accelerator = Accelerator(mixed_precision="bf16")
    set_seed(args.seed)

    cfg = Config()
    cfg.beta = args.beta
    cfg.gamma = args.gamma
    cfg.lambda_reg = args.lambda_reg
    cfg.temperature = args.temperature
    cfg.batch_size_alignment = args.batch_size_alignment
    cfg.epochs_alignment = args.epochs_alignment
    cfg.lr_alignment = args.lr_alignment
    cfg.lr_y_encoder_alignment = args.lr_y_encoder_alignment
    cfg.freeze_y_encoder = args.freeze_y_encoder
    cfg.sampling_mode = args.sampling_mode
    cfg.routing_mode = args.routing_mode
    cfg.simpo_variant = args.simpo_variant
    cfg.cpo_bc_weight = args.cpo_bc_weight
    cfg.pref_use_in_batch_negatives = args.pref_use_in_batch_negatives
    cfg.use_cross_gpu_negatives = args.use_cross_gpu_negatives
    cfg.pref_cross_gpu_negatives = args.pref_cross_gpu_negatives
    cfg.pref_weight = args.pref_weight

    if accelerator.is_main_process:
        print(f"Alignment | loss={args.loss_type} | routing={cfg.routing_mode} | sampling={cfg.sampling_mode}")
        print(
            f"beta={cfg.beta} | gamma={cfg.gamma} | lambda={cfg.lambda_reg} | "
            f"freeze_y={cfg.freeze_y_encoder} | simpo_variant={cfg.simpo_variant}"
        )
        print("----------")
        os.makedirs(cfg.output_dir, exist_ok=True)

    model = VL_JEPA(cfg)
    if not os.path.exists(args.load_from):
        raise FileNotFoundError(f"Checkpoint not found at {args.load_from}")

    if accelerator.is_main_process:
        print(f"Loading Phase 1 baseline from {args.load_from}")
        print("----------")

    state_dict = torch.load(args.load_from, map_location="cpu")
    model.load_state_dict(state_dict)
    model.predictor_model.gradient_checkpointing_enable()

    maybe_freeze_y_encoder(model, cfg.freeze_y_encoder)
    optimizer = build_optimizer(model, cfg, cfg.freeze_y_encoder)

    if accelerator.is_main_process:
        print("Loading RLHF-V, SafeVL, and VQAv2 anchor datasets...")
        print("----------")

    dataset_rlhf = RLHFDataset(cfg, split="train")
    dataset_safe = SafeVLDataset(cfg, split="train")
    dataset_vqa = VQADataset(cfg, split="train")

    combined_dataset = ConcatDataset([dataset_rlhf, dataset_safe, dataset_vqa])

    lengths = [len(dataset_rlhf), len(dataset_safe), len(dataset_vqa)]
    weights = build_sampling_weights(lengths, cfg.sampling_mode)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )

    dataloader = DataLoader(
        combined_dataset,
        batch_size=cfg.batch_size_alignment,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=alignment_collate,
    )

    num_training_steps = len(dataloader) * cfg.epochs_alignment
    num_warmup_steps = int(0.1 * num_training_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    for epoch in range(cfg.epochs_alignment):
        model.train()
        running_total = 0.0
        running_nce = 0.0
        running_pref = 0.0
        running_pref_fraction = 0.0

        progress = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Epoch {epoch + 1}/{cfg.epochs_alignment}",
            disable=not accelerator.is_main_process,
        )

        for step, batch in enumerate(progress):
            video = batch["video"].to(accelerator.device)
            q_ids = batch["q_ids"].to(accelerator.device)
            win_ids = batch["win_ids"].to(accelerator.device)
            win_mask = batch["win_mask"].to(accelerator.device)
            lose_ids = batch["lose_ids"].to(accelerator.device)
            lose_mask = batch["lose_mask"].to(accelerator.device)

            optimizer.zero_grad()

            pred_emb = model_forward_predictor(model, accelerator, video, q_ids)
            win_emb = model_forward_y_encoder(model, accelerator, win_ids, win_mask)
            lose_emb = model_forward_y_encoder(model, accelerator, lose_ids, lose_mask)

            # per-sample mask
            pref_mask = (win_ids != lose_ids).any(dim=-1)
            pref_fraction = pref_mask.float().mean()

            # fixed-lambda InfoNCE regularizer
            if cfg.use_cross_gpu_negatives:
                all_pred = gather_with_grad(pred_emb, accelerator)
                all_win = gather_with_grad(win_emb, accelerator)
            else:
                all_pred = pred_emb
                all_win = win_emb

            loss_nce = infonce_loss(all_pred, all_win, temperature=cfg.temperature, reduction="mean")
            loss_pref = pred_emb.new_zeros(())

            if args.loss_type == "infonce":
                loss = loss_nce

            elif args.loss_type == "triplet-margin":
                pref_vec = triplet_margin_loss(
                    pred_emb, win_emb, lose_emb, margin=cfg.gamma, reduction="none"
                )

                # can reproduce old routing or use clean masked routing
                if cfg.routing_mode == "legacy-scalar":
                    loss_pref = pref_fraction * pref_vec.mean()
                else:
                    loss_pref = masked_mean(pref_vec, pref_mask)

                loss = cfg.pref_weight * loss_pref + cfg.lambda_reg * loss_nce

            elif args.loss_type == "latent-simpo":
                pref_vec = latent_simpo_loss(
                    pred_emb,
                    win_emb,
                    lose_emb,
                    beta=cfg.beta,
                    gamma=cfg.gamma,
                    variant=cfg.simpo_variant,
                    reduction="none",
                )

                # CHANGED: can reproduce old routing or use clean masked routing
                if cfg.routing_mode == "legacy-scalar":
                    loss_pref = pref_fraction * pref_vec.mean()
                else:
                    loss_pref = masked_mean(pref_vec, pref_mask)

                loss = cfg.pref_weight * loss_pref + cfg.lambda_reg * loss_nce

            elif args.loss_type == "latent-cpo":
                pref_vec = latent_cpo_loss(
                    pred_emb,
                    win_emb,
                    lose_emb,
                    beta=cfg.beta,
                    bc_weight=cfg.cpo_bc_weight,
                    reduction="none",
                )

                if cfg.routing_mode == "legacy-scalar":
                    loss_pref = pref_fraction * pref_vec.mean()
                else:
                    loss_pref = masked_mean(pref_vec, pref_mask)

                loss = cfg.pref_weight * loss_pref + cfg.lambda_reg * loss_nce

            elif args.loss_type == "pref-infonce":
                if cfg.pref_cross_gpu_negatives:
                    all_pred_pref = gather_with_grad(pred_emb, accelerator)
                    all_win_pref = gather_with_grad(win_emb, accelerator)
                    all_lose_pref = gather_with_grad(lose_emb, accelerator)

                    global_vec = preference_infonce_loss(
                        all_pred_pref,
                        all_win_pref,
                        all_lose_pref,
                        temperature=cfg.temperature,
                        use_in_batch_negatives=cfg.pref_use_in_batch_negatives,
                        reduction="none",
                    )

                    local_batch = pred_emb.shape[0]
                    start = accelerator.process_index * local_batch
                    end = start + local_batch
                    pref_vec = global_vec[start:end]
                else:
                    pref_vec = preference_infonce_loss(
                        pred_emb,
                        win_emb,
                        lose_emb,
                        temperature=cfg.temperature,
                        use_in_batch_negatives=cfg.pref_use_in_batch_negatives,
                        reduction="none",
                    )

                if cfg.routing_mode == "legacy-scalar":
                    loss_pref = pref_fraction * pref_vec.mean()
                else:
                    loss_pref = masked_mean(pref_vec, pref_mask)

                loss = cfg.pref_weight * loss_pref + cfg.lambda_reg * loss_nce

            else:
                raise ValueError(f"Unsupported loss_type: {args.loss_type}")

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()

            if accelerator.is_main_process:
                running_total += loss.item()
                running_nce += loss_nce.item()
                running_pref += loss_pref.item() if loss_pref.ndim == 0 else float(loss_pref.mean().item())
                running_pref_fraction += pref_fraction.item()

                progress.set_postfix(
                    loss=f"{running_total / (step + 1):.4f}",
                    nce=f"{running_nce / (step + 1):.4f}",
                    pref=f"{running_pref / (step + 1):.4f}",
                    pref_frac=f"{running_pref_fraction / (step + 1):.2f}",
                    lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                )

        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            save_path = os.path.join(cfg.output_dir, args.save_name)
            torch.save(unwrapped_model.state_dict(), save_path)
            print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()