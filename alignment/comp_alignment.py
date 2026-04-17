import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
import os
import argparse
from accelerate import Accelerator
from dotenv import load_dotenv
import huggingface_hub

from src.config import Config
from src.model import VL_JEPA
from src.datasets import (
    RLHFDataset, SafeVLDataset, VQADataset, 
    DenseCOCODataset, AOKVQADataset, ChartQADataset, DocVQADataset
)
from src.losses import infonce_loss, latent_simpo_loss, triplet_margin_loss

# Authenticate
load_dotenv() 
token = os.getenv("HF_TOKEN")
if token:
    huggingface_hub.login(token=token)

def alignment_collate(batch):
    out = {}

    keys =[
        "video",
        "q_ids",
        "win_ids",
        "win_mask",
        "lose_ids",
        "lose_mask"
    ]

    for k in keys:
        vals =[]
        for b in batch:
            if k in b:
                vals.append(b[k])
            else:
                # create dummy if missing
                vals.append(torch.zeros_like(batch[0][k]))

        out[k] = torch.stack(vals)

    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss_type", type=str, required=True, choices=["infonce", "latent-simpo", "triplet-margin"])
    parser.add_argument("--load_from", type=str, required=True, help="Path to Phase 1 baseline")
    parser.add_argument("--save_name", type=str, required=True, help="Name of output checkpoint")
    
    parser.add_argument("--beta", type=float, default=10.0, help="Reward scale (SimPO)")
    parser.add_argument("--gamma", type=float, default=0.2, help="Target margin")
    parser.add_argument("--lambda_reg", type=float, default=0.1, help="InfoNCE Regularization weight")
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="bf16")
    cfg = Config()
    
    cfg.beta = args.beta
    cfg.gamma = args.gamma
    cfg.lambda_reg = args.lambda_reg
    
    if accelerator.is_main_process:
        print(f"Comprehensive Alignment | Loss: {args.loss_type.upper()} | GPUs: {accelerator.num_processes}")
        print(f"Beta: {cfg.beta} | Gamma: {cfg.gamma} | Lambda: {cfg.lambda_reg}")
        print("----------")
        if not os.path.exists(cfg.output_dir): os.makedirs(cfg.output_dir)

    # load model and baseline
    model = VL_JEPA(cfg)
    if os.path.exists(args.load_from):
        if accelerator.is_main_process: 
            print(f"Loading Phase 1 Baseline from {args.load_from}...")
            print("----------")
        state_dict = torch.load(args.load_from, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {args.load_from}")

    model.predictor_model.gradient_checkpointing_enable()
    
    optimizer = optim.AdamW([
        {'params': filter(lambda p: p.requires_grad, model.predictor_model.parameters()), 'lr': cfg.lr_alignment},
        {'params': model.predictor_head.parameters(), 'lr': cfg.lr_alignment},
        {'params': model.y_encoder.parameters(), 'lr': cfg.lr_y_encoder_alignment},
        {'params': model.y_proj.parameters(), 'lr': cfg.lr_y_encoder_alignment}
    ])

    # load combined rlhf dataset
    if accelerator.is_main_process: 
        print("Loading RLHF-V (Images) and PKU-SafeRLHF (Safety) Datasets...")
        print("----------")
        
    dataset_rlhf = RLHFDataset(cfg, split='train')
    dataset_safe = SafeVLDataset(cfg, split='train')
    dataset_vqa = VQADataset(cfg, split='train')
    dataset_dense = DenseCOCODataset(cfg, split='train')
    dataset_aok = AOKVQADataset(cfg, split='train')
    dataset_chart = ChartQADataset(cfg, split='train')
    dataset_doc = DocVQADataset(cfg, split='train')   

    combined_dataset = ConcatDataset([dataset_rlhf, dataset_safe, dataset_vqa, 
                                      dataset_dense, dataset_aok, dataset_chart, dataset_doc])

    len_rlhf = len(dataset_rlhf)
    len_safe = len(dataset_safe)
    len_vqa = len(dataset_vqa)
    len_dense = len(dataset_dense)
    len_aok = len(dataset_aok)
    len_chart = len(dataset_chart)
    len_doc = len(dataset_doc)

    # balance weights: 50% alignment (RLHF/Safe), 50% semantic anchors
    weights = (
        [0.25 / len_rlhf] * len_rlhf +
        [0.25 / len_safe] * len_safe +
        [0.10 / len_vqa] * len_vqa +
        [0.10 / len_dense] * len_dense +
        [0.10 / len_aok] * len_aok +
        [0.10 / len_chart] * len_chart +
        [0.10 / len_doc] * len_doc
    )

    # balance weights: 20% alignment (RLHF/Safe), 80% semantic anchors
    '''
    weights = (
        [0.10 / len_rlhf] * len_rlhf +
        [0.10 / len_safe] * len_safe +
        [0.22 / len_vqa] * len_vqa +
        [0.16 / len_dense] * len_dense +
        [0.16 / len_aok] * len_aok +
        [0.13 / len_chart] * len_chart +
        [0.13 / len_doc] * len_doc
    )
    '''

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    dataloader = DataLoader(
        combined_dataset,
        batch_size=cfg.batch_size_alignment,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=alignment_collate
    )
    
    total_steps = len(dataloader) // accelerator.num_processes

    # learning rate scheduler
    num_training_steps = total_steps * cfg.epochs_alignment
    num_warmup_steps = int(0.1 * num_training_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )

    # prepare accelerate
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    model.train()
    for epoch in range(cfg.epochs_alignment):
        total_loss = 0
        progress = tqdm(dataloader, total=total_steps, desc=f"Epoch {epoch+1}/{cfg.epochs_alignment}", disable=not accelerator.is_main_process)
        
        for batch_index, batch in enumerate(progress):
            video = batch['video'].to(accelerator.device)
            q_ids = batch['q_ids'].to(accelerator.device)
            win_ids = batch['win_ids'].to(accelerator.device)
            win_mask = batch['win_mask'].to(accelerator.device)
            lose_ids = batch['lose_ids'].to(accelerator.device)
            lose_mask = batch['lose_mask'].to(accelerator.device)
            
            optimizer.zero_grad()
            
            # Predictor forward
            if accelerator.num_processes > 1:
                pred_emb = model.module.forward_predictor(video, q_ids)
            else:
                pred_emb = model.forward_predictor(video, q_ids)
                
            # Y-Encoder forward
            if accelerator.num_processes > 1:
                win_emb = model.module.forward_y_encoder(win_ids, win_mask)
                lose_emb = model.module.forward_y_encoder(lose_ids, lose_mask)
            else:
                win_emb = model.forward_y_encoder(win_ids, win_mask)
                lose_emb = model.forward_y_encoder(lose_ids, lose_mask)

            # has_pref: local batch — 1.0 where preference pairs differ, 0.0 for semantic-anchor pairs (win==lose)
            has_pref = (win_ids != lose_ids).any(dim=-1).float()

            # Per-sample InfoNCE weight: cfg.lambda_reg for preference rows, 1.0 for anchor-only rows.
            # loss_nce is a global scalar (gathered batch); scale by the global mean weight so all ranks agree.
            dynamic_lambda = torch.where(has_pref > 0, cfg.lambda_reg, 1.0)
            nce_scale = accelerator.gather_for_metrics(dynamic_lambda).mean()

            all_pred = accelerator.gather(pred_emb)
            all_win = accelerator.gather(win_emb)
            loss_nce = infonce_loss(all_pred, all_win, temperature=cfg.temperature)

            if args.loss_type == "infonce":
                loss = nce_scale * loss_nce

            elif args.loss_type == "triplet-margin":
                loss_pref = triplet_margin_loss(pred_emb, win_emb, lose_emb, margin=cfg.gamma)
                loss = nce_scale * loss_nce + (has_pref * loss_pref).mean()

            elif args.loss_type == "latent-simpo":
                loss_pref = latent_simpo_loss(pred_emb, win_emb, lose_emb, beta=cfg.beta, gamma=cfg.gamma)
                loss = nce_scale * loss_nce + (has_pref * loss_pref).mean()

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step() # step the learning rate
            
            if accelerator.is_main_process:
                total_loss += loss.item()
                current_lr = lr_scheduler.get_last_lr()[0]
                progress.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            save_path = os.path.join(cfg.output_dir, args.save_name)
            torch.save(unwrapped_model.state_dict(), save_path)
            print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()