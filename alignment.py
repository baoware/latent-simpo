import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
from accelerate import Accelerator
from dotenv import load_dotenv
import huggingface_hub

from src.config import Config
from src.model import VL_JEPA
from src.datasets import RLHFDataset
from src.losses import infonce_loss, latent_simpo_loss, triplet_margin_loss
from transformers import get_cosine_schedule_with_warmup

# authenticate
load_dotenv() 
token = os.getenv("HF_TOKEN")
if token:
    huggingface_hub.login(token=token)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss_type", type=str, required=True, choices=["infonce", "latent-simpo", "triplet-margin"], help="which objective to optimize")
    parser.add_argument("--load_from", type=str, required=True, help="path to phase 1 baseline checkpoint")
    parser.add_argument("--save_name", type=str, required=True, help="name of output checkpoint")
    args = parser.parse_args()

    # init accelerator
    accelerator = Accelerator(mixed_precision="bf16")
    cfg = Config()
    
    if accelerator.is_main_process:
        print(f"Alignment | Loss: {args.loss_type.upper()} | GPUs: {accelerator.num_processes}")
        print("----------")
        if not os.path.exists(cfg.output_dir):
            os.makedirs(cfg.output_dir)

    # initialize and load baseline
    model = VL_JEPA(cfg)
    if os.path.exists(args.load_from):
        if accelerator.is_main_process: 
            print(f"Baseline from {args.load_from}...")
            print("----------")
        state_dict = torch.load(args.load_from, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {args.load_from}")

    # optimizer (freeze y-encoder during alignment to prevent semantic drift)
    model.predictor_model.gradient_checkpointing_enable()
    
    for p in model.y_encoder.parameters(): p.requires_grad = False
    for p in model.y_proj.parameters(): p.requires_grad = False

    optimizer = optim.AdamW([
        {'params': filter(lambda p: p.requires_grad, model.predictor_model.parameters()), 'lr': cfg.lr_alignment},
        {'params': model.predictor_head.parameters(), 'lr': cfg.lr_alignment},
        {'params': model.y_encoder.parameters(), 'lr': cfg.lr_y_encoder_alignment},
        {'params': model.y_proj.parameters(), 'lr': cfg.lr_y_encoder_alignment}
    ])

    # load rlhf data
    if accelerator.is_main_process: 
        print("Loading RLHF-V Dataset...")
        print("----------")
    dataset = RLHFDataset(cfg, split='train')
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size_alignment, shuffle=True, num_workers=4)
    total_steps = len(dataloader) // accelerator.num_processes

    num_training_steps = total_steps * cfg.epochs_alignment
    num_warmup_steps = int(0.1 * num_training_steps)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # prepare with accelerate
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    # training loop
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
            
            # predictor forward
            if accelerator.num_processes > 1:
                pred_emb = model.module.forward_predictor(video, q_ids)
            else:
                pred_emb = model.forward_predictor(video, q_ids)
                
            # y-encoder forward (frozen targets)
            with torch.no_grad():
                if accelerator.num_processes > 1:
                    win_emb = model.module.forward_y_encoder(win_ids, win_mask)
                    lose_emb = model.module.forward_y_encoder(lose_ids, lose_mask)
                else:
                    win_emb = model.forward_y_encoder(win_ids, win_mask)
                    lose_emb = model.forward_y_encoder(lose_ids, lose_mask)
            
            if args.loss_type == "infonce":
                # pure infonce fine-tuning (only looks at the 'winner')
                all_pred = accelerator.gather(pred_emb)
                all_win = accelerator.gather(win_emb)
                loss = infonce_loss(all_pred, all_win, temperature=cfg.temperature)
                
            elif args.loss_type == "triplet-margin":
                trip = triplet_margin_loss(pred_emb, win_emb, lose_emb, margin=cfg.gamma)
                
                # infonce regularizer
                all_pred = accelerator.gather(pred_emb)
                all_win = accelerator.gather(win_emb)
                reg = infonce_loss(all_pred, all_win, temperature=cfg.temperature)
                loss = trip + (cfg.lambda_reg * reg)
                
            elif args.loss_type == "latent-simpo":
                simpo = latent_simpo_loss(pred_emb, win_emb, lose_emb, beta=cfg.beta, gamma=cfg.gamma)
                
                # infonce regularizer
                all_pred = accelerator.gather(pred_emb)
                all_win = accelerator.gather(win_emb)
                reg = infonce_loss(all_pred, all_win, temperature=cfg.temperature)
                loss = simpo + (cfg.lambda_reg * reg)

            # backward and step
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # logging
            if accelerator.is_main_process:
                total_loss += loss.item()
                progress.set_postfix({"loss": f"{loss.item():.4f}"})

        # end of epoch save
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            save_path = os.path.join(cfg.output_dir, args.save_name)
            torch.save(unwrapped_model.state_dict(), save_path)
            print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()