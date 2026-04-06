import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
from accelerate import Accelerator

from src.config import Config
from src.model import VL_JEPA
from src.losses import infonce_loss
from src.datasets import COCODataset, DataCompDataset

from dotenv import load_dotenv
import huggingface_hub

load_dotenv() 
token = os.getenv("HF_TOKEN")
if token:
    huggingface_hub.login(token=token)

def main():
    accelerator = Accelerator(mixed_precision="bf16")

    cfg = Config()
    if accelerator.is_main_process:
        print(f"Baseline training on {accelerator.num_processes} GPUs...")
        print(f"Model: {cfg.x_encoder_source} + {cfg.predictor_source}")
        print("----------")
        if not os.path.exists(cfg.output_dir):
            os.makedirs(cfg.output_dir)

    # initiallize architecture
    model = VL_JEPA(cfg)
    
    # optimizer with differential learning rates
    # Predictor gets normal LR, Y-Encoder gets 0.05 * LR
    optimizer = optim.AdamW([
        {'params': filter(lambda p: p.requires_grad, model.predictor_model.parameters()), 'lr': cfg.lr_predictor},
        {'params': model.predictor_head.parameters(), 'lr': cfg.lr_predictor},
        {'params': model.y_encoder.parameters(), 'lr': cfg.lr_y_encoder},
        {'params': model.y_proj.parameters(), 'lr': cfg.lr_y_encoder}
    ])

    # load data
    print("Loading dataset...")
    print("----------")
    if cfg.dataset_name == 'datacomp':
        dataset = DataCompDataset(cfg)
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size_base, num_workers=0)
        total_steps = (12800000 // cfg.batch_size_base) // accelerator.num_processes
    else:
        dataset = COCODataset(cfg, split='train')
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size_base, shuffle=True, num_workers=0)
        
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    if cfg.dataset_name != 'datacomp':
        total_steps = len(dataloader)

    # training loop
    model.train()
    
    for epoch in range(cfg.epochs_base):
        total_loss = 0
        progress = tqdm(dataloader, total=total_steps, desc=f"Epoch {epoch+1}/{cfg.epochs_base}", disable=not accelerator.is_main_process)
        
        for batch_index, batch in enumerate(progress):
            # move to gpu
            video = batch['video'].to(accelerator.device)
            q_ids = batch['q_ids'].to(accelerator.device)
            t_ids = batch['t_ids'].to(accelerator.device)
            t_mask = batch['t_mask'].to(accelerator.device)
            
            optimizer.zero_grad()
            
            if accelerator.num_processes > 1:
                pred_emb = model.module.forward_predictor(video, q_ids)
                target_emb = model.module.forward_y_encoder(t_ids, t_mask)
            else:
                pred_emb = model.forward_predictor(video, q_ids)
                target_emb = model.forward_y_encoder(t_ids, t_mask)
            
            # calculate loss
            loss = infonce_loss(pred_emb, target_emb, temperature=cfg.temperature)

            # backward and step
            accelerator.backward(loss)
            optimizer.step()
            
            # logging
            if accelerator.is_main_process:
                total_loss += loss.item()
                progress.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # save interim checkpoints
                if batch_index > 0 and batch_index % 500 == 0:
                    # unwrap the model to save raw weights, not DDP wrappers
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save(unwrapped_model.state_dict(), os.path.join(cfg.output_dir, "latest_checkpoint.pt"))

            if cfg.dataset_name == 'datacomp' and batch_index >= total_steps - 1:
                break


        if accelerator.is_main_process:
            avg_loss = total_loss / total_steps
            print(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")
            
            unwrapped_model = accelerator.unwrap_model(model)
            save_path = os.path.join(cfg.output_dir, f"baseline_epoch_{epoch+1}.pt")
            torch.save(unwrapped_model.state_dict(), save_path)
            print(f"Saved: {save_path}")
            print("----------")

if __name__ == "__main__":
    main()