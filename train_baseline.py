import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time

from src.config import Config
from src.model import VL_JEPA
from src.datasets import COCODataset
from src.losses import infonce_loss

from dotenv import load_dotenv
import huggingface_hub

load_dotenv() 
token = os.getenv("HF_TOKEN")
huggingface_hub.login(token=token)

def main():
    cfg = Config()
    print(f"Baseline training...")
    print(f"Device: {cfg.device}")
    print(f"Model: {cfg.x_encoder_source} + {cfg.predictor_source}")
    print("----------")
    
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    # initiallize architecture
    model = VL_JEPA(cfg).to(cfg.device)
    
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
    dataset = COCODataset(cfg, split='train')
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size_base, shuffle=True, num_workers=0)

    # training loop
    model.train()
    
    for epoch in range(cfg.epochs_base):
        total_loss = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs_base}")
        
        for batch_index, batch in enumerate(progress):
            # move to gpu
            video = batch['video'].to(cfg.device)
            q_ids = batch['q_ids'].to(cfg.device)
            t_ids = batch['t_ids'].to(cfg.device)
            t_mask = batch['t_mask'].to(cfg.device)
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # forward pass to get embeddings
                pred_emb = model.forward_predictor(video, q_ids)
                target_emb = model.forward_y_encoder(t_ids, t_mask)
                
                # InfoNCE loss
                loss = infonce_loss(pred_emb, target_emb, temperature=cfg.temperature)
            
            # backward and step
            loss.backward()
            optimizer.step()
            
            # logging
            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # save interim checkpoints every 500 steps
            if batch_index > 0 and batch_index % 500 == 0:
                torch.save(model.state_dict(), os.path.join(cfg.output_dir, "latest_checkpoint.pt"))

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")
        
        # save epoch checkpoint
        save_path = os.path.join(cfg.output_dir, f"baseline_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")
        print("----------")

if __name__ == "__main__":
    main()