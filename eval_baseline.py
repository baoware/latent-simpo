import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import argparse


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.model import VL_JEPA
from src.datasets import COCODataset

def evaluate(checkpoint_name):
    cfg = Config()
    
    eval_batch_size = 32
    
    print(f"Retrieval evaluation...")
    print(f"Device: {cfg.device}")
    print("----------")
    
    # initialize model
    model = VL_JEPA(cfg).to(cfg.device)
    
    # load the requested weights
    checkpoint_path = os.path.join(cfg.output_dir, checkpoint_name)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=cfg.device)
        model.load_state_dict(state_dict)
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found.")
        print("Evaluating with random weights (Expect ~0% accuracy).")

    model.eval()
    
    # load validation data
    print("Loading validation dataset (COCO val2014)...")
    try:
        # full_dataset = COCODataset(cfg, split='val')
        # indices = list(range(5000))
        # dataset = data.Subset(full_dataset, indices)
        # use num_workers=0 to prevent multiprocessing crashes during eval
        # dataloader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False, num_workers=4)

        dataset = COCODataset(cfg, split='val')
        dataloader = DataLoader(
            dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    except Exception as e:
        print(f"Error loading validation set: {e}")
        print("Did you download and unzip 'val2014.zip' into data/coco/?")
        return

    print(f"Evaluating on {len(dataset)} samples...")
    print("----------")
    
    img_embeds = []
    text_embeds =[]
    
    # generate embeddings
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if "cuda" in cfg.device else "cpu", dtype=torch.bfloat16):
            for batch in tqdm(dataloader, desc="Encoding"):
                # move to gpu
                video = batch['video'].to(cfg.device, non_blocking=True)

                # safety
                if video.shape[1] == cfg.num_frames and video.shape[2] == 3:
                    video = video.permute(0, 2, 1, 3, 4)
                
                q_ids = batch['q_ids'].to(cfg.device, non_blocking=True)
                t_ids = batch['t_ids'].to(cfg.device, non_blocking=True)
                t_mask = batch['t_mask'].to(cfg.device, non_blocking=True)
                
                # get image embeddings
                v_emb = model.forward_predictor(video, q_ids)
                
                # get text embeddings
                t_emb = model.forward_y_encoder(t_ids, t_mask)
                
                img_embeds.append(v_emb.cpu().float())
                text_embeds.append(t_emb.cpu().float())

    # concatenate all into one massive tensor
    img_embeds = torch.cat(img_embeds, dim=0)   # [N, Dim]
    text_embeds = torch.cat(text_embeds, dim=0) # [N, Dim]
    
    # compute metrics (Recall@K)
    img_embeds = F.normalize(img_embeds, p=2, dim=-1)
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)

    img_embeds = img_embeds.to(cfg.device)
    text_embeds = text_embeds.to(cfg.device)
    
    # compute similarity matrix
    print("Computing Similarity Matrix...")
    print("----------")
    

    n_samples = img_embeds.shape[0]

    r1 = 0
    r5 = 0
    r10 = 0

    chunk_size = 512

    for start in tqdm(range(0, n_samples, chunk_size), desc="Retrieval"):

        end = min(start + chunk_size, n_samples)
        img_chunk = img_embeds[start:end]

        sim = torch.matmul(img_chunk, text_embeds.T)
        _, topk = torch.topk(sim, k=10, dim=1)

        correct = torch.arange(start, end, device=cfg.device).unsqueeze(1)

        r1 += (topk[:, :1] == correct).any(dim=1).sum().item()
        r5 += (topk[:, :5] == correct).any(dim=1).sum().item()
        r10 += (topk[:, :10] == correct).any(dim=1).sum().item()
        
    print(f"\nRetrieval results ({checkpoint_name}):")
    print(f"Total samples: {n_samples}")
    print(f"R@1:  {r1/n_samples:.2%} (Chance: {1/n_samples:.2%})")
    print(f"R@5:  {r5/n_samples:.2%}")
    print(f"R@10: {r10/n_samples:.2%}")
    print(f"----------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="datacomp_baseline_epoch_1.pt", help="Name of checkpoint in checkpoints/")
    args = parser.parse_args()
    
    evaluate(args.ckpt)