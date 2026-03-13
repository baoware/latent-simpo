import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv

from src.config import Config
from src.datasets import DataCompDataset, COCODataset

def check_loader():
    load_dotenv()
    
    cfg = Config()
    
    print(f"Data loader test...")
    print(f"Target dataset: {cfg.dataset_name}")
    print(f"Batch size:     {cfg.batch_size_base}")
    print(f"Num frames:     {cfg.num_frames}")
    print(f"Tokenizer:      {cfg.predictor_source}")
    print("----------")
    
    # initialize dataset
    try:
        if cfg.dataset_name == 'datacomp':
            print("Initializing DataComp dataset...")
            dataset = DataCompDataset(cfg)
            # workers > 0 required for streaming performance
            dataloader = DataLoader(dataset, batch_size=cfg.batch_size_base, num_workers=4)
        elif cfg.dataset_name == 'coco':
            print("Initializing COCO...")
            dataset = COCODataset(cfg, split='train')
            dataloader = DataLoader(dataset, batch_size=cfg.batch_size_base, num_workers=2)
        else:
            print(f"Unknown dataset: {cfg.dataset_name}")
            return
    except Exception as e:
        print(f"Could not initialize dataset: {e}")
        return

    print("Attempting to fetch 10 batches...")
    start_time = time.time()
    
    try:
        iterator = iter(dataloader)
        for i in range(10):
            batch_t0 = time.time()
            batch = next(iterator)
            batch_time = time.time() - batch_t0
            
            video = batch['video']
            q_ids = batch['q_ids']
            t_ids = batch['t_ids']
            
            print(f"  Batch {i+1}: OK ({batch_time:.3f}s) | "
                  f"Video: {list(video.shape)} | Text: {list(t_ids.shape)}")
            
            # Sanity Check Dimensions
            expected_video = [cfg.batch_size_base, 3, cfg.num_frames, cfg.resolution, cfg.resolution]
            if list(video.shape) != expected_video:
                print(f"Video shape mismatch! Expected {expected_video}")

    except Exception as e:
        print(f"Error during iteration: {e}")
        return
        
    total_time = time.time() - start_time
    avg_time = total_time / 10
    
    print("----------")
    print(f"Loader works...")
    print(f"Average Batch Time: {avg_time:.3f}s")
    print(f"Est. Epoch Time (12.8M imgs): {(12800000/cfg.batch_size_base * avg_time) / 3600:.1f} hours")
    print("----------")

if __name__ == "__main__":
    check_loader()