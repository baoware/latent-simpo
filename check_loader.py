import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
import argparse
from dotenv import load_dotenv

from src.config import Config
from src.datasets import DataCompDataset, COCODataset, RLHFDataset, SafeVLDataset, VQADataset

def check_loader(dataset_choice):
    load_dotenv()
    
    cfg = Config()
    
    print(f"Data loader test...")
    print(f"Target dataset: {dataset_choice}")
    print(f"Batch size:     {cfg.batch_size_base}")
    print(f"Num frames:     {cfg.num_frames}")
    print(f"Tokenizer:      {cfg.predictor_source}")
    print("----------")
    
    # Initialize Dataset
    try:
        if dataset_choice == 'datacomp':
            dataset = DataCompDataset(cfg)
            dataloader = DataLoader(dataset, batch_size=cfg.batch_size_base, num_workers=4)
        elif dataset_choice == 'coco':
            dataset = COCODataset(cfg, split='train')
            dataloader = DataLoader(dataset, batch_size=cfg.batch_size_base, num_workers=2)
        elif dataset_choice == 'rlhf':
            dataset = RLHFDataset(cfg, split='train')
            dataloader = DataLoader(dataset, batch_size=cfg.batch_size_base, num_workers=2)
        elif dataset_choice == 'safe':
            dataset = SafeVLDataset(cfg, split='train')
            dataloader = DataLoader(dataset, batch_size=cfg.batch_size_base, num_workers=2)
        elif dataset_choice == 'vqa':
            dataset = VQADataset(cfg, split='train')
            dataloader = DataLoader(dataset, batch_size=cfg.batch_size_base, num_workers=2)
        else:
            print(f"Unknown dataset: {dataset_choice}")
            return
    except Exception as e:
        print(f"Could not initialize dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Attempting to fetch 5 batches...")
    print("----------")
    start_time = time.time()
    
    try:
        iterator = iter(dataloader)
        for i in range(5):
            batch_t0 = time.time()
            batch = next(iterator)
            batch_time = time.time() - batch_t0
            
            video = batch['video']
            q_ids = batch['q_ids']
            
            print(f"  Batch {i+1}: OK ({batch_time:.3f}s) | "
                  f"Video: {list(video.shape)} | Query: {list(q_ids.shape)}")
            
            expected_video =[cfg.batch_size_base, 3, cfg.num_frames, cfg.resolution, cfg.resolution]
            if list(video.shape) != expected_video:
                print(f"Video shape mismatch! Expected {expected_video}")

            if dataset_choice in['rlhf', 'safe', 'vqa']:
                win_ids = batch['win_ids']
                lose_ids = batch['lose_ids']
                print(f"    Win IDs: {list(win_ids.shape)} | Lose IDs: {list(lose_ids.shape)}")

    except Exception as e:
        print(f"Error during iteration: {e}")
        import traceback
        traceback.print_exc()
        return
        
    total_time = time.time() - start_time
    avg_time = total_time / 5
    
    print(f"Loader works...")
    print(f"Average Batch Time: {avg_time:.3f}s")
    print("----------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="vqa", choices=["datacomp", "coco", "rlhf", "safe", "vqa"])
    args = parser.parse_args()
    
    check_loader(args.dataset)