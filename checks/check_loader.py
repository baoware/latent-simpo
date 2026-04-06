import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv

from src.config import Config
from src.datasets import (
    DataCompDataset, COCODataset, RLHFDataset, 
    SafeVLDataset, VQADataset, DenseCOCODataset, 
    AOKVQADataset, ChartQADataset, DocVQADataset
)

def check_loader():
    load_dotenv()
    cfg = Config()
    
    # list of all datasets to check
    all_datasets = {
        'datacomp': DataCompDataset(cfg),
        'coco': COCODataset(cfg, split='train'),
        'rlhf': RLHFDataset(cfg, split='train'),
        'safe': SafeVLDataset(cfg, split='train'),
        'vqa': VQADataset(cfg, split='train'),
        'dense': DenseCOCODataset(cfg, split='train'),
        'aok': AOKVQADataset(cfg, split='train'),
        'chart': ChartQADataset(cfg, split='train'),
        'doc': DocVQADataset(cfg, split='train')
    }
    
    for name, dataset in all_datasets.items():
        print(f"Testing {name}")
        try:
            dataloader = DataLoader(dataset, batch_size=4, num_workers=0) # small batch for testing
            iterator = iter(dataloader)
            
            # fetch 2 batches
            for i in range(2):
                batch = next(iterator)
                video = batch['video']
                print(f"  Batch {i+1}: OK | Video Shape: {list(video.shape)}")
                
                # check for preference data
                if 'win_ids' in batch:
                    print(f"    Win shape: {list(batch['win_ids'].shape)}")
            
            print(f"{name} passed.")
            
        except Exception as e:
            print(f"{name} failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    check_loader()