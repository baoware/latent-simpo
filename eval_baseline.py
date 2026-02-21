import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.model import VL_JEPA
from src.datasets import COCODataset

def evaluate():
    cfg = Config()
    
    eval_batch_size = 32
    
    print(f"Baseline evaluation...")
    print(f"Device: {cfg.device}")
    print("----------")
    
    # initialize model
    model = VL_JEPA(cfg).to(cfg.device)
    
    # load the trained weights
    # look for the last epoch saved
    checkpoint_path = os.path.join(cfg.output_dir, f"baseline_epoch_{cfg.epochs_base}.pt")
    
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
        dataset = COCODataset(cfg, split='val')
        dataloader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False, num_workers=4)
    except Exception as e:
        print(f"Error loading validation set: {e}")
        print("Did you download and unzip 'val2014.zip' into data/coco/?")
        return

    print(f"Evaluating on {len(dataset)} samples...")
    print("----------")
    
    img_embeds = []
    text_embeds = []
    
    # generate embeddings
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding"):
            # move to gpu
            video = batch['video'].to(cfg.device)
            q_ids = batch['q_ids'].to(cfg.device)
            t_ids = batch['t_ids'].to(cfg.device)
            t_mask = batch['t_mask'].to(cfg.device)
            
            # get image embeddings
            v_emb = model.forward_predictor(video, q_ids)
            
            # get text embeddings
            t_emb = model.forward_y_encoder(t_ids, t_mask)
            
            img_embeds.append(v_emb.cpu())
            text_embeds.append(t_emb.cpu())

    # concatenate all into one massive tensor
    img_embeds = torch.cat(img_embeds, dim=0)   # [N, Dim]
    text_embeds = torch.cat(text_embeds, dim=0) # [N, Dim]
    
    # compute metrics (Recall@K)
    # normalize again just to be safe (though model output is already normalized)
    img_embeds = F.normalize(img_embeds, p=2, dim=-1)
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)
    
    # compute similarity matrix
    print("Computing Similarity Matrix...")
    print("----------")
    sim_matrix = torch.matmul(img_embeds, text_embeds.T)
    
    n_samples = sim_matrix.shape[0]
    
    # for each image, the correct caption index is its own index (diagonal)
    labels = torch.arange(n_samples)
    
    # get top-k predictions for each image
    # dim=1 means look across all text columns for each image row
    _, topk_indices = torch.topk(sim_matrix, k=10, dim=1)
    
    r1 = 0
    r5 = 0
    r10 = 0
    
    for i in range(n_samples):
        # is the correct label (i) inside the top-k predictions?
        if i in topk_indices[i, :1]: r1 += 1
        if i in topk_indices[i, :5]: r5 += 1
        if i in topk_indices[i, :10]: r10 += 1
        
    print(f"\nBaseline retrieval results:")
    print(f"Total samples: {n_samples}")
    print(f"R@1:  {r1/n_samples:.2%} (Chance: {1/n_samples:.2%})")
    print(f"R@5:  {r5/n_samples:.2%}")
    print(f"R@10: {r10/n_samples:.2%}")
    print(f"----------")

if __name__ == "__main__":
    evaluate()