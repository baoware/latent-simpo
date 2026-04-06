import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
from dotenv import load_dotenv
import huggingface_hub

from src.config import Config
from src.model import VL_JEPA
from src.datasets import POPEDataset, SugarCrepeDataset, MMSafetyDataset, VQADataset, GQADataset, ChartQADataset, DocVQADataset, AOKVQADataset
from transformers import AutoTokenizer

load_dotenv()
token = os.getenv("HF_TOKEN")
if token:
    huggingface_hub.login(token=token)

def evaluate_binary(model, cfg, dataset_class, name, pos_keys, neg_keys, is_lower_better=False):
    dataset = dataset_class(cfg)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    correct, total = 0, 0
    
    print(f"Evaluating on {name}...")
    print("----------")
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if "cuda" in cfg.device else "cpu", dtype=torch.bfloat16):
            for batch in tqdm(dataloader, desc=name):
                video = batch['video'].to(cfg.device)
                q_ids = batch['q_ids'].to(cfg.device)
                
                pos_ids, pos_mask = batch[pos_keys[0]].to(cfg.device), batch[pos_keys[1]].to(cfg.device)
                neg_ids, neg_mask = batch[neg_keys[0]].to(cfg.device), batch[neg_keys[1]].to(cfg.device)
                
                pred_emb = model.forward_predictor(video, q_ids).float()
                pos_emb = model.forward_y_encoder(pos_ids, pos_mask).float()
                neg_emb = model.forward_y_encoder(neg_ids, neg_mask).float()
                
                sim_pos = torch.sum(pred_emb * pos_emb, dim=-1)
                sim_neg = torch.sum(pred_emb * neg_emb, dim=-1)
                
                if 'label' in batch:
                    preds = (sim_pos > sim_neg).long()
                    correct += (preds == batch['label'].to(cfg.device)).sum().item()
                else:
                    correct += (sim_pos > sim_neg).sum().item()
                
                total += video.size(0)
                
    acc = correct / total
    print(f"{name} Result: {acc:.2%}" + (" (Lower is better)" if is_lower_better else ""))
    return acc

def evaluate_multiclass(model, cfg, dataset_class, name):
    dataset = dataset_class(cfg, split='val' if name in['ChartQA', 'GQA'] else 'test')
    
    # use subset for speed
    subset_indices = list(range(min(2000, len(dataset))))
    eval_dataset = data.Subset(dataset, subset_indices)
    dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    unique_answers = dataset.unique_answers
    print(f"Pre-encoding {len(unique_answers)} candidates for {name}...")
    print("----------")
    
    c_embeds =[]
    y_tokenizer = AutoTokenizer.from_pretrained(cfg.y_encoder_source)
    
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if "cuda" in cfg.device else "cpu", dtype=torch.bfloat16):
            for i in tqdm(range(0, len(unique_answers), 32), desc="Encoding Candidates"):
                batch_texts = [f"task: sentence similarity | query: {ans}" for ans in unique_answers[i:i+32]]
                tokens = y_tokenizer(batch_texts, return_tensors='pt', padding='max_length', max_length=16, truncation=True)
                emb = model.forward_y_encoder(tokens.input_ids.to(cfg.device), tokens.attention_mask.to(cfg.device)).float()
                c_embeds.append(emb)
                
    candidate_embeddings = F.normalize(torch.cat(c_embeds, dim=0), p=2, dim=-1)
    
    print(f"Evaluating {name}...")
    print("----------")
    correct, total = 0, 0
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if "cuda" in cfg.device else "cpu", dtype=torch.bfloat16):
            for batch in tqdm(dataloader, desc=name):
                video = batch['video'].to(cfg.device)
                q_ids = batch['q_ids'].to(cfg.device)
                answers = batch['answer_str']
                
                pred_emb = model.forward_predictor(video, q_ids).float()
                logits = torch.matmul(pred_emb, candidate_embeddings.T)
                preds = torch.argmax(logits, dim=1)
                
                for i, pred_idx in enumerate(preds):
                    if str(unique_answers[pred_idx.item()]).strip().lower() == str(answers[i]).strip().lower():
                        correct += 1
                total += video.size(0)
                
    acc = correct / total
    print(f"{name} Accuracy: {acc:.2%} (Chance: {1.0/len(unique_answers):.2%})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint filename")
    parser.add_argument("--task", type=str, required=True, choices=["pope", "sugarcrepe", "safety", "vqa", "gqa", "chartqa", "docvqa", "aokvqa", "all"])
    args = parser.parse_args()

    cfg = Config()
    print(f"Loading Model on {cfg.device} from {args.ckpt}...")
    print("----------")
    model = VL_JEPA(cfg).to(cfg.device)
    
    ckpt_path = os.path.join(cfg.output_dir, args.ckpt)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))
    else:
        print(f"Checkpoint {ckpt_path} not found. Using random weights.")
        
    model.eval()
    
    # binary pairwise evaluations
    if args.task in ["pope", "all"]:
        evaluate_binary(model, cfg, POPEDataset, "POPE", ["yes_ids", "yes_mask"],["no_ids", "no_mask"])
    if args.task in ["sugarcrepe", "all"]:
        evaluate_binary(model, cfg, SugarCrepeDataset, "SugarCrepe++",["pos_ids", "pos_mask"], ["neg_ids", "neg_mask"])
    if args.task in["safety", "all"]:
        acc = evaluate_binary(model, cfg, MMSafetyDataset, "MM-SafetyBench",["safe_ids", "safe_mask"], ["unsafe_ids", "unsafe_mask"])
        print(f"Attack Success Rate (ASR): {1.0 - acc:.2%} (Lower is better)")
        
    # multi-class retrieval evaluations
    if args.task in ["vqa", "all"]:
        evaluate_multiclass(model, cfg, VQADataset, "VQAv2")
    if args.task in ["gqa", "all"]:
        evaluate_multiclass(model, cfg, GQADataset, "GQA")
    if args.task in ["chartqa", "all"]:
        evaluate_multiclass(model, cfg, ChartQADataset, "ChartQA")
    if args.task in ["docvqa", "all"]:
        evaluate_multiclass(model, cfg, DocVQADataset, "DocVQA")
    if args.task in ["aokvqa", "all"]:
        evaluate_multiclass(model, cfg, AOKVQADataset, "A-OKVQA")

if __name__ == "__main__":
    main()