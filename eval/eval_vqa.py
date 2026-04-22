import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
from dotenv import load_dotenv
import huggingface_hub
from transformers import AutoTokenizer

from src.config import Config
from src.model import VL_JEPA
from src.datasets import VQADataset, ChartQADataset, DocVQADataset, AOKVQADataset

load_dotenv()
token = os.getenv("HF_TOKEN")
if token:
    huggingface_hub.login(token=token)

def evaluate_multiclass(model, cfg, dataset_class, name, split):
    # initialize dataset (using the evaluation split to trigger is_test=True)
    dataset = dataset_class(cfg, split=split)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    unique_answers = dataset.unique_answers
    
    print(f"Pre-encoding {len(unique_answers)} candidate answers for {name}...")
    print(f"----------")
    
    c_embeds =[]
    y_tokenizer = AutoTokenizer.from_pretrained(cfg.y_encoder_source)
    
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if "cuda" in cfg.device else "cpu", dtype=torch.bfloat16):
            # encode candidates in chunks to avoid OOM
            for i in tqdm(range(0, len(unique_answers), 32), desc="Encoding Candidates"):
                batch_texts =[f"task: sentence similarity | query: {ans}" for ans in unique_answers[i:i+32]]
                tokens = y_tokenizer(batch_texts, return_tensors='pt', padding='max_length', max_length=cfg.max_seq_len, truncation=True)
                
                input_ids = tokens.input_ids.to(cfg.device)
                mask = tokens.attention_mask.to(cfg.device)
                
                emb = model.forward_y_encoder(input_ids, mask).float()
                c_embeds.append(emb)
                
    # [Num_Candidates, Dim]
    candidate_embeddings = torch.cat(c_embeds, dim=0)
    candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=-1)
    
    print(f"Evaluating on {name}...")
    print(f"----------")
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if "cuda" in cfg.device else "cpu", dtype=torch.bfloat16):
            for batch in tqdm(dataloader, desc=f"{name} Eval"):
                video = batch['video'].to(cfg.device)
                q_ids = batch['q_ids'].to(cfg.device)
                answers = batch['answer_str']
                
                # predict embedding
                pred_emb = model.forward_predictor(video, q_ids).float()
                
                # similarity with all candidates: [Batch, Dim] @[Candidates, Dim].T -> [Batch, Candidates]
                logits = torch.matmul(pred_emb, candidate_embeddings.T)
                
                # get the index of the highest similarity score
                preds = torch.argmax(logits, dim=1)
                
                # check against ground truth
                for i, pred_idx in enumerate(preds):
                    predicted_text = str(unique_answers[pred_idx.item()]).strip()
                    if predicted_text.lower() == str(answers[i]).strip().lower():
                        correct += 1
                
                total += video.size(0)
                
    acc = correct / total
    chance = 1.0 / len(unique_answers) if len(unique_answers) > 0 else 0.0
    print(f"{name} Accuracy: {acc:.2%} (Random Chance: {chance:.2%})")
    print(f"----------")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint filename in checkpoints/")
    parser.add_argument("--task", type=str, required=True, choices=["vqa", "gqa", "chartqa", "docvqa", "aokvqa", "all"], help="benchmark to run")
    args = parser.parse_args()

    cfg = Config()
    
    print(f"Loading Model on {cfg.device}...")
    model = VL_JEPA(cfg).to(cfg.device)
    
    ckpt_path = os.path.join(cfg.output_dir, args.ckpt)
    if os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))
    else:
        print(f"Checkpoint {ckpt_path} not found. Running with random weights.")
        
    model.eval()
    
    # multi-class VQA evaluations
    # note: split parameters are chosen based on how they trigger `is_test=True` in src/datasets.py
    if args.task == "vqa" or args.task == "all":
        evaluate_multiclass(model, cfg, VQADataset, "VQAv2", split="validation")
        
    if args.task == "chartqa" or args.task == "all":
        evaluate_multiclass(model, cfg, ChartQADataset, "ChartQA", split="test")
        
    if args.task == "docvqa" or args.task == "all":
        evaluate_multiclass(model, cfg, DocVQADataset, "DocVQA", split="test")
        
    if args.task == "aokvqa" or args.task == "all":
        evaluate_multiclass(model, cfg, AOKVQADataset, "A-OKVQA", split="validation")

if __name__ == "__main__":
    main()