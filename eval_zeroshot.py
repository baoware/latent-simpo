import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
import torchvision.transforms as T
from tqdm import tqdm
from src.config import Config
from src.model import VL_JEPA
from transformers import AutoTokenizer
from datasets import load_dataset
import os
import argparse

class ImageNetStreamingDataset(IterableDataset):
    def __init__(self, config):
        self.config = config
        self.transform = T.Compose([
            T.Resize((config.resolution, config.resolution)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("Initializing ImageNet-1k validation stream...")
        print("----------")
        self.dataset = load_dataset("imagenet-1k", split="validation", streaming=True, trust_remote_code=True)
        self.classes = self.dataset.features['label'].names

    def prepare_video(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        frame = self.transform(image)
        # duplicate frames to match V-JEPA input [C, T, H, W]
        return torch.stack([frame] * self.config.num_frames, dim=1)

    def __iter__(self):
        for item in self.dataset:
            try:
                image = item['image']
                label = item['label']
                video = self.prepare_video(image)
                yield video, label
            except Exception:
                continue

"""
    pre-computes the Y-Encoder embeddings for all class names
"""
def get_class_embeddings(model, class_names, tokenizer, device):
    print("Pre-computing class embeddings...")
    print("----------")
    embeddings = []
    
    templates =[lambda c: f"A photo of a {c}."]
    
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
            for c in tqdm(class_names):
                c_embeds =[]
                for temp in templates:
                    text = temp(c)
                    tokens = tokenizer(text, return_tensors='pt', padding='max_length', max_length=16, truncation=True)
                    
                    input_ids = tokens.input_ids.to(device)
                    mask = tokens.attention_mask.to(device)
                    
                    emb = model.forward_y_encoder(input_ids, mask)
                    c_embeds.append(emb)
                
                c_stack = torch.stack(c_embeds).mean(dim=0)
                c_stack = F.normalize(c_stack.float(), p=2, dim=-1) # keep embeddings in float32 for metric precision
                embeddings.append(c_stack)
            
    return torch.cat(embeddings, dim=0)

def main(checkpoint_name):
    cfg = Config()
    
    print(f"Zero-shot classification evaluation...")
    print(f"Device: {cfg.device}")
    print("----------")
    
    # load model
    model = VL_JEPA(cfg).to(cfg.device)
    
    # load checkpoint
    ckpt_path = os.path.join(cfg.output_dir, checkpoint_name)
    if os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path))
    else:
        print("No checkpoint found. Evaluating random init.")

    # prepare validation data
    val_dataset = ImageNetStreamingDataset(cfg)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)
    class_names = val_dataset.classes
    
    # pre-compute target embeddings
    y_tokenizer = AutoTokenizer.from_pretrained(cfg.y_encoder_source)
    class_embeds = get_class_embeddings(model, class_names, y_tokenizer, cfg.device)
    
    # evaluation Loop
    print("Running zero-shot evaluation...")
    print("----------")
    correct = 0
    total = 0
    
    # prepare query prompt
    q_tokenizer = AutoTokenizer.from_pretrained(cfg.predictor_source)
    q_tokenizer.pad_token = q_tokenizer.eos_token
    query_text = "Describe this image:"
    q_tokens = q_tokenizer(query_text, return_tensors='pt', padding='max_length', max_length=16)
    q_ids = q_tokens.input_ids.repeat(32, 1).to(cfg.device) 
    
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if "cuda" in cfg.device else "cpu", dtype=torch.bfloat16):
            for video, labels in tqdm(val_loader):
                video = video.to(cfg.device)
                labels = labels.to(cfg.device)
                current_batch = video.size(0)
                
                # adjust q_ids batch size for the last batch
                curr_q_ids = q_ids[:current_batch]
                
                # predict
                pred_emb = model.forward_predictor(video, curr_q_ids).float() # cast to f32 for multiplication
                
                # compare with class embeddings
                logits = torch.matmul(pred_emb, class_embeds.T)
                
                # accuracy
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += current_batch
                
                # Optional: limit evaluation to 5000 images to save time
                if total >= 5000:
                    break
            
    acc = correct / total
    print(f"Zero-shot accuracy ({checkpoint_name}): {acc*100:.2f}%")
    print(f"Random baseline: {100/len(class_names):.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="datacomp_baseline_epoch_1.pt", help="Name of checkpoint in checkpoints/")
    args = parser.parse_args()
    
    main(args.ckpt)