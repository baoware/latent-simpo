import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
from src.config import Config
from src.model import VL_JEPA
from transformers import AutoTokenizer

class ImageNetStreamingDataset(IterableDataset):
    def __init__(self, config):
        self.config = config
        self.transform = T.Compose([
            T.Resize((config.resolution, config.resolution)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # load ImageNet-1k validation set in streaming Mode
        # prevents downloading 150GB to hpc
        print("Initializing ImageNet-1k validation stream...")
        self.dataset = load_dataset("imagenet-1k", split="validation", streaming=True, trust_remote_code=True)
        
        # extract class names from features before iterating
        self.classes = self.dataset.features['label'].names

    def prepare_video(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        frame = self.transform(image)
        # duplicate frames to match V-JEPA input [3, 16, H, W]
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
    format: "a photo of a {class_name}"
"""
def get_class_embeddings(model, class_names, tokenizer, device):
    print("Pre-computing class embeddings...")
    embeddings = []
    
    # simple prompt template
    templates = [lambda c: f"A photo of a {c}."]
    
    model.eval()
    with torch.no_grad():
        for c in tqdm(class_names):
            # average embedding over templates (if multiple)
            c_embeds = []
            for temp in templates:
                text = temp(c)
                tokens = tokenizer(text, return_tensors='pt', padding='max_length', max_length=16, truncation=True)
                
                input_ids = tokens.input_ids.to(device)
                mask = tokens.attention_mask.to(device)
                
                # get Y-Encoder embedding
                emb = model.forward_y_encoder(input_ids, mask)
                c_embeds.append(emb)
            
            # stack and mean
            c_stack = torch.stack(c_embeds).mean(dim=0)
            c_stack = F.normalize(c_stack, p=2, dim=-1)
            embeddings.append(c_stack)
            
    # shape: [Num_Classes, Dim]
    return torch.cat(embeddings, dim=0)

def main():
    cfg = Config()
    
    # load model and weights
    model = VL_JEPA(cfg).to(cfg.device)
    
    # load your checkpoint
    ckpt_path = f"{cfg.output_dir}/baseline_epoch_2.pt" 
    if os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path))
    else:
        print("No checkpoint found. Running with random init (expect ~1% acc)")

    # prepare validation data
    print("Loading CIFAR-100 test set...")
    transform = T.Compose([
        T.Resize((cfg.resolution, cfg.resolution)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # downloads automatically to ./data
    val_dataset = ImageNetStreamingDataset(cfg)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # get class names
    class_names = val_dataset.classes
    
    # pre-compute target embeddings
    # get the tokenizer for the Y-Encoder
    y_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', cfg.y_encoder_source) 
    y_tokenizer = AutoTokenizer.from_pretrained(cfg.y_encoder_source)
    
    class_embeds = get_class_embeddings(model, class_names, y_tokenizer, cfg.device)
    
    # evaluation Loop
    print("Running zero-shot evaluation...")
    correct = 0
    total = 0
    
    # query for the Predictor
    # the Predictor needs a prompt to know "what" to extract
    # for classification, ask it to describe the image
    q_tokenizer = AutoTokenizer.from_pretrained(cfg.predictor_source)
    q_tokenizer.pad_token = q_tokenizer.eos_token
    query_text = "Describe this image:"
    q_tokens = q_tokenizer(query_text, return_tensors='pt', padding='max_length', max_length=16)
    q_ids = q_tokens.input_ids.repeat(32, 1).to(cfg.device) # batch repeat
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)
            current_batch = images.size(0)
            
            # predict
            # adjust q_ids batch size for the last batch
            curr_q_ids = q_ids[:current_batch]
            pred_emb = model.forward_predictor(video, curr_q_ids) # [B, Dim]
            
            # compare with class embeddings
            # [B, Dim] @ [Classes, Dim].T -> [B, Classes]
            logits = torch.matmul(pred_emb, class_embeds.T)
            
            # accuracy
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += current_batch
            
    acc = correct / total
    print(f"----------")
    print(f"Zero-shot accuracy: {acc*100:.2f}%")
    print(f"Random baseline: {100/len(class_names):.2f}%")

if __name__ == "__main__":
    main()