import torch
import os
import random
from torchvision.datasets import CocoCaptions
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import torchvision.transforms as T


class BaseJEPADataset(Dataset):
    def __init__(self, config):
        self.config = config
        
        # tokenizer for Predictor
        self.predictor_tokenizer = AutoTokenizer.from_pretrained(config.predictor_source)
        self.predictor_tokenizer.pad_token = self.predictor_tokenizer.eos_token
        
        # tokenizer for the Y-Encoder
        self.y_encoder_tokenizer = AutoTokenizer.from_pretrained(config.y_encoder_source)
        
        self.transform = T.Compose([
            T.Resize((config.resolution, config.resolution)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def prepare_video(self, image):
        frame = self.transform(image.convert("RGB"))
        return torch.stack([frame] * self.config.num_frames, dim=1)

    def prepare_text(self, text, tokenizer, max_len):
        return tokenizer(text, return_tensors='pt', padding='max_length', 
                         max_length=max_len, truncation=True)

class COCODataset(BaseJEPADataset):
    def __init__(self, config, split='train'):
        super().__init__(config)
        
        root_dir = os.path.join(config.data_dir, "coco")
        
        if split == 'train':
            img_path = os.path.join(root_dir, "train2014")
            ann_path = os.path.join(root_dir, "annotations/captions_train2014.json")
        else:
            img_path = os.path.join(root_dir, "val2014")
            ann_path = os.path.join(root_dir, "annotations/captions_val2014.json")
        print(f"Loading local COCO from {img_path}...")
        self.coco = CocoCaptions(root=img_path, annFile=ann_path)
        print("----------")

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, idx):
        image, captions = self.coco[idx]
        
        # vision input
        video = self.prepare_video(image)
        
        # query input
        q_tok = self.prepare_text("Describe this image:", self.predictor_tokenizer, 16)
        
        # target input (randomly pick one of the valid captions)
        selected_caption = random.choice(captions)
        
        # Add the mandatory instruction prefix
        formatted_caption = f"task: sentence similarity | query: {selected_caption}"
        
        t_tok = self.prepare_text(
            formatted_caption, 
            self.y_encoder_tokenizer, 
            self.config.max_seq_len
        )
        
        return {
            "video": video,
            "q_ids": q_tok.input_ids.squeeze(0),
            "t_ids": t_tok.input_ids.squeeze(0),
            "t_mask": t_tok.attention_mask.squeeze(0)
        }