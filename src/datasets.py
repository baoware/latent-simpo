import torch
import os
import glob
import random
from torchvision.datasets import CocoCaptions
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import torchvision.transforms as T
import webdataset as wds


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
        if image.mode != "RGB":
            image = image.convert("RGB")
        frame = self.transform(image) # Shape: [C, H, W]

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

class DataCompDataset(IterableDataset):
    def __init__(self, config):
        self.config = config
        
        # tokenizers
        self.predictor_tokenizer = AutoTokenizer.from_pretrained(config.predictor_source)
        if self.predictor_tokenizer.pad_token is None:
            self.predictor_tokenizer.pad_token = self.predictor_tokenizer.eos_token
            
        self.y_encoder_tokenizer = AutoTokenizer.from_pretrained(config.y_encoder_source)
        
        # transforms
        self.transform = T.Compose([
            T.Resize((config.resolution, config.resolution)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("Initializing DataComp-Small (WebDataset Stream)...")
        
        current_file_dir = os.path.dirname(os.path.abspath(__file__)) # .../latent-simpo/src
        project_root = os.path.dirname(current_file_dir)              # .../latent-simpo
        
        # construct the exact absolute path to the shards
        tar_pattern = os.path.join(project_root, "data", "datacomp_small_dataset", "shards", "*.tar")
        
        print(f"Looking for files at: {tar_pattern}")
        
        # glob to find files
        tar_files = sorted(glob.glob(tar_pattern))
        
        if len(tar_files) == 0:
            raise FileNotFoundError(f"Could not find any .tar files! Checked exact path: {tar_pattern}")
            
        print(f"Found {len(tar_files)} tar shards.")
        
        self.dataset = (
            wds.WebDataset(tar_files, resampled=True, nodesplitter=wds.split_by_node)
            .shuffle(1000)
            .decode("pil")
            .to_tuple("jpg", "txt")
        )

    def prepare_video(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        frame = self.transform(image)

        # [3, H, W] to [3, 16, H, W]
        return torch.stack([frame] * self.config.num_frames, dim=0)

    def prepare_text(self, text, tokenizer, max_len):
        return tokenizer(
            text, 
            return_tensors='pt', 
            padding='max_length', 
            max_length=max_len, 
            truncation=True
        )

    def __iter__(self):
        # iterate over the webdataset stream
        for image, caption in self.dataset:
            try:
                # make sure the caption is valid text
                if not caption or not isinstance(caption, str): 
                    continue
                
                # vision input
                video = self.prepare_video(image)
                
                # query input
                q_tok = self.prepare_text("Describe this image:", self.predictor_tokenizer, 16)
                
                # target input
                target_text = f"task: sentence similarity | query: {caption}"
                t_tok = self.prepare_text(target_text, self.y_encoder_tokenizer, self.config.max_seq_len)
                
                yield {
                    "video": video,
                    "q_ids": q_tok.input_ids.squeeze(0),
                    "t_ids": t_tok.input_ids.squeeze(0),
                    "t_mask": t_tok.attention_mask.squeeze(0)
                }
            except Exception as e:
                print(f"Data error: {e}")
                continue