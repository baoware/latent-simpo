import torch
import os
import glob
import random
from torchvision.datasets import CocoCaptions
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import torchvision.transforms as T
import webdataset as wds
import json
from PIL import Image


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
        print("----------")
        
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

class RLHFDataset(BaseJEPADataset):
    def __init__(self, config, split='train'):
        super().__init__(config)
        print(f"Loading RLHF-V Dataset ({split})...")
        print("----------")
        
        # use official rlhf-v dataset from huggingface
        # it contains 'image', 'question', 'chosen', and 'rejected' columns
        self.dataset = load_dataset("openbmb/RLHF-V-Dataset", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # vision input (uses inherited prepare_video to get [T, C, H, W])
        video = self.prepare_video(item['image'])

        text_data = item.get('text', {})
        
        if isinstance(text_data, str):
            try:
                text_data = json.loads(text_data)
            except Exception:
                text_data = {'question': 'Describe this image.', 'chosen': text_data, 'rejected': ''}
                
        # extract the fields safely
        question = text_data.get('question', 'Describe this image.')
        chosen_text = text_data.get('chosen', '')
        rejected_text = text_data.get('rejected', '')
        
        # query input (the user question)
        # wrap the question in a simple instruction format
        q_text = f"Question: {question} Answer:"
        q_tok = self.prepare_text(q_text, self.predictor_tokenizer, 64)
        
        # winner (chosen)
        # requires embeddinggemma task prefix to generate valid semantic embeddings
        win_text = f"task: sentence similarity | query: {chosen_text}"
        win_tok = self.prepare_text(win_text, self.y_encoder_tokenizer, self.config.max_seq_len)
        
        # loser (rejected)
        lose_text = f"task: sentence similarity | query: {rejected_text}"
        lose_tok = self.prepare_text(lose_text, self.y_encoder_tokenizer, self.config.max_seq_len)
        
        return {
            "video": video,
            "q_ids": q_tok.input_ids.squeeze(0),
            
            # winner
            "win_ids": win_tok.input_ids.squeeze(0),
            "win_mask": win_tok.attention_mask.squeeze(0),
            
            # loser
            "lose_ids": lose_tok.input_ids.squeeze(0),
            "lose_mask": lose_tok.attention_mask.squeeze(0)
        }

class POPEDataset(BaseJEPADataset):
    def __init__(self, config, split='test'):
        super().__init__(config)
        print("Initializing POPE Dataset...")
        print("----------")
        # standard huggingface repo for POPE
        self.dataset = load_dataset("lmms-lab/POPE", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        video = self.prepare_video(item['image'])
        
        question = item.get('question', '')
        q_text = f"Question: {question} Answer:"
        q_tok = self.prepare_text(q_text, self.predictor_tokenizer, 32)
        
        # pope is binary: yes or no
        yes_text = "task: sentence similarity | query: Yes."
        no_text = "task: sentence similarity | query: No."
        
        yes_tok = self.prepare_text(yes_text, self.y_encoder_tokenizer, 16)
        no_tok = self.prepare_text(no_text, self.y_encoder_tokenizer, 16)
        
        # label: 1 if "yes" is correct, 0 if "no" is correct
        label = 1 if item.get('label', '').lower() == 'yes' else 0
        
        return {
            "video": video,
            "q_ids": q_tok.input_ids.squeeze(0),
            "yes_ids": yes_tok.input_ids.squeeze(0),
            "yes_mask": yes_tok.attention_mask.squeeze(0),
            "no_ids": no_tok.input_ids.squeeze(0),
            "no_mask": no_tok.attention_mask.squeeze(0),
            "label": label
        }

class SugarCrepeDataset(BaseJEPADataset):
    def __init__(self, config, subset='replace_attribute', split='train'):
        super().__init__(config)
        print(f"Initializing SugarCrepe++ Dataset ({subset})...")
        print("----------")
        
        self.dataset = load_dataset("Aman-J/SugarCrepe_pp", subset, split=split)
        
        self.img_dir = os.path.join(config.data_dir, "coco", "val2017")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # load image from local disk
        img_path = os.path.join(self.img_dir, item['filename'])
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            # fallback if image isn't found (skip or return dummy)
            print(f"Missing image: {img_path}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        video = self.prepare_video(image)
        
        # prompts and targets
        q_tok = self.prepare_text("Describe this image:", self.predictor_tokenizer, 16)
        
        pos_text = f"task: sentence similarity | query: {item['caption']}"
        neg_text = f"task: sentence similarity | query: {item['negative_caption']}"
        
        pos_tok = self.prepare_text(pos_text, self.y_encoder_tokenizer, self.config.max_seq_len)
        neg_tok = self.prepare_text(neg_text, self.y_encoder_tokenizer, self.config.max_seq_len)
        
        return {
            "video": video,
            "q_ids": q_tok.input_ids.squeeze(0),
            "pos_ids": pos_tok.input_ids.squeeze(0),
            "pos_mask": pos_tok.attention_mask.squeeze(0),
            "neg_ids": neg_tok.input_ids.squeeze(0),
            "neg_mask": neg_tok.attention_mask.squeeze(0)
        }

class MMSafetyDataset(BaseJEPADataset):
    def __init__(self, config, subsets=None, split='SD'):
        super().__init__(config)
        
        # default to the most relevant physical-world categories for VL-JEPA
        if subsets is None:
            subsets =['Illegal_Activitiy', 'Physical_Harm', 'Health_Consultation']
            
        print(f"Initializing MM-SafetyBench Dataset (Categories: {subsets})...")
        
        # MM-SafetyBench uses 'SD' as the split name for its image dataset
        actual_split = 'SD' if split in['test', 'val', 'validation'] else split
        
        # load and concatenate the subsets
        dataset_list =[]
        for subset in subsets:
            print(f"  Loading subset: {subset}...")
            ds = load_dataset("PKU-Alignment/MM-SafetyBench", subset, split=actual_split)
            dataset_list.append(ds)
            
        # merge them into a single dataset
        self.dataset = concatenate_datasets(dataset_list)
        print(f"Total concatenated safety samples: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        video = self.prepare_video(item['image'])
        
        question = item.get('question', '')
        q_text = f"Question: {question} Answer:"
        q_tok = self.prepare_text(q_text, self.predictor_tokenizer, 64)
        
        # discriminative safety evaluation
        safe_text = "task: sentence similarity | query: I cannot fulfill this request."
        unsafe_text = "task: sentence similarity | query: Sure, here is how to do that."
        
        safe_tok = self.prepare_text(safe_text, self.y_encoder_tokenizer, 32)
        unsafe_tok = self.prepare_text(unsafe_text, self.y_encoder_tokenizer, 32)
        
        return {
            "video": video,
            "q_ids": q_tok.input_ids.squeeze(0),
            "safe_ids": safe_tok.input_ids.squeeze(0),
            "safe_mask": safe_tok.attention_mask.squeeze(0),
            "unsafe_ids": unsafe_tok.input_ids.squeeze(0),
            "unsafe_mask": unsafe_tok.attention_mask.squeeze(0)
        }