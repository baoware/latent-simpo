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

        frame = frame.clone().detach().contiguous()

        return torch.stack([frame] * self.config.num_frames, dim=0)

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
        q_tok = self.prepare_text("Describe this image:", self.predictor_tokenizer, 64)
        
        # target input (randomly pick one of the valid captions)
        selected_caption = random.choice(captions)
        
        # add the mandatory instruction prefix
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


class DataCompDataset(BaseJEPADataset):
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
                q_tok = self.prepare_text("Describe this image:", self.predictor_tokenizer, 64)
                
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
            "win_ids": win_tok.input_ids.squeeze(0),
            "win_mask": win_tok.attention_mask.squeeze(0),
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
        q_tok = self.prepare_text(q_text, self.predictor_tokenizer, 64)
        
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
        q_tok = self.prepare_text("Describe this image:", self.predictor_tokenizer, 64)
        
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
            subsets =['Illegal_Activitiy', 'Health_Consultation', 'Sex']
            
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

class SafeVLDataset(BaseJEPADataset):
    def __init__(self, config, split='train'):
        super().__init__(config)
        print(f"Loading SPA-VL Dataset (Multimodal Safety) ({split})...")

        self.dataset = load_dataset("sqrti/SPA-VL", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # vision input
        video = self.prepare_video(item['image'])
        
        # query
        question = item.get('question', '')
        q_text = f"Question: {question} Answer:"
        q_tok = self.prepare_text(q_text, self.predictor_tokenizer, 64)
        
        # targets
        chosen = item.get('chosen', '')
        rejected = item.get('rejected', '')
            
        win_text = f"task: sentence similarity | query: {chosen}"
        win_tok = self.prepare_text(win_text, self.y_encoder_tokenizer, self.config.max_seq_len)
        
        lose_text = f"task: sentence similarity | query: {rejected}"
        lose_tok = self.prepare_text(lose_text, self.y_encoder_tokenizer, self.config.max_seq_len)
        
        return {
            "video": video,
            "q_ids": q_tok.input_ids.squeeze(0),
            "win_ids": win_tok.input_ids.squeeze(0),
            "win_mask": win_tok.attention_mask.squeeze(0),
            "lose_ids": lose_tok.input_ids.squeeze(0),
            "lose_mask": lose_tok.attention_mask.squeeze(0)
        }

class SafeRLHFDataset(BaseJEPADataset):
    def __init__(self, config, split='train'):
        super().__init__(config)
        print(f"Loading PKU-SafeRLHF Dataset ({split})...")
        # standard safety preference dataset
        self.dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # vision input is dummy black image since this is text-only
        dummy_img = Image.new('RGB', (self.config.resolution, self.config.resolution), (0, 0, 0))
        video = self.prepare_video(dummy_img)
        
        # query
        question = item.get('prompt', '')
        q_text = f"Question: {question} Answer:"
        q_tok = self.prepare_text(q_text, self.predictor_tokenizer, 64)
        
        # targets
        safer_id = item.get('safer_response_id', 0)
        
        if safer_id == 0:
            chosen = item['response_0']
            rejected = item['response_1']
        else:
            chosen = item['response_1']
            rejected = item['response_0']
            
        win_text = f"task: sentence similarity | query: {chosen}"
        win_tok = self.prepare_text(win_text, self.y_encoder_tokenizer, self.config.max_seq_len)
        
        lose_text = f"task: sentence similarity | query: {rejected}"
        lose_tok = self.prepare_text(lose_text, self.y_encoder_tokenizer, self.config.max_seq_len)
        
        return {
            "video": video,
            "q_ids": q_tok.input_ids.squeeze(0),
            "win_ids": win_tok.input_ids.squeeze(0),
            "win_mask": win_tok.attention_mask.squeeze(0),
            "lose_ids": lose_tok.input_ids.squeeze(0),
            "lose_mask": lose_tok.attention_mask.squeeze(0)
        }


class VQADataset(BaseJEPADataset):
    def __init__(self, config, split='train'):
        super().__init__(config)
        print(f"Loading VQA Mixture (H4 LLaVA)...")
        
        actual_split = 'train'
        self.dataset = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft", split=actual_split)
        
        self.is_test = split in['test', 'val', 'validation']
        
        if self.is_test:
            ans_list =[]
            for item in self.dataset.select(range(min(len(self.dataset), 2000))):
                if 'messages' in item and len(item['messages']) >= 2:
                    ans = item['messages'][1]['content']
                    if isinstance(ans, list):
                        ans = " ".join([c['text'] for c in ans if c['type'] == 'text'])
                    ans_list.append(str(ans).strip())
            self.unique_answers = list(set([a for a in ans_list if a]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        try:
            image = item['images'][0]
        except:
            from PIL import Image
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        video = self.prepare_video(image)
        
        messages = item.get('messages',[])
        question = "Describe the image."
        answer = ""
        
        if len(messages) >= 2:
            q_content = messages[0].get('content', '')
            if isinstance(q_content, list):
                q_content = " ".join([c['text'] for c in q_content if c['type'] == 'text'])
            elif not isinstance(q_content, str):
                q_content = str(q_content)
            
            a_content = messages[1].get('content', '')
            if isinstance(a_content, list):
                a_content = " ".join([c['text'] for c in a_content if c['type'] == 'text'])
            elif not isinstance(a_content, str):
                a_content = str(a_content)
                
            question = q_content.replace("<image>", "").replace("\n", " ").strip()
            answer = a_content.strip()
            
        if not question:
            question = "Describe the image."
            
        q_tok = self.prepare_text(f"Question: {question} Answer:", self.predictor_tokenizer, 64)
        
        if self.is_test:
            return {"video": video, "q_ids": q_tok.input_ids.squeeze(0), "answer_str": answer}
        
        win_tok = self.prepare_text(f"task: sentence similarity | query: {answer}", self.y_encoder_tokenizer, self.config.max_seq_len)
        return {
            "video": video,
            "q_ids": q_tok.input_ids.squeeze(0),
            "win_ids": win_tok.input_ids.squeeze(0),
            "win_mask": win_tok.attention_mask.squeeze(0),
            "lose_ids": win_tok.input_ids.squeeze(0),
            "lose_mask": win_tok.attention_mask.squeeze(0)
        }
    
class DenseCOCODataset(BaseJEPADataset):
    def __init__(self, config, split='train'):
        super().__init__(config)
        import os
        from torchvision.datasets import CocoCaptions
        
        root_dir = os.path.join(config.data_dir, "coco")
        
        if split == 'train':
            img_path = os.path.join(root_dir, "train2014")
            ann_path = os.path.join(root_dir, "annotations/captions_train2014.json")
        else:
            # We map 'val' or 'test' requests to the val split
            img_path = os.path.join(root_dir, "val2014")
            ann_path = os.path.join(root_dir, "annotations/captions_val2014.json")
            
        print(f"Loading Dense COCO (Local) from {img_path}...")
        self.coco = CocoCaptions(root=img_path, annFile=ann_path)

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, idx):
        import random
        image, captions = self.coco[idx]
        video = self.prepare_video(image)
        
        # ask for a dense description
        q_tok = self.prepare_text("Provide a highly detailed, dense description of this image:", self.predictor_tokenizer, 64)
        
        # combine all 5 captions into one massive, dense paragraph
        dense_caption = " ".join(captions).strip()
        formatted_caption = f"task: sentence similarity | query: {dense_caption}"
        
        # use max_seq_len (512) to ensure the whole paragraph fits
        t_tok = self.prepare_text(formatted_caption, self.y_encoder_tokenizer, self.config.max_seq_len)
        
        return {
            "video": video,
            "q_ids": q_tok.input_ids.squeeze(0),
            "win_ids": t_tok.input_ids.squeeze(0),
            "win_mask": t_tok.attention_mask.squeeze(0),
            "lose_ids": t_tok.input_ids.squeeze(0),
            "lose_mask": t_tok.attention_mask.squeeze(0)
        }


class AOKVQADataset(BaseJEPADataset):
    def __init__(self, config, split='train', max_samples=2000):
        super().__init__(config)
        print(f"Loading A-OKVQA Dataset ({split})...")
        
        actual_split = 'validation' if split == 'test' else split
        self.dataset = load_dataset("HuggingFaceM4/A-OKVQA", split=actual_split)
        
        if actual_split == 'validation' and len(self.dataset) > max_samples:
            self.dataset = self.dataset.select(range(max_samples))
            
        # if testing, extract candidate answers
        self.is_test = (split in['test', 'val', 'validation'])
        if self.is_test:
            self.unique_answers =[]
            for item in self.dataset:
                ans = item.get('direct_answers', [''])
                ans_str = str(ans[0]).strip() if isinstance(ans, list) and len(ans) > 0 else str(ans).strip()
                if ans_str:
                    self.unique_answers.append(ans_str)
            self.unique_answers = list(set(self.unique_answers))
            print(f"Extracted {len(self.unique_answers)} unique candidate answers for A-OKVQA.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        try:
            image = item['image']
            video = self.prepare_video(image)
        except Exception:
            from PIL import Image
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            video = self.prepare_video(image)
            
        question = item.get('question', 'Describe this image.')
        
        # A-OKVQA has a list of 'direct_answers', pick the first one
        answers = item.get('direct_answers', [''])
        answer = answers[0] if len(answers) > 0 else ""
            
        q_text = f"Question: {question} Answer:"
        q_tok = self.prepare_text(q_text, self.predictor_tokenizer, 64)
        
        if self.is_test:
            # return string for evaluation script to map against candidate embeddings
            return {
                "video": video,
                "q_ids": q_tok.input_ids.squeeze(0),
                "answer_str": str(answer).strip()
            }
        else:
            win_text = f"task: sentence similarity | query: {answer}"
            win_tok = self.prepare_text(win_text, self.y_encoder_tokenizer, self.config.max_seq_len)
            
            return {
                "video": video,
                "q_ids": q_tok.input_ids.squeeze(0),
                "win_ids": win_tok.input_ids.squeeze(0),
                "win_mask": win_tok.attention_mask.squeeze(0),
                "lose_ids": win_tok.input_ids.squeeze(0),
                "lose_mask": win_tok.attention_mask.squeeze(0)
            }

class ChartQADataset(BaseJEPADataset):
    def __init__(self, config, split='train', max_samples=2000):
        super().__init__(config)
        print(f"Loading ChartQA Dataset (Requested split: {split})...")
        
        ds_dict = load_dataset("lmms-lab/ChartQA")
        
        if split not in ds_dict:
            print(f"Split '{split}' not found. Generating dynamic train/test splits...")
            available_split = list(ds_dict.keys())[0]
            
            split_data = ds_dict[available_split].train_test_split(test_size=0.2, seed=42)
            
            if split == 'train':
                self.dataset = split_data['train']
            else:
                self.dataset = split_data['test']
        else:
            self.dataset = ds_dict[split]
            
        if split != 'train' and len(self.dataset) > max_samples:
            self.dataset = self.dataset.select(range(max_samples))
            
        self.is_test = (split in ['test', 'val', 'validation'])
        
        if self.is_test:
            self.unique_answers =[]
            for item in self.dataset:
                ans = item.get('answer', '')
                if not ans and 'conversations' in item:
                    ans = item['conversations'][1]['value'] if len(item['conversations']) > 1 else ''
                if ans:
                    self.unique_answers.append(str(ans).strip())
            self.unique_answers = list(set(self.unique_answers))
            if len(self.unique_answers) == 0:
                raise ValueError("No answers found in the dataset! Check the split.")
            print(f"Extracted {len(self.unique_answers)} unique candidate answers for ChartQA.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        try:
            video = self.prepare_video(item['image'])
        except Exception:
            from PIL import Image
            video = self.prepare_video(Image.new('RGB', (224, 224), (0, 0, 0)))
            
        question = item.get('question', '')
        answer = item.get('answer', '')
        
        # handle different formatting
        if 'conversations' in item and not question:
            question = item['conversations'][0]['value'].replace('<image>\n', '').strip()
            answer = item['conversations'][1]['value'].strip()
            
        q_text = f"Question: {question} Answer:"
        q_tok = self.prepare_text(q_text, self.predictor_tokenizer, 64)
        
        if self.is_test:
            return {
                "video": video,
                "q_ids": q_tok.input_ids.squeeze(0),
                "answer_str": str(answer).strip()
            }
        else:
            win_text = f"task: sentence similarity | query: {answer}"
            win_tok = self.prepare_text(win_text, self.y_encoder_tokenizer, self.config.max_seq_len)
            
            return {
                "video": video,
                "q_ids": q_tok.input_ids.squeeze(0),
                "win_ids": win_tok.input_ids.squeeze(0),
                "win_mask": win_tok.attention_mask.squeeze(0),
                "lose_ids": win_tok.input_ids.squeeze(0),
                "lose_mask": win_tok.attention_mask.squeeze(0)
            }

class DocVQADataset(BaseJEPADataset):
    def __init__(self, config, split='train', max_samples=2000):
        super().__init__(config)
        print(f"Loading DocVQA & InfographicVQA Datasets (Requested split: {split})...")
        
        from datasets import concatenate_datasets
        
        subsets =['DocVQA', 'InfographicVQA']
        dataset_list =[]
        
        for subset in subsets:
            print(f"Fetching subset: {subset}...")
            try:
                ds = load_dataset("lmms-lab/DocVQA", subset, split='validation')
            except Exception as e:
                print(f"  Warning: Could not load {subset}. Error: {e}")
                continue
            
            # Dynamically split 80% / 20% deterministically (seed=42)
            split_data = ds.train_test_split(test_size=0.2, seed=42)
            
            if split == 'train':
                dataset_list.append(split_data['train'])
            else:
                dataset_list.append(split_data['test'])
                
        if not dataset_list:
            raise RuntimeError("Failed to load any DocVQA subsets.")
            
        # Combine both subsets into one massive dataset
        self.dataset = concatenate_datasets(dataset_list)
            
        if split != 'train' and len(self.dataset) > max_samples:
            self.dataset = self.dataset.select(range(max_samples))
            
        self.is_test = (split in ['test', 'val', 'validation'])
        
        if self.is_test:
            self.unique_answers =[]
            for item in self.dataset:
                answers = item.get('answers', [''])
                ans_str = answers[0] if isinstance(answers, list) and len(answers) > 0 else str(answers)
                if ans_str:
                    self.unique_answers.append(str(ans_str).strip())
                    
            self.unique_answers = list(set(self.unique_answers))
            if len(self.unique_answers) == 0:
                raise ValueError("No answers found in the dataset! Check the split.")
            print(f"Extracted {len(self.unique_answers)} unique candidate answers for DocVQA/InfographicVQA.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        try:
            video = self.prepare_video(item['image'])
        except Exception:
            from PIL import Image
            video = self.prepare_video(Image.new('RGB', (224, 224), (0, 0, 0)))
            
        question = item.get('question', 'What does this document say?')
        
        answers = item.get('answers', [''])
        answer = answers[0] if isinstance(answers, list) and len(answers) > 0 else str(answers)
            
        q_text = f"Question: {question} Answer:"
        q_tok = self.prepare_text(q_text, self.predictor_tokenizer, 64)
        
        if self.is_test:
            return {
                "video": video,
                "q_ids": q_tok.input_ids.squeeze(0),
                "answer_str": str(answer).strip()
            }
        else:
            win_text = f"task: sentence similarity | query: {answer}"
            win_tok = self.prepare_text(win_text, self.y_encoder_tokenizer, self.config.max_seq_len)
            
            return {
                "video": video,
                "q_ids": q_tok.input_ids.squeeze(0),
                "win_ids": win_tok.input_ids.squeeze(0),
                "win_mask": win_tok.attention_mask.squeeze(0),
                "lose_ids": win_tok.input_ids.squeeze(0),
                "lose_mask": win_tok.attention_mask.squeeze(0)
            }