import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import transformers
transformers.logging.set_verbosity_error()
import timm
import sys
import os

class VL_JEPA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # vision X-Encoder
        print(f"Initializing X-Encoder...")
        print("----------")
        self.x_encoder = timm.create_model(
            'vit_large_patch16_224', 
            pretrained=True, 
            num_classes=0
        )
            
        self.x_encoder.eval()
        for p in self.x_encoder.parameters(): 
            p.requires_grad = False
            
        self.x_proj = nn.Linear(config.x_encoder_dim, config.predictor_dim)

        # reasoning Predictor
        print(f"Initializing Predictor from {config.predictor_source}...")
        print("----------")
        
        # load full model logic
        self.predictor_model = AutoModel.from_pretrained(
            config.predictor_source, 
            trust_remote_code=True,
            dtype=torch.float32 
        )
        
        # freeze model
        self.predictor_model.eval()
        for param in self.predictor_model.parameters():
            param.requires_grad = False
            
        # unfreeze only the last n layers
        # standard HF structure: model.layers (Llama/Mistral/Qwen/Phi)
        # need to find where the layers list is
        if hasattr(self.predictor_model, "layers"):
            layers = self.predictor_model.layers
        elif hasattr(self.predictor_model, "model"):
            layers = self.predictor_model.model.layers
        else:
            raise AttributeError("Could not find layers in Predictor")

        # enable gradients for the last n layers
        for i in range(len(layers) - config.num_predictor_layers, len(layers)):
            print(f"Unfreezing Predictor Layer {i}")
            for param in layers[i].parameters():
                param.requires_grad = True
        
        # unfreeze the final norm
        if hasattr(self.predictor_model, "norm"):
            for p in self.predictor_model.norm.parameters(): p.requires_grad = True
        elif hasattr(self.predictor_model, "model") and hasattr(self.predictor_model.model, "norm"):
             for p in self.predictor_model.model.norm.parameters(): p.requires_grad = True

        
        # need access to the embedding layer for the forward pass
        self.predictor_embed = self.predictor_model.get_input_embeddings()

        # final projection
        self.predictor_head = nn.Linear(config.predictor_dim, config.target_dim)

        # text targets Y-Encoder
        print(f"Initializing Y-Encoder from {config.y_encoder_source}...")
        print("----------")
        self.y_encoder = AutoModel.from_pretrained(
            config.y_encoder_source,
            dtype=torch.bfloat16,
            trust_remote_code=True
        )
        y_dim = getattr(self.y_encoder.config, "hidden_size", 0)
        if y_dim == 0: y_dim = getattr(self.y_encoder.config, "d_model", 768)
        self.y_proj = nn.Linear(y_dim, config.target_dim)

    def forward_predictor(self, video_pixel_values, query_ids):
        # X-Encoder
        with torch.no_grad():
            b, c, t, h, w = video_pixel_values.shape
            images = video_pixel_values.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            features = self.x_encoder.forward_features(images)
            if len(features.shape) == 2: features = features.unsqueeze(1)
            _, n_tok, dim = features.shape
            x_features = features.view(b, t * n_tok, dim)
        
        # embed
        x_embeds = self.x_proj(x_features)
        query_embeds = self.predictor_embed(query_ids)
        
        # concatenate
        # [Batch, Seq_Len, Dim]
        inputs_embeds = torch.cat([x_embeds, query_embeds], dim=1)
        
        # forward Pass using full model wrapper
        # handles RoPE, caching, and masks automatically
        # pass 'inputs_embeds' directly
        outputs = self.predictor_model(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True
        )
        
        # get last hidden state
        hidden_states = outputs.last_hidden_state
        
        # pooling and projection
        # simple mean pool over the sequence
        pooled = torch.mean(hidden_states, dim=1)
        pred_emb = self.predictor_head(pooled)

        return F.normalize(pred_emb, p=2, dim=-1)

    def forward_y_encoder(self, input_ids, attention_mask):
        # same as before
        outputs = self.y_encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'last_hidden_state'):
            last_hidden = outputs.last_hidden_state
        else:
            last_hidden = outputs[0]
        
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        target_emb = self.y_proj(pooled.to(dtype=torch.float32))
        return F.normalize(target_emb, p=2, dim=-1)