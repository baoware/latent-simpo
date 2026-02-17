import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, LlamaModel
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
        
        # initialize architecture structure
        # img_size 256, num_frames 16, patch_size 16
        self.x_encoder = timm.create_model(
            'vit_large_patch16_224', 
            pretrained=True, 
            num_classes=0
        )
        
        # load pre-trained weights locally
        checkpoint_path = "checkpoints/vjepa/vitl16.pth"
        print(f"Loading weights from {checkpoint_path}...")
        print("----------")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        
        # handle different checkpoint formats (encoder vs target_encoder keys)
        if 'encoder' in checkpoint:
            state_dict = checkpoint['encoder']
        elif 'target_encoder' in checkpoint:
            state_dict = checkpoint['target_encoder']
        else:
            state_dict = checkpoint
            
        # remove 'module.' prefix from ddp training if present
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # load weights strict=False to ignore head mismatches
        self.x_encoder.load_state_dict(state_dict, strict=False)
            
        self.x_encoder.eval()
        
        # freeze X-Encoder by loop through all parameters and setting requires_grad to false
        for p in self.x_encoder.parameters(): 
            p.requires_grad = False
            
        # linear projection from vision dimension to predictor dimension
        self.x_proj = nn.Linear(config.x_encoder_dim, config.predictor_dim)

        # reasoning Predictor
        print(f"Initializing Predictor from {config.predictor_source}...")
        print("----------")
        predictor_base = AutoModel.from_pretrained(
            config.predictor_source, 
            trust_remote_code=True,
            dtype=torch.float32 
        )
        
        # embeddings for text query
        self.predictor_embed = predictor_base.embed_tokens
        
        # keep only the last specified transformer layers
        # slice the layer list and wrap it in nn.ModuleList so PyTorch registers them as sub-modules
        self.predictor_layers = nn.ModuleList(predictor_base.layers[-config.num_predictor_layers:])
        # extract the final layer norm from Llama (needed to stabilize outputs)
        self.predictor_norm = predictor_base.norm
        
        # final projection from predictor head to target dimension
        self.predictor_head = nn.Linear(config.predictor_dim, config.target_dim)

        # text targets Y-Encoder
        print(f"Initializing Y-Encoder from {config.y_encoder_source}...")
        print("----------")
        self.y_encoder = AutoModel.from_pretrained(
            config.y_encoder_source,
            dtype=torch.bfloat16,
            trust_remote_code=True
        )
        # projection from y-encoder to target dimension
        self.y_proj = nn.Linear(config.y_encoder_dim, config.target_dim)

    """
        forward pass for Predictor

        inputs: 
            video_pixel_values: Tensor [Batch, Channels, Frames, Height, Width] 
            query_ids: Tensor [Batch, Seq_Len] (token IDs for the prompt)

        outputs:
            pred_emb: Tensor [Batch, Target_Dim] (normalized predicted embedding)
    """
    def forward_predictor(self, video_pixel_values, query_ids):
        # frozen X-Encoder forward
        with torch.no_grad():
            # input is [B, C, T, H, W]
            b, c, t, h, w = video_pixel_values.shape
            
            # permute to [B, T, C, H, W] and flatten to [B*T, C, H, W]
            # this treats every frame as an independent image for the ViT
            images = video_pixel_values.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            
            # forward pass through timm 
            # outputs: [B*T, N_Patches, Dim] or [B*T, Dim]
            features = self.x_encoder.forward_features(images)
            
            # handle CLS token only output
            if len(features.shape) == 2:
                features = features.unsqueeze(1) # [B*T, 1, Dim]
                
            # reshape back to [B, T * N_Patches, Dim]
            # flatten time and patches together into a long sequence
            _, n_tok, dim = features.shape
            x_features = features.view(b, t * n_tok, dim)
        
        # project visual features to match Llama's dimension
        x_embeds = self.x_proj(x_features)
        query_embeds = self.predictor_embed(query_ids)
        
        # combine visual tokens and text tokens along the sequence dimension
        # new shape: [Batch, (N_Patches + Text_Seq_Len), Predictor_Dim]
        hidden_states = torch.cat([x_embeds, query_embeds], dim=1)
        
        # Predictor forward (bidirectional attention)
        batch, seq_len, _ = hidden_states.shape
        # create a mask of 1s (attend to everything) on the correct device
        attention_mask = torch.ones((batch, 1, seq_len, seq_len), device=hidden_states.device)
        
        # iterate through the extracted Llama layers manually
        for layer in self.predictor_layers:
            # pass hidden states through the layer
            # pass the attention_mask to override the default causal behavior
            # [0] selects the first output, hidden states, and ignores attentions weights
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
        
        # apply the final layer norm from Llama
        hidden_states = self.predictor_norm(hidden_states)
        
        # paper: "average pooling on non-[PAD] tokens"
        # use simple mean pooling over the whole sequence
        # collapsing the sequence [Batch, Seq_Len, Dim] to [Batch, Dim]
        pooled = torch.mean(hidden_states, dim=1)
        
        # project the pooled vector to the shared embedding dimension
        pred_emb = self.predictor_head(pooled)

        # L2 normalize the result for cosine similarity loss
        return F.normalize(pred_emb, p=2, dim=-1)

    """
        forward pass for Y-Encoder
        
        inputs:
            input_ids: Tensor [Batch, Seq_Len] (token IDs for the answer/caption)
            attention_mask: Tensor [Batch, Seq_Len] (1 for tokens, 0 for padding)
            
        outputs:
            target_emb: Tensor [Batch, Target_Dim] (normalized target embedding)
    """
    def forward_y_encoder(self, input_ids, attention_mask):
        # pass input through the Y-Encoder
        outputs = self.y_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # get hidden states of last layer
        last_hidden = outputs.last_hidden_state
        
        # expand the attention mask to match the dimensions of the hidden states
        # new shape: [Batch, Seq_Len, Hidden_Dim]
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        # multiply hidden states by mask (zeros out padding vectors) and sum along the sequence dimension
        sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
        # sum the mask to count how many valid tokens there were and clamp min=1e-9 to prevent division by zero errors
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        # divide sum by count to get the true mean
        pooled = sum_embeddings / sum_mask
        
        # project to the shared embedding dimension
        target_emb = self.y_proj(pooled)

        # L2 normalize the result for cosine similarity loss
        return F.normalize(target_emb, p=2, dim=-1)