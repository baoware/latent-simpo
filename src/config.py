from dataclasses import dataclass
import torch

@dataclass
class Config:
    # paths
    output_dir: str = "./checkpoints"
    data_dir: str = "./data"
    
    # vl-jepa architecture components from paper
    # X-Encoder: V-JEPA 2 ViT-L
    # x_encoder_source: str = "vit_large" 
    x_encoder_source: str = "facebook/vjepa2-vitl-fpc64-256" 
    
    # Predictor: Llama-3.2-1B (once accessible)
    predictor_source: str = "meta-llama/Llama-3.2-1B"
    # predictor_source: str = "microsoft/Phi-4-mini-instruct"
    
    # Y-Encoder: EmbeddingGemma-300M
    y_encoder_source = "google/embeddinggemma-300m"
    
    # architecture dimensions from paper
    x_encoder_dim: int = 1024   # ViT-L output dimension
    predictor_dim: int = 2048   # Llama-3.2-1B or Phi-4-mini hidden dimension
    y_encoder_dim: int = 768    # EmbeddingGemma-300M
    target_dim: int = 1536      # shared embedding space
    
    # predictor initialization
    num_predictor_layers: int = 8  # last 8 transformer layers
    
    # datasets
    # options: 'coco', 'datacomp'
    # dataset_name: str = 'coco' 
    dataset_name: str = 'datacomp' 

    # input specifications
    num_frames: int = 2
    resolution: int = 256      
    max_seq_len: int = 64       # short for pre-training but normally 512 query tokens
    
    # baseline training
    batch_size: int = 1024
    batch_size_base: int = 512
    epochs_base: int = 1
    
    # learning rates 
    # vl-jepa paper: "setting a learning rate multiplier of ×0.05 to all text encoder parameters improves performance"
    lr_predictor: float = 1e-4
    lr_y_encoder: float = 5e-6  # 1e-4 * 0.05
    
    temperature: float = 0.07   # InfoNCE temperature
    
    # latent-simpo training
    batch_size_simpo: int = 16
    epochs_simpo: int = 3
    lr_simpo: float = 1e-5
    
    # simpo hyperparams
    beta: float = 10.0          # reward scale
    gamma: float = 0.2          # target margin 
    lambda_reg: float = 0.1     # stability regularization 
    
    # system
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 10 # encara messi