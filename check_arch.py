import torch
import time
import os

from src.config import Config
from src.model import VL_JEPA
from src.losses import infonce_loss

def check_arch():
    print("VL-JEPA architecture test...")
    print("----------")

    # load config
    cfg = Config()
    device = cfg.device
    print(f"Using device: {device}")
    
    # override batch size for a quick local test
    b = 2 
    t = cfg.num_frames
    h = w = cfg.resolution
    q_len = 16
    t_len = cfg.max_seq_len

    # initialize model
    print("Initializing model...")
    start_time = time.time()
    try:
        model = VL_JEPA(cfg).to(device)
        print(f"Model initialized in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Model initialization failed: {e}")
        return

    # check parameter states (frozen vs trainable)
    print("Verifying parameter states...")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1e6:.2f} M")
    print(f"Frozen parameters: {frozen_params / 1e6:.2f} M")
    
    # generate dummy data
    print("Generating dummy data...")
    # video shape:[batch, channels, time, height, width]
    dummy_video = torch.randn(b, t, 3, h, w).to(device)
    # token shapes: [batch, seq_len]
    dummy_q_ids = torch.randint(0, 1000, (b, q_len)).to(device)
    dummy_t_ids = torch.randint(0, 1000, (b, t_len)).to(device)
    dummy_t_mask = torch.ones(b, t_len).to(device)
    
    print(f"Video tensor: {list(dummy_video.shape)}")
    print(f"Query tensor: {list(dummy_q_ids.shape)}")
    
    # execute forward and backward pass
    print("Executing forward and backward pass...")
    model.train() 
    
    try:
        # use autocast to mimic the actual training loop
        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
            
            # forward predictor
            pred_emb = model.forward_predictor(dummy_video, dummy_q_ids)
            print(f"Predictor forward success; output shape: {list(pred_emb.shape)}")
            
            # forward y-encoder
            target_emb = model.forward_y_encoder(dummy_t_ids, dummy_t_mask)
            print(f"Y-encoder forward success; output shape: {list(target_emb.shape)}")
            
            # dimensionality check
            assert pred_emb.shape == target_emb.shape, f"Shape mismatch! pred: {pred_emb.shape}, target: {target_emb.shape}"
            assert pred_emb.shape[-1] == cfg.target_dim, f"output dim {pred_emb.shape[-1]} != target_dim {cfg.target_dim}"
            
            # loss calculation
            loss = infonce_loss(pred_emb, target_emb, temperature=cfg.temperature)
            print(f"Loss calculation success; dummy loss: {loss.item():.4f}")
            
        # backward pass
        loss.backward()
        print("Backward pass success")
        
        # verify gradients flowed to the right places
        has_pred_grad = any(p.grad is not None for p in model.predictor_head.parameters())
        has_y_grad = any(p.grad is not None for p in model.y_proj.parameters())
        has_x_grad = any(p.grad is not None for p in model.x_encoder.parameters())
    
        print("Gradient check")
        print(f"Predictor receiving gradients?: {'yes' if has_pred_grad else 'no'}")
        print(f"Y-encoder receiving gradients?: {'yes' if has_y_grad else 'no'}")
        print(f"X-encoder remains frozen?     : {'yes' if not has_x_grad else 'no'}")

    except Exception as e:
        print("Execution failed")
        import traceback
        traceback.print_exc()
        return

    print("----------")
    print("Architecture good...")
    print("----------")

if __name__ == "__main__":
    check_arch()