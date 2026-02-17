import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    calculates the InfoNCE (contrastive) loss
    
    goal: make the predicted embedding match the specific target embedding for that image,
    while pushing it away from all other target embeddings in the batch
    
    inputs:
        pred_embeddings:   [Batch_Size, Dim] (from Predictor)
        target_embeddings: [Batch_Size, Dim] (from Y-Encoder)
        temperature:       Scalar scaling factor (controls gradient sharpness)
        
    outputs:
        scalar loss value
"""
def infonce_loss(pred_embeddings, target_embeddings, temperature=0.07):
    
    # normalize vectors to unit length so that dot product equals cosine similarity.
    pred_norm = F.normalize(pred_embeddings, p=2, dim=-1)
    target_norm = F.normalize(target_embeddings, p=2, dim=-1)
    
    # compute similarity matrix (Batch x Batch)
    # multiply every prediction against every target in the batch
    # shape: [Batch_Size, Batch_Size]
    # logits[i][j] = similarity between image_i and text_j
    logits = torch.matmul(pred_norm, target_norm.T) / temperature
    
    # create labels: correct match for image_i is text_i.
    # the "correct" class is the diagonal index (0, 1, 2, ...)
    batch_size = logits.shape[0]
    labels = torch.arange(batch_size).to(logits.device)
    
    # cross entropy: softmax + log-likelihood.
    # tries to maximize the diagonal values (positive pairs) and minimize off-diagonal values (random negatives)
    return F.cross_entropy(logits, labels)
