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


"""
    calculates the latent simple preference optimization
    
    goal: force predicted embedding to be closer to the 'winner' (safe/factual)
    than the 'loser' (unsafe/hallucinated) by a specific margin
    
    inputs:
        pred_embeddings:    [Batch_Size, Dim] (the model's actual prediction)
        win_embeddings:     [Batch_Size, Dim] (target embedding for the 'chosen' text)
        lose_embeddings:    [Batch_Size, Dim] (target embedding for the 'rejected' text)
        beta:     scalar reward scale (temperature inverse)
        gamma:    scalar target margin (the 'safety buffer')
    
    outputs:
        scalar loss value
"""
def latent_simpo_loss(pred_emb, win_emb, lose_emb, beta=10.0, gamma=0.2):
    
    # normalize everything
    pred_norm = F.normalize(pred_emb, p=2, dim=-1)
    win_norm = F.normalize(win_emb, p=2, dim=-1)
    lose_norm = F.normalize(lose_emb, p=2, dim=-1)

    # calculate cosine similarities
    sim_win = torch.sum(pred_norm * win_norm, dim=-1)
    sim_lose = torch.sum(pred_norm * lose_norm, dim=-1)

    # compute margin: (win - lose) - gamma
    margin = sim_win - sim_lose - gamma
    
    # log-sigmoid loss
    loss = -F.logsigmoid(beta * margin)
    
    return loss.mean()

"""
    calculates the latent triplet margin loss (the metric learning baseline)
    
    goal: push the predicted embedding closer to the 'winner' than the 'loser' 
    by a fixed margin using a hard hinge loss, stopping gradients once the margin is met
    
    inputs:
        pred_emb:         [Batch_Size, Dim] (the model's actual prediction)
        win_emb:          [Batch_Size, Dim] (target embedding for the 'chosen' text)
        lose_emb:         [Batch_Size, Dim] (target embedding for the 'rejected' text)
        margin:           scalar target margin
    
    outputs:
        scalar loss value
"""
def triplet_margin_loss(pred_emb, win_emb, lose_emb, margin=0.2):
    
    # normalize everything
    pred_norm = F.normalize(pred_emb, p=2, dim=-1)
    win_norm = F.normalize(win_emb, p=2, dim=-1)
    lose_norm = F.normalize(lose_emb, p=2, dim=-1)

    # calculate cosine similarities
    sim_win = torch.sum(pred_norm * win_norm, dim=-1)
    sim_lose = torch.sum(pred_norm * lose_norm, dim=-1)

    # triplet hinge loss: max(0, sim_lose - sim_win + margin)
    # if win is greater than lose by margin, loss becomes 0
    loss = F.relu(sim_lose - sim_win + margin)
    
    return loss.mean()


def anchor_loss(pred_emb, ref_emb):
    pred_norm = torch.nn.functional.normalize(pred_emb, p=2, dim=-1)
    ref_norm = torch.nn.functional.normalize(ref_emb, p=2, dim=-1)
    return (1.0 - torch.sum(pred_norm * ref_norm, dim=-1)).mean()