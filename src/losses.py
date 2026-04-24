import torch
import torch.nn as nn
import torch.nn.functional as F

def _apply_reduction(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")

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
def infonce_loss(pred_embeddings, target_embeddings, temperature=0.07, reduction="mean"):
    
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
    loss = F.cross_entropy(logits, labels, reduction="none")
    return _apply_reduction(loss, reduction)


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

    variant="paper":
        softplus(gamma - beta * (sim_win - sim_lose))
        == -logsigmoid(beta * (sim_win - sim_lose) - gamma)

    variant="legacy":
        softplus(-beta * ((sim_win - sim_lose) - gamma))
        == -logsigmoid(beta * ((sim_win - sim_lose) - gamma))
"""
def latent_simpo_loss(pred_emb, win_emb, lose_emb, beta=10.0, gamma=2.0, variant="paper", reduction="none"):

    # normalize everything
    pred_norm = F.normalize(pred_emb, p=2, dim=-1)
    win_norm = F.normalize(win_emb, p=2, dim=-1)
    lose_norm = F.normalize(lose_emb, p=2, dim=-1)

    # calculate cosine similarities
    sim_win = torch.sum(pred_norm * win_norm, dim=-1)
    sim_lose = torch.sum(pred_norm * lose_norm, dim=-1)
    delta = sim_win - sim_lose
    
    # log-sigmoid loss
    if variant == "paper":
        loss = F.softplus(gamma - beta * delta)
    elif variant == "legacy":
        loss = F.softplus(-beta * (delta - gamma))
    else:
        raise ValueError(f"Unknown SimPO variant: {variant}")

    return _apply_reduction(loss, reduction)

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
def triplet_margin_loss(pred_emb, win_emb, lose_emb, margin=0.2, reduction="none"):
    
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
    return _apply_reduction(loss, reduction)


def anchor_loss(pred_emb, ref_emb, reduction="none"):
    pred_norm = F.normalize(pred_emb.float(), p=2, dim=-1)
    ref_norm = F.normalize(ref_emb.float(), p=2, dim=-1)
    loss = 1.0 - torch.sum(pred_norm * ref_norm, dim=-1)
    return _apply_reduction(loss, reduction)

"""
    calculates the latent contrastive preference optimization (CPO) loss
    
    goal: push the predicted embedding closer to the 'winner' than the 'loser'
    using a relative preference loss, while simultaneously applying a behavioral 
    cloning (BC) penalty to keep the prediction anchored explicitly to the 'winner'
    
    inputs:
        pred_emb:  [Batch_Size, Dim] (the model's actual prediction)
        win_emb:   [Batch_Size, Dim] (target embedding for the 'chosen' text)
        lose_emb:  [Batch_Size, Dim] (target embedding for the 'rejected' text)
        beta:      scalar reward scale (temperature inverse) for the preference term
        bc_weight: scalar weight for the behavioral cloning (anchoring) term
    
    outputs:
        scalar loss value (or batched tensor if reduction="none")
"""
def latent_cpo_loss(pred_emb, win_emb, lose_emb, beta=10.0, bc_weight=1.0, reduction="none"):
    pred_norm = F.normalize(pred_emb.float(), p=2, dim=-1)
    win_norm = F.normalize(win_emb.float(), p=2, dim=-1)
    lose_norm = F.normalize(lose_emb.float(), p=2, dim=-1)

    sim_win = torch.sum(pred_norm * win_norm, dim=-1)
    sim_lose = torch.sum(pred_norm * lose_norm, dim=-1)

    pref_loss = F.softplus(-beta * (sim_win - sim_lose))
    bc_loss = 1.0 - sim_win

    loss = pref_loss + bc_weight * bc_loss
    return _apply_reduction(loss, reduction)

"""
    calculates the preference InfoNCE loss
    
    goal: pull the prediction towards the 'winner' while pushing it away from the 
    'loser'. Can optionally use all other winners in the batch as additional 
    negative samples for a stronger contrastive signal
    
    inputs:
        pred_emb:               [Batch_Size, Dim] (the model's actual prediction)
        win_emb:                [Batch_Size, Dim] (target embedding for the 'chosen' text)
        lose_emb:               [Batch_Size, Dim] (target embedding for the 'rejected' text)
        temperature:            scalar scaling factor (controls gradient sharpness)
        use_in_batch_negatives: boolean, if True uses other batch winners as negatives
    
    outputs:
        scalar loss value (or batched tensor if reduction="none")
"""
def preference_infonce_loss(
    pred_emb,
    win_emb,
    lose_emb,
    temperature=0.07,
    use_in_batch_negatives=True,
    reduction="none",
):
    pred_norm = F.normalize(pred_emb.float(), p=2, dim=-1)
    win_norm = F.normalize(win_emb.float(), p=2, dim=-1)
    lose_norm = F.normalize(lose_emb.float(), p=2, dim=-1)

    batch_size = pred_emb.size(0)

    if use_in_batch_negatives:
        sim_win = torch.matmul(pred_norm, win_norm.T) / temperature
        sim_lose = torch.sum(pred_norm * lose_norm, dim=-1, keepdim=True) / temperature
        logits = torch.cat([sim_win, sim_lose], dim=1)
        labels = torch.arange(batch_size, device=pred_emb.device)
    else:
        sim_win = torch.sum(pred_norm * win_norm, dim=-1, keepdim=True) / temperature
        sim_lose = torch.sum(pred_norm * lose_norm, dim=-1, keepdim=True) / temperature
        logits = torch.cat([sim_win, sim_lose], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=pred_emb.device)

    loss = F.cross_entropy(logits, labels, reduction="none")
    return _apply_reduction(loss, reduction)