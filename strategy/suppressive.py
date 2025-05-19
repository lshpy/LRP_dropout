import torch
import torch.nn.functional as F

def apply_suppressive_dropout(features, relevance, unit='pixel', drop_ratio=0.15):
    B, C, H, W = features.shape
    mask = torch.ones_like(features)

    if unit == 'pixel':
        scores = relevance.view(B, -1)
        threshold = torch.quantile(scores, 1 - drop_ratio, dim=1, keepdim=True)
        drop = (relevance.view(B, -1) >= threshold).view(B, C, H, W)
        mask[drop] = 0

    elif unit == 'patch':
        patch_size = 16
        patches = relevance.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patch_scores = patches.mean(dim=(-1, -2))  # (B, C, H//p, W//p)
        threshold = torch.quantile(patch_scores.view(B, -1), 1 - drop_ratio, dim=1, keepdim=True)
        drop = (patch_scores.view(B, -1) >= threshold).view_as(patch_scores)
        drop = drop.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, patch_size, patch_size)
        patch_mask = drop.reshape(B, C, H, W)
        mask[patch_mask.bool()] = 0

    elif unit == 'channel':
        channel_scores = relevance.mean(dim=(2, 3))  # (B, C)
        threshold = torch.quantile(channel_scores, 1 - drop_ratio, dim=1, keepdim=True)
        drop = (channel_scores >= threshold).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        mask[drop] = 0

    return features * mask
