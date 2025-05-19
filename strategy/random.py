import torch
import torch.nn.functional as F

def apply_random_dropout(features, unit='pixel', drop_rate=0.15):
    B, C, H, W = features.size()
    if unit == 'pixel':
        mask = torch.rand((B, 1, H, W), device=features.device) > drop_rate
    elif unit == 'channel':
        mask = torch.rand((B, C, 1, 1), device=features.device) > drop_rate
    elif unit == 'patch':
        patch_size = 16
        ph, pw = H // patch_size, W // patch_size
        patch_mask = torch.rand((B, 1, ph, pw), device=features.device) > drop_rate
        mask = F.interpolate(patch_mask.float(), size=(H, W), mode='nearest') > 0.5
    else:
        return features
    return features * mask
