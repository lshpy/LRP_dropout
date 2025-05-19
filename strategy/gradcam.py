def apply_amplification(features, saliency, unit='pixel', amp_ratio=1.5, top_k=0.15):
    B, C, H, W = features.size()
    if unit == 'pixel':
        flat = saliency.view(B, -1)
        threshold = torch.quantile(flat, 1 - top_k, dim=1).view(B, 1, 1, 1)
        mask = (saliency > threshold).float()
    elif unit == 'channel':
        avg_score = saliency.view(B, C, -1).mean(dim=2)
        threshold = torch.quantile(avg_score, 1 - top_k, dim=1).view(B, 1, 1, 1)
        mask = (avg_score.view(B, C, 1, 1) > threshold).float()
    elif unit == 'patch':
        patch_size = 16
        pooled = torch.nn.functional.adaptive_avg_pool2d(saliency, (H // patch_size, W // patch_size))
        threshold = torch.quantile(pooled.view(B, -1), 1 - top_k, dim=1).view(B, 1, 1)
        mask = (pooled > threshold).float()
        mask = torch.nn.functional.interpolate(mask, size=(H, W), mode='nearest')
    else:
        return features
    return features * (1 + amp_ratio * mask)
