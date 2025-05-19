def apply_hybrid_mask(features, relevance, saliency, unit='pixel', mode='drop', top_k=0.15):
    if mode == 'drop':
        hybrid_score = relevance - saliency  # 억제는 높고 주목은 낮은
        threshold = torch.quantile(hybrid_score.view(hybrid_score.size(0), -1), 1 - top_k, dim=1).view(-1, 1, 1, 1)
        mask = (hybrid_score < threshold).float()
        return features * mask
    elif mode == 'amp':
        hybrid_score = relevance + saliency  # 둘 다 높은 부분 강조
        threshold = torch.quantile(hybrid_score.view(hybrid_score.size(0), -1), 1 - top_k, dim=1).view(-1, 1, 1, 1)
        mask = (hybrid_score > threshold).float()
        return features * (1 + 1.5 * mask)
    else:
        return features
