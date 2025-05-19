from .suppressive import apply_suppressive_dropout
from .gradcam import apply_amplification

def apply_mixed(features, relevance, saliency, unit='pixel'):
    dropped = apply_suppressive_dropout(features, relevance, unit)
    amplified = apply_amplification(dropped, saliency, unit)
    return amplified
