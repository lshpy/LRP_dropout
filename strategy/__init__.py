from .random import apply_random_dropout
from .suppressive import apply_suppressive_dropout
from .gradcam import apply_amplification
from .hybrid import apply_hybrid_mask
from .mixed import apply_mixed
from .recovery import apply_recovery_loss

def apply_strategy(features, R, A, strategy_type, unit):
    if strategy_type == 'random':
        return apply_random_dropout(features, unit)
    elif strategy_type == 'suppressive':
        return apply_suppressive_dropout(features, R, unit)
    elif strategy_type == 'gradcam_amp':
        return apply_amplification(features, A, unit)
    elif strategy_type == 'hybrid_drop':
        return apply_hybrid_mask(features, R, A, unit, mode='drop')
    elif strategy_type == 'hybrid_amp':
        return apply_hybrid_mask(features, R, A, unit, mode='amp')
    elif strategy_type == 'mixed':
        return apply_mixed(features, R, A, unit)
    elif strategy_type == 'recovery':
        return apply_recovery_loss(features, R, unit)
    else:
        return features  # No dropout
