def apply_recovery_loss(features, mask, original, beta=0.5):
    recon_loss = ((features - original) ** 2 * mask).mean()
    return beta * recon_loss
