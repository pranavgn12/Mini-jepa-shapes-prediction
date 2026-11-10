import torch
import torch.nn.functional as F


def jepa_loss(predicted_tokens, target_features, target_mask):
    """
    JEPA MSE on masked latent positions only.
    predicted_tokens: (B, N, C)
    target_features:  (B, C, H, W) or (B, N, C)
    target_mask:      (B, 1, H, W) or (B, H, W)
    """
    if target_features.ndim == 4:
        target_tokens = target_features.flatten(2).transpose(1, 2)
    elif target_features.ndim == 3:
        target_tokens = target_features
    else:
        raise ValueError("target_features must be (B,C,H,W) or (B,N,C)")

    if predicted_tokens.shape != target_tokens.shape:
        raise ValueError(
            f"Predictor/target mismatch: {predicted_tokens.shape} "
            f"vs {target_tokens.shape}"
        )

    if target_mask.ndim == 4:
        mask = target_mask.flatten(1)
    elif target_mask.ndim == 3:
        mask = target_mask.flatten(1)
    else:
        raise ValueError("target_mask must be (B,1,H,W) or (B,H,W)")

    errors = (predicted_tokens - target_tokens.detach()).square().mean(dim=-1)
    mask = mask.to(errors.dtype)
    counts = mask.sum(dim=1)

    per_sample = (errors * mask).sum(dim=1) / counts.clamp_min(1.0)
    # Defensive fallback if a sample happens to contain no masked latent cells.
    per_sample = torch.where(counts > 0, per_sample, errors.mean(dim=1))
    return per_sample.mean()


def full_image_reconstruction_loss(predicted_image, original_image):
    """
    Full-image L1: every RGB pixel contributes, not just masked pixels.
    The original image is a target only; it is never a decoder input.
    """
    if predicted_image.shape != original_image.shape:
        raise ValueError(
            f"Image shape mismatch: {predicted_image.shape} "
            f"vs {original_image.shape}"
        )
    return F.l1_loss(predicted_image, original_image)


def compute_losses(
    predicted_tokens,
    target_features,
    latent_mask,
    reconstructed_image,
    original_image,
    lambda_jepa=1.0,
    lambda_reconstruction=1.0,
):
    jepa = jepa_loss(predicted_tokens, target_features, latent_mask)
    reconstruction = full_image_reconstruction_loss(
        reconstructed_image, original_image
    )
    total = lambda_jepa * jepa + lambda_reconstruction * reconstruction
    return {
        "jepa": jepa,
        "reconstruction": reconstruction,
        "total": total,
    }
