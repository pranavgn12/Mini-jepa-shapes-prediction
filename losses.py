
import torch
import torch.nn.functional as F


def jepa_loss(
    predicted_tokens,
    target_features,
    target_mask
):
    """
    Compare predicted and target embeddings
    only at masked spatial positions.

    predicted_tokens: (B,64,128)
    target_features:  (B,128,8,8)
    target_mask:      (B,1,8,8)
    """

    # Convert target features into spatial tokens.
    target_tokens = (
        target_features.flatten(2).transpose(1, 2)
    )

    # Convert mask to (B,64).
    mask = target_mask.flatten(1).bool()

    # Select masked positions from both tensors.
    predicted_targets = predicted_tokens[mask]
    actual_targets = target_tokens[mask]

    # Embedding-space prediction loss.
    loss = F.smooth_l1_loss(
        predicted_targets,
        actual_targets
    )

    return loss


def reconstruction_loss(
    reconstructed,
    original,
    pixel_mask
):
    """
    Calculate pixel reconstruction loss
    only over masked regions.

    reconstructed: (B,3,64,64)
    original:      (B,3,64,64)
    pixel_mask:    (B,1,64,64)
    """

    # Absolute pixel differences.
    difference = torch.abs(reconstructed - original)

    # Ignore visible pixels.
    masked_difference = difference * pixel_mask

    # Normalize by the number of masked RGB values.
    denominator = (
        pixel_mask.sum() * reconstructed.shape[1]
    ).clamp(min=1)

    return masked_difference.sum() / denominator


def combined_loss(
    loss_jepa,
    loss_reconstruction,
    lambda_reconstruction=1.0
):

    total = (
        loss_jepa
        + lambda_reconstruction * loss_reconstruction
    )

    return total
