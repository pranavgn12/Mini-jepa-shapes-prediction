"""Small CPU smoke tests for shape and gradient routing.

Run with: python test_gradients.py
"""

import torch

from models import JEPAEncoders, JEPAPredictor, CNNDecoder, combine_features
from losses import jepa_loss, full_image_reconstruction_loss


def main():
    torch.manual_seed(7)
    batch_size = 2
    original = torch.rand(batch_size, 3, 64, 64)
    masked = torch.rand(batch_size, 3, 64, 64)
    pixel_mask = torch.zeros(batch_size, 1, 64, 64)
    pixel_mask[:, :, 16:48, 20:44] = 1.0
    latent_mask = torch.nn.functional.adaptive_max_pool2d(
        pixel_mask, (16, 16)
    )

    encoders = JEPAEncoders()
    predictor = JEPAPredictor()
    decoder = CNNDecoder()

    context_features, target_features = encoders(masked, original)
    predicted_tokens = predictor(context_features, latent_mask)
    combined = combine_features(
        context_features, predicted_tokens, latent_mask
    )
    reconstruction = decoder(combined, latent_mask)

    assert context_features.shape == (batch_size, 64, 16, 16)
    assert predicted_tokens.shape == (batch_size, 256, 64)
    assert combined.shape == (batch_size, 64, 16, 16)
    assert reconstruction.shape == (batch_size, 3, 64, 64)
    assert reconstruction.min().item() >= 0.0
    assert reconstruction.max().item() <= 1.0

    # Reconstruction-only backward must reach the online representation and
    # decoder, but never the frozen target encoder.
    recon_loss = full_image_reconstruction_loss(reconstruction, original)
    recon_loss.backward()

    context_grad = sum(
        p.grad.abs().sum().item()
        for p in encoders.context_encoder.parameters()
        if p.grad is not None
    )
    predictor_grad = sum(
        p.grad.abs().sum().item()
        for p in predictor.parameters()
        if p.grad is not None
    )
    decoder_grad = sum(
        p.grad.abs().sum().item()
        for p in decoder.parameters()
        if p.grad is not None
    )
    target_has_grad = any(
        p.grad is not None for p in encoders.target_encoder.parameters()
    )

    assert context_grad > 0, "Reconstruction did not update context encoder"
    assert predictor_grad > 0, "Reconstruction did not update predictor"
    assert decoder_grad > 0, "Reconstruction did not update decoder"
    assert not target_has_grad, "Frozen target encoder unexpectedly got gradients"

    # Run a fresh forward graph for the separate JEPA-only backward test.
    # (A completed backward pass frees its saved intermediate tensors.)
    encoders.zero_grad(set_to_none=True)
    predictor.zero_grad(set_to_none=True)
    decoder.zero_grad(set_to_none=True)
    context_features_2, target_features_2 = encoders(masked, original)
    predicted_tokens_2 = predictor(context_features_2, latent_mask)
    jepa = jepa_loss(predicted_tokens_2, target_features_2, latent_mask)
    jepa.backward()

    assert any(p.grad is not None for p in predictor.parameters())
    assert any(p.grad is not None for p in encoders.context_encoder.parameters())
    assert not any(p.grad is not None for p in encoders.target_encoder.parameters())

    print("PASS: tensor shapes")
    print("PASS: nearest-expansion decoder output range")
    print("PASS: reconstruction loss -> context encoder + predictor + decoder")
    print("PASS: target encoder remains gradient-free")
    print("PASS: JEPA loss -> context encoder + predictor")


if __name__ == "__main__":
    main()
