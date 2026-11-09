
import torch
import torch.nn.functional as F

from dataset import ShapesDataset
from models import (
    JEPAEncoders,
    JEPAPredictor,
    CNNDecoder,
    combine_features
)
from losses import (
    jepa_loss,
    reconstruction_loss,
    combined_loss
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# Generate batch.
dataset = ShapesDataset(length=16)
batch = [dataset[i] for i in range(16)]

masked = torch.stack(
    [x["masked"] for x in batch]
).to(device)

original = torch.stack(
    [x["original"] for x in batch]
).to(device)

pixel_mask = torch.stack(
    [x["mask"] for x in batch]
).to(device)

target_mask = F.adaptive_max_pool2d(
    pixel_mask, (8, 8)
)

# Initialize models.
encoders = JEPAEncoders().to(device)
predictor = JEPAPredictor().to(device)
decoder = CNNDecoder().to(device)

# Forward pass.
context_features, target_features 

predicted_tokens = predictor(
    context_features, target_mask
)

combined_features = combine_features(
    context_features,
    predicted_tokens,
    target_mask
)

reconstructed = decoder(
    combined_features,
    skip16,
    skip32,
    pixel_mask
)

# Calculate losses.
loss_jepa = jepa_loss(
    predicted_tokens,
    target_features,
    target_mask
)

loss_reconstruction = reconstruction_loss(
    reconstructed,
    original,
    pixel_mask
)

loss_total = combined_loss(
    loss_jepa,
    loss_reconstruction,
    lambda_reconstruction=1.0
)

print("\nLosses:")
print("JEPA loss:", loss_jepa.item())
print("Reconstruction loss:", loss_reconstruction.item())
print("Total loss:", loss_total.item())

# Backpropagation.
loss_total.backward()

print("\nGradient checks:")

print(
    "Context encoder:",
    next(encoders.context_encoder.parameters()).grad is not None
)

print(
    "Predictor:",
    next(predictor.parameters()).grad is not None
)

print(
    "Decoder:",
    next(decoder.parameters()).grad is not None
)

print(
    "Target encoder:",
    next(encoders.target_encoder.parameters()).grad
)

# Test EMA update.
encoders.update_target_encoder()

print("\nEMA update successful!")

print("\nLoss test successful!")
