
import torch
import torch.nn.functional as F

from dataset import ShapesDataset
from models import (
    JEPAEncoders,
    JEPAPredictor,
    CNNDecoder,
    combine_features
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# Generate a batch.
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

# Create the feature-grid mask.
target_mask = F.adaptive_max_pool2d(
    pixel_mask, (8, 8)
)

# Initialize models.
encoders = JEPAEncoders().to(device)
predictor = JEPAPredictor().to(device)
decoder = CNNDecoder().to(device)

# Forward pass.
(
    context_features,
    skip16,
    skip32,
    target_features
) = encoders(masked, original)

predicted_tokens = predictor(
    context_features, target_mask
)

# Combine visible context and predicted masked features.
combined_features = combine_features(
    context_features,
    predicted_tokens,
    target_mask
)

# Decode into RGB image.
reconstructed = decoder(
    combined_features,
    skip16,
    skip32,
    pixel_mask
)

print("\nTensor dimensions:")
print("Context:", context_features.shape)
print("Predicted:", predicted_tokens.shape)
print("Combined:", combined_features.shape)
print("Reconstructed:", reconstructed.shape)

assert reconstructed.shape == original.shape

# Reconstruction loss only on masked pixels.
loss = (
    (reconstructed - original).abs() * pixel_mask
).sum() / (pixel_mask.sum() * 3).clamp(min=1)

print("\nReconstruction loss:", loss.item())

# Test backpropagation.
loss.backward()

print(
    "Context encoder gradient:",
    next(
        encoders.context_encoder.parameters()
    ).grad is not None
)

print(
    "Predictor gradient:",
    next(predictor.parameters()).grad is not None
)

print(
    "Decoder gradient:",
    next(decoder.parameters()).grad is not None
)

print(
    "Target encoder gradient:",
    next(encoders.target_encoder.parameters()).grad
)

print("\nDecoder test successful!")
