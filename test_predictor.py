
import torch
import torch.nn.functional as F

from dataset import ShapesDataset
from models import JEPAEncoders, JEPAPredictor

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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

# Convert 64x64 pixel mask into 8x8 feature mask.
target_mask = F.adaptive_max_pool2d(
    pixel_mask, (8, 8)
)

encoders = JEPAEncoders().to(device)
predictor = JEPAPredictor().to(device)

context_features, target_features = encoders(
    masked, original
)

predicted_targets = predictor(
    context_features, target_mask
)

# Extract corresponding target embeddings.

predicted_tokens = predictor(
    context_features, target_mask
)

# Convert target encoder output to tokens.
target_tokens = target_features.flatten(2).transpose(1, 2)

# Convert mask into (B,64).
target_mask_flat = target_mask.flatten(1).bool()

# Select corresponding masked positions.
predicted_targets = predicted_tokens[target_mask_flat]
target_targets = target_tokens[target_mask_flat]

print("Context:", context_features.shape)
print("Target:", target_features.shape)
print("Mask:", target_mask.shape)
print("All predicted tokens:", predicted_tokens.shape)
print("Selected predictions:", predicted_targets.shape)
print("Selected targets:", target_targets.shape)

assert predicted_targets.shape == target_targets.shape

print("Predictor test successful!")

