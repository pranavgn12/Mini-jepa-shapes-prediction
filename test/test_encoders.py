
import torch

from models import JEPAEncoders
from dataset import ShapesDataset

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

model = JEPAEncoders().to(device)

dataset = ShapesDataset(length=16)

masked = torch.stack([
    dataset[i]["masked"] for i in range(16)
]).to(device)

original = torch.stack([
    dataset[i]["original"] for i in range(16)
]).to(device)

context_features, target_features = model(
    masked,
    original
)

print("Masked input:", masked.shape)
print("Original input:", original.shape)

print("Context features:", context_features.shape)
print("Target features:", target_features.shape)

print(
    "Context requires grad:",
    context_features.requires_grad
)

print(
    "Target requires grad:",
    target_features.requires_grad
)
