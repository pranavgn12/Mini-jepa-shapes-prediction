import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """64x64 RGB image -> 64-channel 16x16 feature grid."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),  # 32 -> 16
        )

    def forward(self, image):
        return self.network(image)


class JEPAEncoders(nn.Module):
    """Online context encoder and frozen EMA target encoder."""

    def __init__(self, ema_decay=0.996):
        super().__init__()
        self.context_encoder = CNNEncoder()
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.ema_decay = ema_decay
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        # The target encoder must remain in eval mode (notably for BatchNorm).
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target_encoder(self):
        for online_param, target_param in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_param.mul_(self.ema_decay)
            target_param.add_(online_param, alpha=1.0 - self.ema_decay)

        # Keep BatchNorm running statistics synchronized with the online encoder.
        for online_buffer, target_buffer in zip(
            self.context_encoder.buffers(), self.target_encoder.buffers()
        ):
            target_buffer.copy_(online_buffer)

    def forward(self, masked_image, original_image):
        context_features = self.context_encoder(masked_image)
        with torch.no_grad():
            target_features = self.target_encoder(original_image)
        return context_features, target_features


class JEPAPredictor(nn.Module):
    """Transformer predictor operating on a 16x16 grid of image tokens."""

    def __init__(
        self,
        embed_dim=64,
        grid_size=16,
        num_heads=4,
        num_layers=2,
        mlp_dim=256,
        dropout=0.0,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_tokens, embed_dim)
        )
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, context_features, target_mask):
        batch, channels, height, width = context_features.shape
        if channels != 64 or (height, width) != (self.grid_size, self.grid_size):
            raise ValueError(
                "Expected context features shaped "
                f"(B,64,{self.grid_size},{self.grid_size}), got "
                f"{tuple(context_features.shape)}"
            )

        tokens = context_features.flatten(2).transpose(1, 2)
        mask_tokens = target_mask.flatten(1).bool().unsqueeze(-1)
        mask_token = self.mask_token.expand(batch, self.num_tokens, -1)
        tokens = torch.where(mask_tokens, mask_token, tokens)
        tokens = tokens + self.position_embedding
        return self.norm(self.transformer(tokens))


class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class CNNDecoder(nn.Module):
    """
    Nearest-colour expansion decoder.

    Inputs:
      combined_features: (B,64,16,16)
      mask_map:          (B,1,16,16)

    It predicts a coarse 16x16 RGB field, expands that field using nearest
    interpolation (so colours are not blended), and predicts a boundary gate
    plus a small high-resolution colour correction. The gate and correction
    are learned only through the ordinary full-image reconstruction loss.

    The original image and high-resolution encoder skip features are never
    supplied to this decoder.
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(65, 64, kernel_size=3, padding=1),
            nn.GELU(),
            ResidualConvBlock(64),
            ResidualConvBlock(64),
        )

        self.coarse_rgb_head = nn.Conv2d(64, 3, kernel_size=1)
        self.boundary_gate_head = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # Inputs: nearest-expanded latent features (64), coarse RGB (3),
        # and nearest-expanded boundary gate (1).
        self.refinement = nn.Sequential(
            nn.Conv2d(68, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
        )

    def forward(self, combined_features, mask_map):
        if combined_features.ndim != 4 or combined_features.shape[1] != 64:
            raise ValueError(
                "combined_features must have shape (B,64,H,W), got "
                f"{tuple(combined_features.shape)}"
            )
        if mask_map.ndim == 3:
            mask_map = mask_map.unsqueeze(1)
        if mask_map.shape[1] != 1:
            raise ValueError("mask_map must have one channel")
        if mask_map.shape[-2:] != combined_features.shape[-2:]:
            mask_map = F.adaptive_max_pool2d(
                mask_map, combined_features.shape[-2:]
            )

        x = torch.cat(
            [combined_features, mask_map.to(combined_features.dtype)], dim=1
        )
        latent_features = self.stem(x)

        coarse_rgb = torch.sigmoid(self.coarse_rgb_head(latent_features))
        boundary_gate = torch.sigmoid(
            self.boundary_gate_head(latent_features)
        )

        output_size = (64, 64)
        # Nearest expansion keeps each coarse colour constant within its region.
        rgb_expanded = F.interpolate(
            coarse_rgb, size=output_size, mode="nearest"
        )
        features_expanded = F.interpolate(
            latent_features, size=output_size, mode="nearest"
        )
        boundary_expanded = F.interpolate(
            boundary_gate, size=output_size, mode="nearest"
        )

        refinement_input = torch.cat(
            [features_expanded, rgb_expanded, boundary_expanded], dim=1
        )
        colour_correction = 0.5 * torch.tanh(
            self.refinement(refinement_input)
        )

        # The learned gate allows the refinement branch to make corrections
        # primarily where the latent representation indicates a boundary.
        reconstructed = rgb_expanded + boundary_expanded * colour_correction
        return reconstructed.clamp(0.0, 1.0)


def combine_features(context_features, predicted_tokens, target_mask):
    """Use context features at visible cells and predicted features at masked cells."""
    batch, channels, height, width = context_features.shape
    context_tokens = context_features.flatten(2).transpose(1, 2)
    mask_tokens = target_mask.flatten(1).bool().unsqueeze(-1)
    combined = torch.where(mask_tokens, predicted_tokens, context_tokens)
    return combined.transpose(1, 2).reshape(batch, channels, height, width)
