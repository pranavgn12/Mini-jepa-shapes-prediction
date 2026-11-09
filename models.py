import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """
    Encoder with a 16x16 latent grid for 64x64 input images.
    Returns:
      skip32: (B, 32, 32, 32)
      latent16: (B, 64, 16, 16)
    """
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),  # 64 -> 32
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),  # 32 -> 16
        )

    def forward(self, x):
        skip32 = self.block1(x)
        latent16 = self.block2(skip32)
        return skip32, latent16


class JEPAEncoders(nn.Module):
    def __init__(self, ema_decay=0.996):
        super().__init__()

        self.context_encoder = CNNEncoder()
        self.target_encoder = copy.deepcopy(self.context_encoder)

        self.ema_decay = ema_decay
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target_encoder(self):
        for context_param, target_param in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target_param.mul_(self.ema_decay)
            target_param.add_(context_param, alpha=1.0 - self.ema_decay)

        # Keep BatchNorm running statistics synchronized.
        for context_buffer, target_buffer in zip(
            self.context_encoder.buffers(),
            self.target_encoder.buffers(),
        ):
            target_buffer.copy_(context_buffer)

    def forward(self, masked_image, original_image):
        skip32, context_features = self.context_encoder(masked_image)

        with torch.no_grad():
            _, target_features = self.target_encoder(original_image)

        return context_features, skip32, target_features


class JEPAPredictor(nn.Module):
    """Transformer predictor operating on 16x16 = 256 spatial tokens."""
    def __init__(
        self,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        mlp_dim=256,
        dropout=0.0,
    ):
        super().__init__()

        self.num_tokens = 16 * 16
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_tokens, embed_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, context_features, target_mask):
        batch_size, channels, height, width = context_features.shape
        tokens = context_features.flatten(2).transpose(1, 2)
        tokens = tokens + self.position_embedding[:, :tokens.shape[1], :]
        predicted = self.transformer(tokens)
        return self.norm(predicted)


class CNNDecoder(nn.Module):
    """
    Decode a 16x16 latent grid to a complete 64x64 RGB image.
    The 32x32 skip features are suppressed wherever their cells overlap
    the masked input region.
    """
    def __init__(self):
        super().__init__()

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # 16 -> 32
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        self.refine1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # 32 -> 64
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )

        self.output = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, features, skip32, pixel_mask):
        mask32 = F.adaptive_max_pool2d(pixel_mask, skip32.shape[-2:])
        visible32 = 1.0 - mask32
        skip32 = skip32 * visible32

        x = self.up1(features)
        x = torch.cat([x, skip32], dim=1)
        x = self.refine1(x)
        x = self.up2(x)
        return self.output(x)


def combine_features(context_features, predicted_tokens, target_mask):
    """
    Use predicted target embeddings in masked cells and context embeddings
    in visible cells. Shapes:
      context_features: (B, 64, 16, 16)
      predicted_tokens: (B, 256, 64)
      target_mask:      (B, 1, 16, 16)
    """
    batch_size, channels, height, width = context_features.shape

    context_tokens = context_features.flatten(2).transpose(1, 2)
    mask_tokens = target_mask.flatten(1).bool()

    combined_tokens = torch.where(
        mask_tokens.unsqueeze(-1),
        predicted_tokens,
        context_tokens,
    )

    return combined_tokens.transpose(1, 2).reshape(
        batch_size, channels, height, width
    )
