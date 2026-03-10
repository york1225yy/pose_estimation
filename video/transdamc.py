"""TransDARC — Transformer-based action recognition.

Design based on:
  "TransDARC: Transformer-based Driver Activity Recognition using Latent Space
  Feature Enhancement with Diffusion Probabilistic Model" (ICRA 2023)
  https://arxiv.org/abs/2209.11478

Architecture overview:
  ┌────────────────────────────────────────────────────────┐
  │  Input  (N, 3, T, H, W)                               │
  │                                                        │
  │  1. Spatial feature extractor (ResNet-50 per-frame)    │
  │     → tokens (N, T, d_model)                          │
  │                                                        │
  │  2. Temporal Transformer encoder                        │
  │     • Positional embedding                             │
  │     • [CLS] token                                      │
  │     • L × TransformerEncoderLayer                      │
  │                                                        │
  │  3. Classification head (CLS → num_classes)            │
  └────────────────────────────────────────────────────────┘

Key details:
  - ResNet-50 truncated before global pool (output: 2048-d spatial avg)
  - d_model = 512, projected from 2048 via a linear bottleneck
  - 6 Transformer encoder layers, 8 heads, feedforward_dim=2048
  - Latent enhancement: dropout + LayerNorm on projected tokens (replaces
    the diffusion module from the paper for purely supervised fine-tuning)
  - ImageNet pre-trained weights via torchvision
"""

import math
import torch
import torch.nn as nn
import torchvision.models as tv_models


# ---------------------------------------------------------------------------
# Spatial backbone (ResNet-50, truncated)
# ---------------------------------------------------------------------------

class FrameEncoder(nn.Module):
    """Extract per-frame spatial features with a truncated ResNet-50."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet  = tv_models.resnet50(weights=weights)

        # Drop final pooling + FC → keep up to layer4 + adaptive pool
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # → (N*T, 2048, 1, 1)
        self.out_dim = 2048

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        """x: (N*T, 3, H, W) → (N*T, 2048)"""
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return x


# ---------------------------------------------------------------------------
# Temporal Transformer
# ---------------------------------------------------------------------------

class TemporalTransformer(nn.Module):
    """
    Temporal self-attention over T frame tokens.
    Uses a learnable [CLS] token; classification is done on the CLS output.
    """

    def __init__(self, in_dim: int = 2048, d_model: int = 512,
                 num_layers: int = 6, num_heads: int = 8,
                 ffn_dim: int = 2048, dropout: float = 0.1,
                 max_seq_len: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_embed = nn.Embedding(max_seq_len + 1, d_model)  # +1 for CLS

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,   # Pre-LN (more stable)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, tokens):
        """
        tokens: (N, T, in_dim)
        returns: (N, d_model)  — CLS representation
        """
        N, T, _ = tokens.shape
        x = self.proj(tokens)                           # (N, T, d_model)

        cls = self.cls_token.expand(N, -1, -1)          # (N, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                # (N, T+1, d_model)

        pos = torch.arange(T + 1, device=x.device)
        x   = x + self.pos_embed(pos)                   # broadcast

        x   = self.encoder(x)                           # (N, T+1, d_model)
        x   = self.norm(x[:, 0])                        # CLS token → (N, d_model)
        return x


# ---------------------------------------------------------------------------
# Full TransDARC model
# ---------------------------------------------------------------------------

class TransDARC(nn.Module):
    """
    TransDARC action recognition model.

    Input : (N, 3, T, H, W)  — video clip
    Output: (N, num_classes) — unnormalised logits
    """

    def __init__(self, num_classes: int, in_channels: int = 3,
                 d_model: int = 512, num_layers: int = 6,
                 num_heads: int = 8, ffn_dim: int = 2048,
                 dropout: float = 0.1, pretrained: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()
        self.frame_enc = FrameEncoder(pretrained=pretrained,
                                      freeze_backbone=freeze_backbone)
        self.temporal   = TemporalTransformer(
            in_dim=self.frame_enc.out_dim, d_model=d_model,
            num_layers=num_layers, num_heads=num_heads,
            ffn_dim=ffn_dim, dropout=dropout,
        )
        self.head = nn.Linear(d_model, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)

    def forward(self, x):
        """x: (N, 3, T, H, W)"""
        N, C, T, H, W = x.shape
        # Process all frames together for efficiency
        x = x.permute(0, 2, 1, 3, 4).reshape(N * T, C, H, W)   # (N*T, 3, H, W)
        feats = self.frame_enc(x)                                  # (N*T, 2048)
        feats = feats.view(N, T, -1)                               # (N, T, 2048)
        cls   = self.temporal(feats)                               # (N, d_model)
        return self.head(cls)
