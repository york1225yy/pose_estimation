"""UniFormerV2 — Unified Transformer V2 for video action recognition.

Reference:
  "UniFormerV2: Spatiotemporal Learning by Arming Image ViTs with
   Video UniFormer" (ICCV 2023)
  https://arxiv.org/abs/2211.09552

Architecture overview:
  ┌─────────────────────────────────────────────────────────────┐
  │  Input  (N, 3, T, H, W)                                    │
  │                                                             │
  │  1. Patch embedding  (ViT-B/16 patchify, 14×14 grid)       │
  │     → spatial tokens  (N*T, 196, d=768) + [CLS]            │
  │                                                             │
  │  2. Spatial ViT blocks  (L_local=12, Pre-LN)               │
  │     • standard MHSA + FFN per frame                        │
  │                                                             │
  │  3. Global temporal attention blocks  (L_global=4)          │
  │     • MHSA over T frame-level CLS tokens                   │
  │     • DW-Conv temporal FFN                                  │
  │                                                             │
  │  4. Classification head (global CLS → num_classes)          │
  └─────────────────────────────────────────────────────────────┘

Implementation notes:
  - Spatial backbone initialised from ViT-B/16 ImageNet weights
    (torchvision ViT_B_16_Weights.IMAGENET1K_V1).
  - Only the global temporal blocks are added from scratch.
  - This closely follows the UniFormerV2 paper's "local" + "global"
    design without requiring external pretrained video weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ---------------------------------------------------------------------------
# Helpers — copied from ViT internals for reuse
# ---------------------------------------------------------------------------

def _build_mlp(d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
    hidden = int(d_model * mlp_ratio)
    return nn.Sequential(
        nn.Linear(d_model, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, d_model),
        nn.Dropout(dropout),
    )


# ---------------------------------------------------------------------------
# Global Temporal Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class GlobalTemporalMHSA(nn.Module):
    """
    Multi-head self-attention across T frame-level CLS tokens.
    Incorporates a depth-wise 1D conv temporal bias (MDPA in UniFormerV2).
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Depth-wise temporal conv for local inductive bias (kernel=3)
        self.dw_conv = nn.Conv1d(d_model, d_model, kernel_size=3,
                                 padding=1, groups=d_model, bias=False)

    def forward(self, x):
        """x: (N, T, d_model)  →  (N, T, d_model)"""
        N, T, D = x.shape
        H = self.num_heads

        # Temporal DW-Conv bias
        dw = self.dw_conv(x.transpose(1, 2)).transpose(1, 2)   # (N, T, D)

        qkv = self.qkv(x + dw).reshape(N, T, 3, H, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)      # (3, N, H, T, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(N, T, D)
        return self.proj_drop(self.proj(out))


# ---------------------------------------------------------------------------
# Global Temporal UniFormer Block
# ---------------------------------------------------------------------------

class GlobalTemporalBlock(nn.Module):
    """One global temporal block: Pre-LN MHSA + Pre-LN FFN."""

    def __init__(self, d_model: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = GlobalTemporalMHSA(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = _build_mlp(d_model, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Spatial ViT backbone extracted from torchvision
# ---------------------------------------------------------------------------

class ViTSpatialBackbone(nn.Module):
    """
    ViT-B/16 spatial encoder (torchvision).
    Processes each frame independently, returns per-frame CLS tokens.
    ImageNet pre-trained weights are loaded by default.
    """

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = tv_models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        vit     = tv_models.vit_b_16(weights=weights)

        # Expose sub-modules needed for forward pass
        self.patch_embedding = vit.conv_proj          # (3, H, W) → (d, h, w)
        self.class_token     = vit.class_token         # (1, 1, d)
        self.encoder         = vit.encoder             # TransformerEncoder inc. pos_embed
        self.d_model         = vit.hidden_dim          # 768

        if freeze_backbone:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        """
        x: (N*T, 3, H, W)
        Returns: (N*T, d_model)  — CLS token
        Supports arbitrary spatial resolutions via positional embedding interpolation.
        """
        NT = x.shape[0]
        # Patch embed
        p = self.patch_embedding(x)                    # (NT, d, h, w)
        h, w = p.shape[2], p.shape[3]
        p = p.flatten(2).transpose(1, 2)               # (NT, num_patches, d)
        num_patches = p.shape[1]

        # Prepend CLS
        cls = self.class_token.expand(NT, -1, -1)
        x   = torch.cat([cls, p], dim=1)               # (NT, 1+num_patches, d)

        # Interpolate positional embedding if spatial size changed
        pos_embed = self.encoder.pos_embedding           # (1, 1+N_train, d)
        N_train   = pos_embed.shape[1] - 1
        if num_patches != N_train:
            cls_pe    = pos_embed[:, :1, :]              # (1, 1, d)
            patch_pe  = pos_embed[:, 1:, :]             # (1, N_train, d)
            # Reshape to 2D grid and interpolate
            gs_old = int(N_train ** 0.5)
            patch_pe = patch_pe.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
            patch_pe = F.interpolate(patch_pe, size=(h, w), mode='bilinear',
                                     align_corners=False)
            patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, num_patches, -1)
            pos_embed = torch.cat([cls_pe, patch_pe], dim=1)

        x = x + pos_embed
        x = self.encoder.dropout(x)
        x = self.encoder.layers(x)
        x = self.encoder.ln(x)
        return x[:, 0]                                   # CLS → (NT, d)


# ---------------------------------------------------------------------------
# Full UniFormerV2 model
# ---------------------------------------------------------------------------

class UniFormerV2(nn.Module):
    """
    UniFormerV2 video action recognition model.

    Input : (N, 3, T, H, W)  — video clip (H=W=224 recommended)
    Output: (N, num_classes)
    """

    def __init__(self, num_classes: int, in_channels: int = 3,
                 num_global_layers: int = 4, num_heads: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.spatial_backbone = ViTSpatialBackbone(
            pretrained=pretrained, freeze_backbone=freeze_backbone
        )
        d_model = self.spatial_backbone.d_model  # 768

        # Global temporal blocks
        self.global_blocks = nn.ModuleList([
            GlobalTemporalBlock(d_model, num_heads, mlp_ratio, dropout)
            for _ in range(num_global_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)

    def forward(self, x):
        """x: (N, 3, T, H, W)"""
        N, C, T, H, W = x.shape
        # Per-frame spatial encoding
        x = x.permute(0, 2, 1, 3, 4).reshape(N * T, C, H, W)  # (N*T, 3, H, W)
        cls_tokens = self.spatial_backbone(x)                    # (N*T, 768)
        cls_tokens = cls_tokens.view(N, T, -1)                   # (N, T, 768)

        # Global temporal attention
        for blk in self.global_blocks:
            cls_tokens = blk(cls_tokens)

        # Mean pool over T then classify
        out = cls_tokens.mean(dim=1)   # (N, 768)
        out = self.norm(out)
        return self.head(out)
