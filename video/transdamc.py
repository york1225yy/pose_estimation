"""TransDARC — Transformer-based Driver Activity Recognition using Latent Space
Feature Enhancement with Diffusion Probabilistic Model.

Reference:
  K. Peng et al., "TransDARC: Transformer-based Driver Activity Recognition
  using Latent Space Feature Enhancement with Diffusion Probabilistic Model",
  ICRA 2023.  https://arxiv.org/abs/2209.11478
  https://github.com/KPeng9510/TransDARC

Architecture
============
  Input  (N, 3, T, H, W)
  │
  ├─ 1. FrameEncoder  — ResNet-50 per-frame spatial features → (N, T, 2048)
  │
  ├─ 2. TemporalTransformer — [CLS] + T tokens, L × TransformerEncoderLayer
  │       → latent z  (N, d_model)
  │
  ├─ 3. ClassificationHead — linear → (N, num_classes)
  │
  └─ 4. DPM (training only)
         ├─ DiffusionSchedule — linear β schedule, T_ddpm steps
         ├─ q_sample: z_t = √ᾱ_t · z + √(1−ᾱ_t) · ε
         └─ Denoiser (MLP + sinusoidal timestep emb.) — predicts ε from (z_t, t)

Training loss
=============
  L = L_CE(head(z), y)  +  λ · MSE(Denoiser(z_t, t), ε)

Inference
=========
  Only the backbone + head path is used; DPM adds zero overhead.

Fidelity
========
  ✔  ResNet-50 spatial backbone (ImageNet-1k pretrained, optional)
  ✔  Temporal Transformer: Pre-LN, [CLS] token, sinusoidal-like pos-embed
  ✔  Linear DDPM noise schedule (β₁…β_T)
  ✔  MSE denoising loss on latent features (ε-prediction)
  ✔  DPM active only during training; inference is identical to backbone-only
  ✗  Skeleton fusion (paper fuses RGB + skeleton; here: video-only)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ---------------------------------------------------------------------------
# 1. Spatial backbone — ResNet-50 per-frame
# ---------------------------------------------------------------------------

class FrameEncoder(nn.Module):
    """Extract per-frame spatial features with a truncated ResNet-50."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet  = tv_models.resnet50(weights=weights)

        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 2048

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        """x: (N*T, 3, H, W) → (N*T, 2048)"""
        return self.pool(self.backbone(x)).flatten(1)


# ---------------------------------------------------------------------------
# 2. Temporal Transformer
# ---------------------------------------------------------------------------

class TemporalTransformer(nn.Module):
    """
    Temporal self-attention over T frame tokens with a learnable [CLS] token.
    Classification is performed on the CLS output.
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

        # +1 for CLS position
        self.pos_embed = nn.Embedding(max_seq_len + 1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,  # Pre-LN: more stable for vision transformers
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm    = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, tokens):
        """tokens: (N, T, in_dim) → (N, d_model)"""
        N, T, _ = tokens.shape
        x   = self.proj(tokens)                        # (N, T, d_model)
        cls = self.cls_token.expand(N, -1, -1)         # (N, 1, d_model)
        x   = torch.cat([cls, x], dim=1)               # (N, T+1, d_model)
        pos = torch.arange(T + 1, device=x.device)
        x   = x + self.pos_embed(pos)                  # broadcast over batch
        x   = self.encoder(x)                          # (N, T+1, d_model)
        return self.norm(x[:, 0])                      # CLS → (N, d_model)


# ---------------------------------------------------------------------------
# 3. DPM components
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """Standard sinusoidal embedding used in DDPM / Diffusion Transformers."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (N,) integer timesteps → (N, dim)"""
        half   = self.dim // 2
        freqs  = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        emb = t.float()[:, None] * freqs[None]         # (N, half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)  # (N, dim)


class Denoiser(nn.Module):
    """
    Small MLP that predicts the noise ε from noisy latent z_t and timestep t.

    Architecture: 3-layer MLP with SiLU activations.
    Input  : concat(z_t, t_emb)  where t_emb is a sinusoidal embedding
    Output : predicted noise, same shape as z_t
    """

    def __init__(self, d_model: int, t_emb_dim: int = 128):
        super().__init__()
        self.t_embed = SinusoidalTimestepEmbedding(t_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(d_model + t_emb_dim, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        z_t : (N, d_model) — noisy latent
        t   : (N,) integer — timestep indices
        Returns predicted noise (N, d_model)
        """
        t_emb = self.t_embed(t)                        # (N, t_emb_dim)
        return self.net(torch.cat([z_t, t_emb], dim=-1))


class DiffusionSchedule(nn.Module):
    """
    Linear DDPM noise schedule: β linearly increases from beta_start to beta_end
    over T_ddpm steps.  Buffers are on the same device as the module.
    """

    def __init__(self, T: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        super().__init__()
        self.T = T
        betas     = torch.linspace(beta_start, beta_end, T)
        alpha_bar = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer('sqrt_alpha_bar',       alpha_bar.sqrt())
        self.register_buffer('sqrt_one_minus_alpha', (1.0 - alpha_bar).sqrt())

    def q_sample(self, x0: torch.Tensor,
                 t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: sample x_t ~ q(x_t | x_0).

        x0 : (N, d_model) — clean latent
        t  : (N,) integer — timestep indices
        Returns (x_t, noise)  — noisy latent and the actual Gaussian noise
        """
        noise  = torch.randn_like(x0)
        sa     = self.sqrt_alpha_bar[t][:, None]       # (N, 1) — broadcast
        sm     = self.sqrt_one_minus_alpha[t][:, None]
        return sa * x0 + sm * noise, noise


# ---------------------------------------------------------------------------
# 4. Full TransDARC model
# ---------------------------------------------------------------------------

class TransDARC(nn.Module):
    """
    TransDARC: ResNet-50 + Temporal Transformer + DDPM latent augmentation.

    Input  : (N, 3, T, H, W) — video clip tensor
    Output : (N, num_classes) — unnormalised logits

    During training call  `forward_train(x)` to get (logits, dpm_loss).
    During inference call  `forward(x)` — no DPM overhead.
    """

    def __init__(self,
                 num_classes:      int,
                 in_channels:      int   = 3,
                 d_model:          int   = 512,
                 num_layers:       int   = 6,
                 num_heads:        int   = 8,
                 ffn_dim:          int   = 2048,
                 dropout:          float = 0.1,
                 pretrained:       bool  = True,
                 freeze_backbone:  bool  = False,
                 T_ddpm:           int   = 1000,
                 beta_start:       float = 1e-4,
                 beta_end:         float = 0.02,
                 t_emb_dim:        int   = 128):
        super().__init__()

        # Spatial encoder
        self.frame_enc = FrameEncoder(pretrained=pretrained,
                                      freeze_backbone=freeze_backbone)

        # Temporal encoder
        self.temporal = TemporalTransformer(
            in_dim=self.frame_enc.out_dim, d_model=d_model,
            num_layers=num_layers, num_heads=num_heads,
            ffn_dim=ffn_dim, dropout=dropout,
        )

        # Classification head
        self.head = nn.Linear(d_model, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)

        # DPM components (used only during training)
        self.diffusion = DiffusionSchedule(T=T_ddpm, beta_start=beta_start,
                                           beta_end=beta_end)
        self.denoiser  = Denoiser(d_model=d_model, t_emb_dim=t_emb_dim)

    # ------------------------------------------------------------------
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone: (N, 3, T, H, W) → latent z (N, d_model)"""
        N, C, T, H, W = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(N * T, C, H, W)
        feats  = self.frame_enc(frames).view(N, T, -1)   # (N, T, 2048)
        return self.temporal(feats)                       # (N, d_model)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference forward pass.
        x : (N, 3, T, H, W)
        Returns logits (N, num_classes).
        """
        return self.head(self._encode(x))

    # ------------------------------------------------------------------
    def forward_train(self, x: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass with DDPM auxiliary loss.

        x : (N, 3, T, H, W)
        Returns:
            logits   : (N, num_classes)
            dpm_loss : scalar — MSE between predicted noise and actual noise
        """
        z      = self._encode(x)                        # (N, d_model)
        logits = self.head(z)

        # -------- DPM latent-space augmentation --------
        N = z.shape[0]
        t = torch.randint(0, self.diffusion.T, (N,), device=z.device)
        z_t, noise_gt  = self.diffusion.q_sample(z.detach(), t)
        noise_pred     = self.denoiser(z_t, t)
        dpm_loss       = F.mse_loss(noise_pred, noise_gt)
        # -----------------------------------------------

        return logits, dpm_loss

