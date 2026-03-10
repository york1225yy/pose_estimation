"""CTR-GCN model for skeleton-based action recognition.

Reference:
    "Channel-wise Topology Refinement Graph Convolution for
    Skeleton-Based Action Recognition" (ICCV 2021)
    https://arxiv.org/abs/2107.12213

Faithful implementation of the full CTR-GCN architecture:

★  CTRGC (spatial module):
     - Keeps A as (k, V, V) — one convolution path per subset, NOT pre-merged.
       (The original paper processes each sub-graph independently and sums,
        exactly as in ST-GCN, but with dynamic topology added per subset.)
     - Dynamic topology M ∈ R^{N×V×V} via scaled dot-product:
           M = tanh( Q·K^T / √C_mid )
       where Q, K are temporally-averaged channel projections of x.
       (Earlier version used subtraction which differs from the paper.)
     - Learnable per-subset scalar α_i weights the dynamic topology:
           A_eff_i = A[i] + α_i · M      (α_i initialised to 0 →
            purely static topology at the start of training)
     - One unified Conv2d produces k*out_C features, then split per subset.

★  MS-TCN (temporal module):  unchanged — 4-branch multi-scale conv.

★  10-layer network: 64×3 → 128×3 → 256×4, stride-down at layers 4 & 7.

Expected parameter count with default 'spatial' strategy (k=3): ~1.4 M
(vs. ST-GCN ~3.1 M;  ST-GCN's 9×1 temporal conv dominates its param count
 while MS-TCN replaces it with 4 lightweight bottleneck branches.)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import get_spatial_graph


# ---------------------------------------------------------------------------
# Spatial: Channel-wise Topology Refinement Graph Convolution
# ---------------------------------------------------------------------------

class CTRGC(nn.Module):
    """
    Channel-wise Topology Refinement Graph Convolution (CTRGC).

    For k A-subsets, computes:
        output = Σ_i  V_i(x)  ×  ( A[i] + α_i · M )
    where M is a shared dynamic topology:
        M = tanh( Q · K^T / √C_mid )          (N, V, V)
    Q, K are temporally-averaged reduced projections of x.
    α_i is a per-subset learnable scalar (init=0).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 k_subsets: int = 3, rel_reduction: int = 8):
        super().__init__()
        self.k            = k_subsets
        self.out_channels = out_channels
        # Reduced mid-channel for topology (min 16; matches paper's rel_reduction=8)
        mid = max(in_channels // rel_reduction, 16)
        self.mid = mid

        # One big conv → k groups of out_channels (more efficient than k separate convs)
        self.conv_v = nn.Conv2d(in_channels, out_channels * k_subsets, 1, bias=False)
        # Shared Q, K for dynamic topology
        self.conv_q = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.conv_k = nn.Conv2d(in_channels, mid, 1, bias=False)

        # Learnable per-subset weight on the dynamic topology (init=0)
        self.alpha = nn.Parameter(torch.zeros(k_subsets))
        self.bn    = nn.BatchNorm2d(out_channels)
        self.tanh  = nn.Tanh()

    def forward(self, x, A):
        """
        x : (N, C, T, V)
        A : (k, V, V)  — raw subset matrices, NOT pre-merged
        """
        N, C, T, V = x.shape

        # Dynamic topology — shared across subsets
        q = self.conv_q(x).mean(dim=2)           # (N, mid, V)  temporal mean
        k = self.conv_k(x).mean(dim=2)           # (N, mid, V)
        # Scaled dot-product: (N, V, mid) @ (N, mid, V) → (N, V, V)
        M = torch.bmm(q.permute(0, 2, 1), k) / (self.mid ** 0.5)
        M = self.tanh(M)                         # (N, V, V)

        # Per-subset graph convolutions, outputs summed
        v   = self.conv_v(x)                                        # (N, out_C*k, T, V)
        v   = v.view(N, self.k, self.out_channels, T, V)            # (N, k, out_C, T, V)
        out = torch.zeros(N, self.out_channels, T, V, device=x.device)
        for i in range(self.k):
            A_eff = A[i].unsqueeze(0) + self.alpha[i] * M           # (N, V, V)
            out   = out + torch.einsum('nctv,nvw->nctw', v[:, i], A_eff)

        return self.bn(out)


class CTRGCSpatial(nn.Module):
    """Spatial graph conv block: CTRGC + residual."""

    def __init__(self, in_channels, out_channels, A, residual=True):
        super().__init__()
        # Keep full (k, V, V) adjacency as a non-trainable buffer
        self.register_buffer('A', torch.from_numpy(A.astype(np.float32)))
        self.ctrgc = CTRGC(in_channels, out_channels, k_subsets=A.shape[0])
        self.relu  = nn.ReLU(inplace=True)

        if not residual:
            self.res = lambda x: 0
        elif in_channels == out_channels:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        res = self.res(x)
        out = self.ctrgc(x, self.A)
        return self.relu(out + res)


# ---------------------------------------------------------------------------
# Temporal: Multi-Scale Temporal Convolution (MS-TCN)
# ---------------------------------------------------------------------------

class MSTemporalConv(nn.Module):
    """
    Multi-Scale Temporal Convolution with four parallel branches:
      (a) 1×1 conv       — point-wise, captures instantaneous patterns
      (b) 3×1 dilation=1 — short-range temporal context
      (c) 3×1 dilation=2 — medium-range temporal context
      (d) MaxPool + 1×1  — aggregate local max activations

    Outputs are concatenated along the channel axis.
    Requires out_channels % 4 == 0.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        assert out_channels % 4 == 0, "out_channels must be divisible by 4 for MS-TCN"
        bch = out_channels // 4

        self.branches = nn.ModuleList([
            # (a) 1×1 with stride
            nn.Sequential(
                nn.Conv2d(in_channels, bch, kernel_size=(1, 1), stride=(stride, 1)),
                nn.BatchNorm2d(bch),
            ),
            # (b) 3×1 dilation=1
            nn.Sequential(
                nn.Conv2d(in_channels, bch, kernel_size=1),
                nn.BatchNorm2d(bch), nn.ReLU(inplace=True),
                nn.Conv2d(bch, bch, kernel_size=(3, 1), stride=(stride, 1),
                          padding=(1, 0), dilation=(1, 1)),
                nn.BatchNorm2d(bch),
            ),
            # (c) 3×1 dilation=2
            nn.Sequential(
                nn.Conv2d(in_channels, bch, kernel_size=1),
                nn.BatchNorm2d(bch), nn.ReLU(inplace=True),
                nn.Conv2d(bch, bch, kernel_size=(3, 1), stride=(stride, 1),
                          padding=(2, 0), dilation=(2, 1)),
                nn.BatchNorm2d(bch),
            ),
            # (d) MaxPool + 1×1
            nn.Sequential(
                nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
                nn.Conv2d(in_channels, bch, kernel_size=1),
                nn.BatchNorm2d(bch),
            ),
        ])

        # Residual: projection when stride or channel change
        if stride == 1 and in_channels == out_channels:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.res(x)
        out = torch.cat([b(x) for b in self.branches], dim=1)
        return self.relu(out + res)


# ---------------------------------------------------------------------------
# CTR-GCN Block & Full Network
# ---------------------------------------------------------------------------

class CTRGCNBlock(nn.Module):
    """One CTR-GCN block: CTRGCSpatial → MSTemporalConv → Dropout."""

    def __init__(self, in_channels, out_channels, A, stride=1, dropout=0.0, residual=True):
        super().__init__()
        self.sgcn = CTRGCSpatial(in_channels, out_channels, A, residual=residual)
        self.tconv = MSTemporalConv(out_channels, out_channels, stride=stride)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.sgcn(x)
        x = self.tconv(x)
        return self.drop(x)


class CTRGCN(nn.Module):
    """
    CTR-GCN: Channel-wise Topology Refinement Graph Convolutional Network.

    10-layer architecture (channels: 64×3 → 128×3 → 256×4):
      Layers 1–3 : 64 ch   (stride=1)
      Layers 4–6 : 128 ch  (stride=2 at layer 4)
      Layers 7–10: 256 ch  (stride=2 at layer 7)

    Input : (N, C_in=3, T=90, V=25)
    Output: (N, num_classes)
    """

    def __init__(self, num_classes, in_channels=3, graph_strategy='spatial', dropout=0.0):
        super().__init__()

        A = get_spatial_graph(strategy=graph_strategy)

        self.data_bn = nn.BatchNorm1d(in_channels * A.shape[-1])

        self.layers = nn.ModuleList([
            CTRGCNBlock(in_channels,  64, A, residual=False, dropout=dropout),
            CTRGCNBlock(64,   64, A, dropout=dropout),
            CTRGCNBlock(64,   64, A, dropout=dropout),
            CTRGCNBlock(64,  128, A, stride=2, dropout=dropout),
            CTRGCNBlock(128, 128, A, dropout=dropout),
            CTRGCNBlock(128, 128, A, dropout=dropout),
            CTRGCNBlock(128, 256, A, stride=2, dropout=dropout),
            CTRGCNBlock(256, 256, A, dropout=dropout),
            CTRGCNBlock(256, 256, A, dropout=dropout),
            CTRGCNBlock(256, 256, A, dropout=dropout),
        ])

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        """x: (N, C, T, V)"""
        N, C, T, V = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()  # (N, C, T, V)

        for layer in self.layers:
            x = layer(x)

        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        return self.fc(x)
