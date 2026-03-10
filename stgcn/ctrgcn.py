"""CTR-GCN model for skeleton-based action recognition.

Reference:
    "Channel-wise Topology Refinement Graph Convolution for
    Skeleton-Based Action Recognition" (ICCV 2021)
    https://arxiv.org/abs/2107.12213

Key innovation over ST-GCN:
    - CTRGC: each output channel gets its own input-dependent topology
      M^c, added on top of the fixed base adjacency A.
    - MS-TCN: multi-scale temporal convolution with 4 parallel branches.
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

    Computes a dynamic adjacency refinement M from the input features via
    pairwise channel-wise differences, then applies graph conv with A + M.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Internal feature dimension for topology computation
        mid = max(in_channels // 8, 8) if in_channels > 8 else in_channels

        self.conv_q = nn.Conv2d(in_channels, mid, kernel_size=1)  # query
        self.conv_k = nn.Conv2d(in_channels, mid, kernel_size=1)  # key
        self.conv_v = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # value
        self.bn = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()

    def forward(self, x, A):
        """
        x : (N, C, T, V)
        A : (V, V)  — pre-merged static base adjacency
        """
        # Temporal average for topology computation
        q = self.conv_q(x).mean(2)   # (N, mid, V)
        k = self.conv_k(x).mean(2)   # (N, mid, V)

        # Pairwise channel-wise difference → dynamic topology (N, V, V)
        M = self.tanh(q.unsqueeze(-1) - k.unsqueeze(-2))  # (N, mid, V, V)
        M = M.mean(1)                                       # (N, V, V)

        # Combine static + dynamic topology
        A_refined = A.unsqueeze(0) + M   # (N, V, V)   broadcast over batch

        # Graph convolution
        v = self.conv_v(x)                                 # (N, out_C, T, V)
        out = torch.einsum('nctv,nvw->nctw', v, A_refined)
        return self.bn(out)


class CTRGCSpatial(nn.Module):
    """Spatial graph conv block: CTRGC + residual."""

    def __init__(self, in_channels, out_channels, A, residual=True):
        super().__init__()
        # Pre-merge subsets into single (V, V) matrix and register as buffer
        A_merged = A.sum(0).astype(np.float32)             # (V, V)
        self.register_buffer('A', torch.from_numpy(A_merged))

        self.ctrgc = CTRGC(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.res = lambda x: 0
        elif in_channels == out_channels:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
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
