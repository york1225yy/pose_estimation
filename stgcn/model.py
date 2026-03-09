"""ST-GCN model for skeleton-based action recognition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .graph import get_spatial_graph


class SpatialGraphConv(nn.Module):
    """Spatial graph convolution layer."""

    def __init__(self, in_channels, out_channels, A, residual=True):
        super().__init__()
        self.num_subsets = A.shape[0]
        self.A = nn.Parameter(torch.from_numpy(A), requires_grad=False)
        # Learnable edge importance weighting
        self.M = nn.Parameter(torch.ones_like(self.A))
        self.conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for _ in range(self.num_subsets)
        ])
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        """x: (N, C, T, V)"""
        res = self.residual(x)
        out = None
        for k in range(self.num_subsets):
            # xk: (N, C, T, V) @ A_k: (V, V) -> (N, C, T, V)
            A_k = self.A[k] * self.M[k]
            xk = torch.einsum('nctv,vw->nctw', x, A_k)
            xk = self.conv[k](xk)
            out = xk if out is None else out + xk
        return self.relu(self.bn(out) + res)


class TemporalConv(nn.Module):
    """Temporal convolution with kernel_size along time axis."""

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, residual=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=(kernel_size, 1),
                      stride=(stride, 1),
                      padding=(padding, 0)),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return self.relu(self.conv(x) + self.residual(x))


class STGCNBlock(nn.Module):
    """One ST-GCN block: spatial GCN + temporal conv + dropout."""

    def __init__(self, in_channels, out_channels, A, stride=1, dropout=0.0, residual=True):
        super().__init__()
        self.sgcn = SpatialGraphConv(in_channels, out_channels, A, residual=residual)
        self.tconv = TemporalConv(out_channels, out_channels, stride=stride, residual=residual)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.sgcn(x)
        x = self.tconv(x)
        x = self.dropout(x)
        return x


class STGCN(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Network.

    Input shape: (N, C_in, T, V) where
        N = batch size
        C_in = input channels (3 for x,y,z)
        T = temporal length (90 frames)
        V = number of joints (25)
    """

    def __init__(self, num_classes, in_channels=3, graph_strategy='spatial', dropout=0.0):
        super().__init__()

        A = get_spatial_graph(strategy=graph_strategy)

        # Data batch normalization
        self.data_bn = nn.BatchNorm1d(in_channels * A.shape[-1])

        # ST-GCN layers (official 9-layer architecture)
        self.layers = nn.ModuleList([
            STGCNBlock(in_channels, 64, A, residual=False, dropout=dropout),
            STGCNBlock(64, 64, A, dropout=dropout),
            STGCNBlock(64, 64, A, dropout=dropout),
            STGCNBlock(64, 128, A, stride=2, dropout=dropout),
            STGCNBlock(128, 128, A, dropout=dropout),
            STGCNBlock(128, 128, A, dropout=dropout),
            STGCNBlock(128, 256, A, stride=2, dropout=dropout),
            STGCNBlock(256, 256, A, dropout=dropout),
            STGCNBlock(256, 256, A, dropout=dropout),
        ])

        # Classifier
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        x: (N, C, T, V) — e.g., (batch, 3, 90, 25)
        """
        N, C, T, V = x.shape
        # Batch norm on input
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()  # (N, C, T, V)

        for layer in self.layers:
            x = layer(x)

        # Global average pooling over T and V
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        return self.fc(x)
