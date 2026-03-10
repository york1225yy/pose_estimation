"""Video-based action recognition package.

Models:
    TransDARC  — Transformer-based Domain-Adaptive Recognition via Contrastive learning
    UniFormerV2 — Unified Transformer V2 (ViT-B/16 backbone + temporal attention)
"""
from .dataset import VideoDataset
from .transdamc import TransDARC
from .uniformerv2 import UniFormerV2
