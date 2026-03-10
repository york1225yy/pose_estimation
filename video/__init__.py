"""Video-based action recognition package.

Models:
    TransDARC  — Transformer-based Driver Activity Recognition using
                 Latent Space Feature Enhancement with Diffusion Probabilistic Model
                 (ICRA 2023, https://arxiv.org/abs/2209.11478)
"""
from .dataset import VideoDataset
from .transdamc import TransDARC
