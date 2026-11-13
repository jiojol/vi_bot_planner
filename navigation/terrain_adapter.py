"""Optional lightweight terrain residual for FUTURE training-only fine-tuning.

Zero-initialized output preserves pretrained S1 conditioning exactly. Revision
5's no-retraining algorithm uses the measured cost field without this adapter.
"""
import torch
from torch import nn
from torch.nn import functional as F


class TerrainConditionAdapter(nn.Module):
    def __init__(self, dimension=256):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(5, 32, 3, padding=1), nn.SiLU(),
                                      nn.Conv2d(32, 32, 3, padding=1), nn.SiLU())
        self.output = nn.Conv2d(32, dimension, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, terrain, observed):
        if terrain.ndim != 4 or terrain.shape[1] != 5:
            raise ValueError('Terrain adapter requires [B,5,H,W]')
        confidence = observed[:, None].to(terrain)
        features = self.features(terrain*confidence)
        features = self.output(features)*confidence
        return F.adaptive_avg_pool2d(features, (4, 8)).flatten(2).transpose(1, 2)

    def condition(self, environment, terrain, observed):
        residual = self(terrain, observed)
        if residual.shape != environment.shape:
            raise ValueError('Terrain/environment token shape mismatch')
        return environment+residual
