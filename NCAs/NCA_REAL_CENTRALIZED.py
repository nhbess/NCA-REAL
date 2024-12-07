import sys
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

import _config

class NCA_REAL_CENT(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        H, W = _config.BOARD_SHAPE
        C = 1  # Assuming a single channel input

        # Convolutional layers before linear layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # After conv, feature_dim = 16 * H * W
        input_dim = 16 * H * W
        self.update = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=2)
        )

        logger.info(f"NN initialized with {sum(p.numel() for p in self.parameters() if p.requires_grad)} trainable parameters")

    def forward(self, input_state: torch.Tensor) -> torch.Tensor:
        H, W = _config.BOARD_SHAPE
        x = input_state.view(input_state.size(0), 1, H, W)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        out = self.update(x)
        return out

if __name__ == '__main__':
    pass
