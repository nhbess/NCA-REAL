import sys

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

import _config
from StateStructure import StateStructure


class NCA_REAL_CENT(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        input_dim =  _config.BOARD_SHAPE[0] * _config.BOARD_SHAPE[1]
        self.update = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=2)
        )

        logger.info(f"NN initialized with {sum(p.numel() for p in self.parameters() if p.requires_grad)} trainable parameters")

    
    def forward(self, input_state: torch.Tensor) -> torch.Tensor:
        x = input_state.view(input_state.size(0), -1)
        out = self.update(x)
        return out
                        

if __name__ == '__main__':
    pass