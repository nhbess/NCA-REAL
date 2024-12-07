import copy
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR

import _config
from ResultsHandler import ResultsHandler
from StateStructure import StateStructure
from tqdm import tqdm

class TrainerRealData:
    def __init__(self, data: np.array) -> None:
        self.rh = ResultsHandler()
        self.data = data

    def _normalize_grads(self, model: nn.Module):
        for p in model.parameters():
            if p.grad is not None:
                p.grad = p.grad / (p.grad.norm() + 1e-8)

    def train_center_finder(self, model: nn.Module, experiment_name: str = None) -> nn.Module:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Training CF model on {device}")
        
        model = model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=_config.LEARNING_RATE, weight_decay=_config.WEIGHT_DECAY)
        scheduler = MultiStepLR(optimizer=optimizer, milestones=_config.MILESTONES, gamma=_config.GAMMA)
        
        if _config.TRAINING_STEPS > len(self.data):
            logger.error(f"Training steps are greater than the number of data points. Must be <= {len(self.data)}")

        logger.info(f"Training for {_config.TRAINING_STEPS} epochs")
        self.rh.set_training_start()

        CMX = self.data[:, 1]
        CMY = self.data[:, 2]
        VALUES = np.vstack(self.data[:, -1]) # shape: [N, H*W]

        H, W = _config.BOARD_SHAPE

        for step in tqdm(range(_config.TRAINING_STEPS)):
            # Select a batch of data points
            batch_indexes = np.random.randint(0, len(self.data), size=_config.BATCH_SIZE)

            # Extract input values and reshape to [B, 1, H, W]
            input_values = VALUES[batch_indexes] # shape: [B, H*W]
            input_values = input_values.reshape(_config.BATCH_SIZE, 1, H, W)
            input_states = torch.from_numpy(input_values).float().to(device)

            # Extract targets
            cmx = CMX[batch_indexes]
            cmy = CMY[batch_indexes]

            # Forward pass
            out = model(input_states)  # out: [B, 2]

            # Create target tensor [B, 2]
            cmx_tensor = torch.from_numpy(cmx).float().to(device)
            cmy_tensor = torch.from_numpy(cmy).float().to(device)
            target = torch.stack([cmx_tensor, cmy_tensor], dim=1)  # [B, 2]

            # Calculate loss
            loss = F.mse_loss(out, target)
            
            loss.backward()
            self._normalize_grads(model)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            self.rh.add_loss(loss)
            if step % (_config.TRAINING_STEPS // 10) == 0 and step != 0:
                data = self.rh.data['training_results'][-1]
                l = data['loss']
                rt = data['time']
                print(f"loss: {round(l*1000,5)}\t\ttotal time: {round(rt,3)}")

        self.rh.save_training_results(experiment_name)
        self.rh.plot_loss(f'Loss_{experiment_name}')
        
        logger.info(f"Finished training model on {device}")
        trained_model = copy.deepcopy(model.cpu())
        return trained_model

if __name__ == '__main__':
    pass
