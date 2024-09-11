import sys

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

import _config
from StateStructure import StateStructure


class NCA_CenterFinder(nn.Module):
    def __init__(self, state_structure: StateStructure, hidden_dim: int = 128):
        super().__init__()
        self.state_structure = state_structure
        self.state_dim = state_structure.state_dim
        self.out_dimension = state_structure.out_dimension
       
        self.update = nn.Sequential(
            nn.Conv2d(in_channels= self.state_dim, out_channels=3*self.state_dim, kernel_size=3, padding=1, groups=self.state_dim, bias=False),  # perceive
            nn.Conv2d(in_channels= 3*self.state_dim, out_channels =hidden_dim, kernel_size=1),  # process perceptual inputs
            nn.ReLU(),                              # nonlinearity
            nn.Conv2d(in_channels=hidden_dim, out_channels=self.out_dimension, kernel_size=1)     # output a residual update
        )
        self._initialize_pre_processing()
        logger.info(f"NCA initialized with {sum(p.numel() for p in self.parameters() if p.requires_grad)} trainable parameters")

    def _initialize_pre_processing(self):
        self.update[-1].weight.data *= 0				 # initial residual updates should be close to zero
        I = np.outer([0, 1, 0], [0, 1, 0])               # identity filter        
        Sx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0       # Sobel x filter

        kernel = np.stack([I, Sx, Sx.T], axis=0)
        kernel = np.tile(kernel, [self.state_dim, 1, 1])
        
        if _config.NEIGHBORHOOD == 'Chebyshev':
            N = np.array([[1, 1, 1], 
                          [1, 1, 1], 
                          [1, 1, 1]])
        elif _config.NEIGHBORHOOD == 'Manhattan':
            N = np.array([[0, 1, 0], 
                          [1, 1, 1], 
                          [0, 1, 0]])
        else:
            raise Exception(f'Invalid neighborhood {_config.NEIGHBORHOOD}') 
        
        # Perform the matrix multiplication
        kernel = kernel*N.T
        
        self.update[0].weight.data[...] = torch.Tensor(kernel)[:, None, :, :]
        self.update[0].weight.requires_grad = False
    
    def forward(self, input_state:torch.Tensor, num_steps:int, return_frames = False, dropout:float=0.2, dead_mask:torch.Tensor = None) -> torch.Tensor:
        frames = []
        for _ in range(num_steps):
            sensor_mask = input_state[..., self.state_structure.sensor_channels, :, :] #continuous values
            in_contact_mask = (sensor_mask > 0).float() #binary values
            not_in_contact_mask = (sensor_mask == 0).float() #binary values
            constant_mask = input_state[..., self.state_structure.constant_channels, :, :]

            update_mask = torch.rand(*_config.BOARD_SHAPE, device=input_state.device) > dropout # drop some updates to make asynchronous            
            
            #Dead mask is here only to avoid updating dead cells. It does not affect the state of any cell.
            if dead_mask is not None:
                update_mask = update_mask * dead_mask

            base = torch.zeros_like(input_state)
            out_channels = self.update(input_state)
            base[..., self.state_structure.estimation_channels, :, :] = out_channels[..., self.state_structure.out_estimation_channels, :, :]
            base[..., self.state_structure.hidden_channels, :, :] = out_channels[..., self.state_structure.out_hidden_channels, :, :]   
            
            input_state = input_state + update_mask * base # Input state is updated. If mask is false there is no update.
           
            #restore unmodified channels
            input_state[..., self.state_structure.constant_channels, :, :] = constant_mask   #constant channels are not modified
            input_state[..., self.state_structure.sensor_channels, :, :] = sensor_mask       #sensor values are not modified

            #if sensible and not in contact, set to constant. They believe they are the center
            input_state[..., self.state_structure.estimation_channels, :, :] = input_state[..., self.state_structure.estimation_channels, :, :] * in_contact_mask + constant_mask * not_in_contact_mask
            
            frames.append(input_state)
            if torch.isnan(input_state).any():
                print(f"input_state: {input_state}")
                sys.exit()

        if return_frames: return torch.stack(frames)
        else: return input_state

if __name__ == '__main__':
    pass