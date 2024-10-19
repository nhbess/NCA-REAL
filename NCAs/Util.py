import sys
sys.path.append('Environment')
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
import _config
import Environment.Shapes as Shapes
from Environment.ContactBoard import ContactBoard
from Environment.Tetromino import Tetromino
from StateStructure import StateStructure


def set_seed(seed=0) -> None:
    logger.info(f"Setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_initial_states(n_states:int, state_structure:StateStructure, board_shape:tuple) -> torch.Tensor:
    '''
    Return a tensor of shape [n_states, state_dim, board_shape[0], board_shape[1]].
    With constant values in the constant channels, and the coordinates in the estimation channels.
    '''
    pool = torch.zeros(n_states, state_structure.state_dim, *board_shape)
    lowx = 0
    lowy = 0
    highx = lowx + board_shape[0]-1
    highy = lowy + board_shape[1]-1
    x, y = torch.meshgrid(torch.linspace(lowx, highx, board_shape[0]), torch.linspace(lowy, highy, board_shape[1]), indexing='ij')
    pool[..., state_structure.constant_channels, :, :] = torch.stack([x, y], dim=0)
    pool[..., state_structure.estimation_channels, :, :] = torch.stack([x, y], dim=0)
    return pool

def create_initial_states_relative(n_states:int, state_structure:StateStructure, board_shape:tuple) -> torch.Tensor:
    '''
    Return a tensor of shape [n_states, state_dim, board_shape[0], board_shape[1]].
    With constant values in the constant channels, and the coordinates in the estimation channels.
    '''
    pool = torch.zeros(n_states, state_structure.state_dim, *board_shape)
    return pool

def moving_contact_masks(n_movements: int, batch_size: int, height: int, width: int, show:bool=False) -> torch.Tensor:
    board = ContactBoard(board_shape=[height, width], tile_size=_config.TILE_SIZE)
    contact_masks = torch.zeros(n_movements, batch_size, height, width)
    tetrominos = []

    #Load tetrominos
    for _ in range(batch_size):
        shape_index =   random.randint(0, len(Shapes.tetrominos)-1)
        shape       =   Shapes.tetrominos[shape_index]
        
        SCALER = 1*_config.TILE_SIZE

        tetro = Tetromino(constructor_vertices=shape, scaler=SCALER)
        tetro.center = np.array([0, 0])
        tetrominos.append(tetro)

    MOVEMENT_SPAN = _config.TILE_SIZE/10
    for batch_index in range(batch_size):
        for movement_index in range(n_movements):
            tetro = tetrominos[batch_index]
            while True:
                displacement = np.random.uniform(-MOVEMENT_SPAN, MOVEMENT_SPAN, size=2)
                if board.has_point_inside(displacement + tetro.center):
                    break
            tetro.translate(displacement)
            tetro.rotate(np.random.uniform(0, 20))
            contact_i = board.get_contact_mask(tetro)
            contact_masks[movement_index, batch_index] = torch.from_numpy(contact_i)
            if False:
                plt.imshow(contact_i)
                plt.show()

    return contact_masks

def calculate_mass_center(sensor_state: torch.Tensor, constant_state:torch.Tensor) -> list:
    if isinstance(sensor_state, torch.Tensor):
        sensor_state = sensor_state.detach().cpu().numpy()
    if isinstance(constant_state, torch.Tensor):
        constant_state = constant_state.detach().cpu().numpy()

    D, H, W = sensor_state.shape

    total_weight = np.sum(sensor_state)
    mass_center = [0,0]
    if total_weight == 0:
        return mass_center
    
    mass_center = [0,0]
    for x in range(W):
        for y in range(H):
            if sensor_state[0][y, x] > 0:
                mass_center[0] += constant_state[0][y, x] * sensor_state[0][y, x]
                mass_center[1] += constant_state[1][y, x] * sensor_state[0][y, x]
    
    mass_center[0] /= total_weight
    mass_center[1] /= total_weight
    
    #Normalize distances
    #mass_center[0] /= (W - 1)
    #mass_center[1] /= (H - 1)

    mass_center = [float(mass_center[0]), float(mass_center[1])]
    return mass_center

def get_target_tensor(sensor_states: torch.Tensor, constant_states: torch.Tensor):
    N, _, H, W = sensor_states.shape
    
    target_tensor = torch.ones(N, 2, H, W)
    for i in range(N):
        sensor_state = sensor_states[i]
        constant_state = constant_states[i]
        
        mass_center = calculate_mass_center(sensor_state, constant_state)
        
        contact_mask = (sensor_state > 0).float()

        target_tensor[i, 0] = mass_center[0] * contact_mask
        target_tensor[i, 1] = mass_center[1] * contact_mask

    return target_tensor

def get_target_tensor_relative(sensor_states: torch.Tensor):
    N, _, H, W = sensor_states.shape
    target_tensor = torch.ones(N, 2, H, W)
    for i in range(N):
        sensor_state = sensor_states[i][0]
        x_coords, y_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

        mcx = torch.sum(x_coords * sensor_state) / torch.sum(sensor_state)
        mcy = torch.sum(y_coords * sensor_state) / torch.sum(sensor_state)

        #relative_x = (x_coords - mcx)*sensor_state
        #relative_y = (y_coords - mcy)*sensor_state
        
        target_tensor[i, 0] = mcx*sensor_state
        target_tensor[i, 1] = mcy*sensor_state

    return target_tensor




def create_initial_states_real_data(n_states:int, state_structure:StateStructure, X:np.array,Y:np.array) -> torch.Tensor:
        '''
        Return a tensor of shape [n_states, state_dim, board_shape[0], board_shape[1]].
        With constant values in the constant channels, and the coordinates in the estimation channels.
        '''
        pool = torch.zeros(n_states, state_structure.state_dim, _config.BOARD_SHAPE[0], _config.BOARD_SHAPE[1])
        x = torch.from_numpy(X)
        y = torch.from_numpy(Y)
        
        pool[..., state_structure.constant_channels, :, :] = torch.stack([x, y], dim=0)
        pool[..., state_structure.estimation_channels, :, :] = torch.stack([x, y], dim=0)
        
        return pool

if __name__ == '__main__':
    pass
