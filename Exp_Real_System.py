import sys

sys.path.append('NCAs')
sys.path.append('Environment')

import copy
import torch
import pickle
import numpy as np
from loguru import logger
import _config
import _folders
from NCAs.NCA import NCA_CenterFinder
from NCAs.NCA_REAL import NCA_REAL
from NCAs.StateStructure import StateStructure
from NCAs.TrainingRealData import TrainerRealData
from NCAs.Util import set_seed, create_initial_states_real_data
from Environment.ContactBoard import ContactBoard

def run_block(state_structure:StateStructure, data:np.array, files_name:str, seed = None):
    if seed is not None:
        set_seed(seed)

    model = NCA_REAL(state_structure=state_structure)
    trained_model = TrainerRealData(data).train_center_finder(model=model, state_structure=state_structure, experiment_name=files_name)
    _folders.save_model(trained_model=trained_model, experiment_name=files_name)
    logger.success(f"Model {files_name} trained and saved")

def some_visuals(experiment_name:str, data:np.array):
    #load model
    model = _folders.load_model(experiment_name).eval()

    CMX = data[:,1]
    CMY = data[:,2]
    VALUES = np.vstack(data[:,-1])
    print(VALUES.shape)
    #pick a random index
    idx = np.random.randint(0, len(data))
    cmx = CMX[idx]
    cmy = CMY[idx]
    values = VALUES[idx].reshape(_config.BOARD_SHAPE)
    
    board = ContactBoard(board_shape=_config.BOARD_SHAPE, tile_size=_config.TILE_SIZE, center=(0,0))
    X_pos,Y_pos = board.sensor_positions

    initial_state = create_initial_states_real_data(n_states=1, state_structure=state_structure, X=X_pos, Y=Y_pos)[0]
            
    # ASSIGN DATA TO INPUT STATES       
    initial_state[..., state_structure.sensor_channels, :, :] = torch.from_numpy(values).unsqueeze(0)
            
    output_states:torch.Tensor = model(initial_state, np.random.randint(*_config.UPDATE_STEPS), return_frames=True)
    estimation_states = output_states[...,state_structure.estimation_channels, :, :]

    print(estimation_states.shape)

    steps = estimation_states.shape[0]
    for i in range(steps):
        estimation = estimation_states[i].detach().cpu().numpy()
        print(estimation)
        sys.exit()
    
    print(f"Real: {cmx, cmy}")

if __name__ == '__main__':

    experiment_name=f'EXP_REAL_SYSTEM'
    _folders.set_experiment_folders(experiment_name)
    _config.set_parameters({
        'BOARD_SHAPE' :     [4,4],
        'TRAINING_STEPS' :  5000,
        'BATCH_SIZE' :      10,
    })

    state_structure = StateStructure(
                        estimation_dim  = 2,    
                        constant_dim    = 2,   
                        sensor_dim      = 1,   
                        hidden_dim      = 10)
    files_name = f'{_config.NEIGHBORHOOD}_{experiment_name}'


    data_path = f"RealData.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    print(data.shape)
    #run_block(state_structure=state_structure, data=data, files_name= files_name, seed=None)
    some_visuals('Chebyshev_EXP_REAL_SYSTEM', data)