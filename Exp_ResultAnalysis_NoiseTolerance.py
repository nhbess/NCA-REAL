import sys

sys.path.append('NCAs')
sys.path.append('Environment')
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from loguru import logger
from tqdm import tqdm

import _colors
import _config
import _folders
from Environment.ContactBoard import ContactBoard
from NCAs.NCA import NCA_CenterFinder
from NCAs.NCA_REAL import NCA_REAL
from NCAs.StateStructure import StateStructure
from NCAs.TrainingRealData import TrainerRealData
from NCAs.Util import create_initial_states_real_data, set_seed


def _run_model(CMX:np.array, CMY:np.array, VALUES:np.array, model:NCA_REAL, state_structure:StateStructure, board:ContactBoard):
    idx = np.random.randint(0, len(VALUES))
    cmx = CMX[idx]
    cmy = CMY[idx]
    values = VALUES[idx].reshape(_config.BOARD_SHAPE)
    X_pos,Y_pos = board.sensor_positions
    initial_state = create_initial_states_real_data(n_states=1, state_structure=state_structure, X=X_pos, Y=Y_pos)[0]
    # ASSIGN DATA TO INPUT STATES       
    initial_state[..., state_structure.sensor_channels, :, :] = torch.from_numpy(values).unsqueeze(0)
    output_states:torch.Tensor = model(initial_state, np.random.randint(*_config.UPDATE_STEPS), return_frames=False)
    estimation_states = output_states[...,state_structure.estimation_channels, :, :]
    return cmx, cmy, estimation_states

def make_data(model, data:np.array, name:str):
    
    board = ContactBoard(board_shape=_config.BOARD_SHAPE, tile_size=_config.TILE_SIZE, center=(0,0))
    CMX = data[:,1]
    CMY = data[:,2]
    VALUES = np.vstack(data[:,-1])
    
    N_FAULTS = _config.BOARD_SHAPE[0]*_config.BOARD_SHAPE[1]
    FAULTS = np.arange(N_FAULTS+1)
    RUNS_PER_FAULT = 100
    
    errors = {}

    for faulty in tqdm(FAULTS):
        values = np.copy(VALUES)
        faulty_mask = np.ones_like(values)
        for m in faulty_mask:
            m[np.random.choice(np.arange(len(m)), faulty, replace=False)] = 0

        values = values * faulty_mask
        errors_run = []
        for i in range(RUNS_PER_FAULT):
            cmx, cmy, estimation_states = _run_model(CMX, CMY, values, model, state_structure, board)
            estimation = estimation_states.detach().cpu().numpy()
            estimation_x = estimation[0].flatten()
            estimation_y = estimation[1].flatten()
            mestx = np.mean(estimation_x)
            mesty = np.mean(estimation_y)

            distance_error = np.sqrt((cmx - mestx)**2 + (cmy - mesty)**2)
            errors_run.append(float(distance_error))
        errors[str(faulty)] = errors_run
    _folders.save_training_results({'errors': errors}, f'NoiseTolerance_{name}')

def make_plot(names):
    palette = _colors.create_palette(len(names))    
    
    # First pass: collect all errors to compute the common bin edges
    for i, name in enumerate(names):
        results_path = f'{_folders.RESULTS_PATH}/NoiseTolerance_{name}'
        with open(f'{results_path}.json', 'r') as json_file:
            data = json.load(json_file)
        
        data_errors = data['errors']
        n_faulty_tiles = np.array(list(data_errors.keys()), dtype=int)
        percent_n_faulty_tiles = n_faulty_tiles / (_config.BOARD_SHAPE[0]*_config.BOARD_SHAPE[1])
        errors = np.array([data_errors[str(n)] for n in n_faulty_tiles])
        means = np.mean(errors, axis=1)
        stds = np.std(errors, axis=1)

        #plot mean and std as filled area
        color = palette[i]
        plt.fill_between(percent_n_faulty_tiles, means-stds, means+stds, color=color, alpha=0.3)
        plt.plot(percent_n_faulty_tiles, means, label=name, color=color)
    
    plt.legend(loc='upper right')
    plt.xlabel('Faulty Tiles')
    plt.ylabel('Distance Error')
    #plt.title('Distance Error Histogram')
    
    image_path = f'{_folders.VISUALIZATIONS_PATH}/NoiseTolerance_.png'
    plt.savefig(image_path, dpi=300, bbox_inches='tight')

if __name__ == '__main__':

    experiment_name=f'Exp_RealSystem'
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
    
    NAMES = ['Calibrated','Uncalibrated']
    
    if True:
        for name in NAMES:
            data_path = f"Dataset/TestData_{name}.pkl"
            with open(data_path, 'rb') as f:
                data = pickle.load(f)

            model = _folders.load_model(name)        
            make_data(model, data, name)
    
    #make_plot(NAMES)