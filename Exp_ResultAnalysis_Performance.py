import sys

sys.path.append('NCAs')
sys.path.append('Environment')
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger

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
    
    RUNS = 1000
    
    errors = []

    for i in range(RUNS): 
        cmx, cmy, estimation_states = _run_model(CMX, CMY, VALUES, model, state_structure, board)
        estimation = estimation_states.detach().cpu().numpy()
        estimation_x = estimation[0].flatten()
        estimation_y = estimation[1].flatten()
        mestx = np.mean(estimation_x)
        mesty = np.mean(estimation_y)

        distance_error = np.sqrt((cmx - mestx)**2 + (cmy - mesty)**2)
        errors.append(float(distance_error))

    _folders.save_training_results({'errors': errors}, f'Evaluation_{name}')

def make_plot(names):
    palette = _colors.create_palette(len(names))    
    all_errors = []
    plt.figure(figsize=_colors.FIG_SIZE)

    # First pass: collect all errors to compute the common bin edges
    for name in names:
        results_path = f'{_folders.RESULTS_PATH}/Evaluation_{name}'
        with open(f'{results_path}.json', 'r') as json_file:
            data = json.load(json_file)
        all_errors.extend(data['errors'])

    # Define bin edges based on the global range of errors
    min_error = np.min(all_errors)
    max_error = np.max(all_errors)
    bins = np.linspace(min_error, max_error, 21)

    # Second pass: plot each histogram with the same bins
    for i, name in enumerate(names):
        results_path = f'{_folders.RESULTS_PATH}/Evaluation_{name}'
        with open(f'{results_path}.json', 'r') as json_file:
            data = json.load(json_file)
        errors = data['errors']
        mean = np.round(np.mean(errors), 2)
        std = np.round(np.std(errors), 2)
        label = f'{name}\n$\\mu$: {mean}, $\\sigma$: {std}'
        
        # Calculate histogram data with density
        counts, bin_edges = np.histogram(errors, bins=bins, density=True)
        counts = counts * 100  # Convert to percentage
        
        # Plot histogram
        plt.hist(bin_edges[:-1], bins=bin_edges, weights=counts, alpha=0.75, color=palette[i], label=label)

    plt.legend(loc='upper right', fontsize=10)
    plt.xlabel('Distance Error [mm]')
    plt.ylabel('Frequency [%]')
    #plt.title('Distance Error Histogram')
    image_path = f'{_folders.VISUALIZATIONS_PATH}/Comparison.png'
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
    
    if False:
        for name in NAMES:
            data_path = f"Dataset/TestData_{name}.pkl"
            with open(data_path, 'rb') as f:
                data = pickle.load(f)

            model = _folders.load_model(name)        
            make_data(model, data, name)
    
    make_plot(NAMES)