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
from NCAs.NCA_REAL import NCA_REAL
from NCAs.StateStructure import StateStructure
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


def _run_model_centralized(CMX:np.array, CMY:np.array, VALUES:np.array, model:NCA_REAL):
    idx = np.random.randint(0, len(VALUES))
    cmx = CMX[idx]
    cmy = CMY[idx]
    values = torch.from_numpy(VALUES[idx]).unsqueeze(0).float()

    output_states:torch.Tensor = model(values)
    return cmx, cmy, output_states


def make_data(model, data:np.array, name:str):
    if 'Centralized' in name:
        make_data_Centralized(model, data, name)
    else:
        make_data_NCA(model, data, name)
    pass

def make_data_Centralized(model, data:np.array, name:str):
    CMX = data[:,1]
    CMY = data[:,2]
    VALUES = np.vstack(data[:,-1])
    
    errors = []

    for i in range(RUNS): 
        cmx, cmy, estimation_states = _run_model_centralized(CMX, CMY, VALUES, model)
        estimation_states = estimation_states.squeeze(0).detach().cpu().numpy()
        estimation_x = estimation_states[0]
        estimation_y = estimation_states[1]
        mestx = np.mean(estimation_x)
        mesty = np.mean(estimation_y)

        distance_error = np.sqrt((cmx - mestx)**2 + (cmy - mesty)**2)
        errors.append(float(distance_error))

    _folders.save_training_results({'errors': errors}, f'Evaluation_{name}')


def make_data_NCA(model, data:np.array, name:str):
    
    board = ContactBoard(board_shape=_config.BOARD_SHAPE, tile_size=_config.TILE_SIZE, center=(0,0))
    CMX = data[:,1]
    CMY = data[:,2]
    VALUES = np.vstack(data[:,-1])
    

    
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

def make_plot_comparison_calibration(names):
    palette = _colors.create_palette(len(names))    
    all_errors = []
    plt.figure(figsize=np.array(_colors.FIG_SIZE))

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


def make_plot_comparison_centralized(names):
    palette = _colors.create_palette(len(names))    
    all_errors = []

    error_filter = np.inf
    # Create figure with two subplots sharing the x-axis
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=np.array(_colors.FIG_SIZE))

    # First pass: collect all errors to compute the common bin edges
    for name in names:
        results_path = f'{_folders.RESULTS_PATH}/Evaluation_{name}'
        with open(f'{results_path}.json', 'r') as json_file:
            data = json.load(json_file)
        all_errors.extend(data['errors'])

    all_errors = [error for error in all_errors if error < error_filter]
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
        #filter errors lower than 75
        errors = [error for error in errors if error < error_filter]
        mean = np.round(np.mean(errors), 2)
        std = np.round(np.std(errors), 2)
        model_name = name.replace('_',' ').replace('Centralized','Cent.')
        label = f'{model_name}\n$\\mu$: {mean}, $\\sigma$: {std}'

        # Calculate histogram data with density
        counts, bin_edges = np.histogram(errors, bins=bins, density=True)
        counts = counts * 100  # Convert to percentage

        # Determine which subplot to use
        ax = axs[0] if i < 2 else axs[1]

        # Plot histogram
        ax.hist(bin_edges[:-1], bins=bin_edges, weights=counts, alpha=0.75, color=palette[i], label=label)

    # Set labels and legends
    axs[0].set_ylabel('Freq. [%]')
    axs[1].set_xlabel('Distance Error [mm]')
    axs[1].set_ylabel('Freq. [%]')
    axs[0].legend(loc='upper right', fontsize=8)
    axs[1].legend(loc='upper right', fontsize=8)

    image_path = f'{_folders.VISUALIZATIONS_PATH}/Comparison_Centralized.png'
    fig.savefig(image_path, dpi=300, bbox_inches='tight')

def make_plot_comparison_centralized(names):
    palette = _colors.create_palette(len(names))    
    all_data = []
    plt.figure(figsize=np.array(_colors.FIG_SIZE))

    # Read all errors for each model and store them
    for name in names:
        results_path = f'{_folders.RESULTS_PATH}/Evaluation_{name}'
        with open(f'{results_path}.json', 'r') as json_file:
            data = json.load(json_file)
        errors = data['errors']
        all_data.append(errors)

    # Plot the data as a boxplot
    model_labels = [name.replace('_',' ').replace('Centralized','Cent.').replace('Calibrated','Cal.').replace('Uncalibrated','Unc.') for name in names]
    bp = plt.boxplot(all_data, labels=model_labels, showfliers=False, patch_artist=True)

    #set fill color
    for i, box in enumerate(bp['boxes']):
        box.set(facecolor=palette[i])

    plt.xlabel('Models')
    plt.ylabel('Distance Error [mm]')

    

    image_path = f'{_folders.VISUALIZATIONS_PATH}/Comparison_Centralized.png'
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
    
    MODEL_NAMES = [ 'Calibrated',
                    'Uncalibrated',
                    ]
    
    DATA_NAMES = [  'Calibrated',
                    'Uncalibrated',
                    ]   
    
    RUNS = 1000
    
    if False:
        for model_name, data_name in zip(MODEL_NAMES, DATA_NAMES):
            data_path = f"Dataset/TestData_{data_name}.pkl"
            with open(data_path, 'rb') as f:
                data = pickle.load(f)

            model = _folders.load_model(model_name)        
            make_data(model, data, model_name)
    
    make_plot_comparison_calibration(MODEL_NAMES)
    
    MODEL_NAMES = [ 'Calibrated',
                    'Calibrated_Centralized',
                    'Uncalibrated',
                    'Uncalibrated_Centralized'
                    ]
    make_plot_comparison_centralized(MODEL_NAMES)
