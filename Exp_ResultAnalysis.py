import sys

sys.path.append('NCAs')
sys.path.append('Environment')
import json
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


def _run_model(CMX:np.array, CMY:np.array, VALUES:np.array, model:NCA_REAL, state_structure:StateStructure, board:ContactBoard):
    idx = np.random.randint(0, len(data))
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

def make_data(model_path:str, data:np.array, experiment_name:str):
    with open(model_path, 'rb') as handle:
        model = pickle.load(handle)
    
    board = ContactBoard(board_shape=_config.BOARD_SHAPE, tile_size=_config.TILE_SIZE, center=(0,0))
    CMX = data[:,1]
    CMY = data[:,2]
    VALUES = np.vstack(data[:,-1])
    
    RUNS = 500
    
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

    _folders.save_training_results({'errors': errors}, experiment_name)

def make_plot(experiment_name:str):
    import matplotlib.pyplot as plt
    results_path = f'{_folders.RESULTS_PATH}/{experiment_name}'
    with open(f'{results_path}.json', 'r') as json_file:
        data = json.load(json_file)
    
    #plot histogram
    errors = data['errors']
    mean = np.mean(errors)
    std = np.std(errors)

    plt.hist(errors, bins=15, alpha=0.75, label='NCA Real Data, $\mu$: {:.2f}, $\sigma$: {:.2f}'.format(mean, std))
    plt.legend(loc='upper right')
    plt.xlabel('Distance Error')
    plt.ylabel('Frequency')
    plt.title('Distance Error Histogram')
    plt.show()

if __name__ == '__main__':

    experiment_name=f'Exp_Performance'
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
    model_path = 'EXP_REAL_SYSTEM\__Models\Chebyshev_EXP_REAL_SYSTEM.pkl'
    #make_data(model_path, data, experiment_name)
    make_plot(experiment_name)