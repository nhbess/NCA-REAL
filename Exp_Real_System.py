import sys

sys.path.append('NCAs')
sys.path.append('Environment')

import pickle
import numpy as np
from loguru import logger
import _config
import _folders
from NCAs.NCA import NCA_CenterFinder
from NCAs.StateStructure import StateStructure
from NCAs.TrainingRealData import TrainerRealData

from NCAs.Util import set_seed
from NCAs.Visuals import simulate_center_finder, visualize_from_file


def run_block(state_structure:StateStructure, data:np.array, files_name:str, seed = None, visualize=False):
    if seed is not None:
        set_seed(seed)

    model = NCA_CenterFinder(state_structure=state_structure)
    #trained_model = Trainer_CenterFinder().train_center_finder(model=model, state_structure=state_structure, experiment_name=files_name)
    trained_model = TrainerRealData(data).train_center_finder(model=model, state_structure=state_structure, experiment_name=files_name)
    _folders.save_model(trained_model=trained_model, experiment_name=files_name)
    logger.success(f"Model {files_name} trained and saved")



    if visualize:
        model = _folders.load_model(files_name)
        for board_shape in [_config.BOARD_SHAPE[0]]:
            simulation_path = f"{_folders.SIMULATIONS_PATH}/{files_name}.npy"            
            
            simulate_center_finder(model = model, 
                                   state_structure = state_structure, 
                                   board_shape = [board_shape,board_shape], 
                                   batch_size=2, 
                                   n_movements=20, 
                                   output_file_name=simulation_path, 
                                   num_steps=15)
            
            animation_path = f"{_folders.VISUALIZATIONS_PATH}/{files_name}.gif"
            visualize_from_file(simulation_path=simulation_path, animation_path=animation_path, state_structure=state_structure)

if __name__ == '__main__':
    import os

    experiment_name=f'EXP_REAL_SYSTEM'
    _folders.set_experiment_folders(experiment_name)
    _config.set_parameters({
        'BOARD_SHAPE' :     [4,4],
        'TRAINING_STEPS' :  2000,
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
    run_block(state_structure=state_structure, data=data, files_name= files_name, seed=None, visualize=False)
