import sys

sys.path.append('NCAs')
sys.path.append('Environment')

import pickle
import numpy as np
from loguru import logger
import _config
import _folders
from NCAs.NCA import NCA_CenterFinder
from NCAs.NCA_REAL import NCA_REAL
from NCAs.StateStructure import StateStructure
from NCAs.TrainingRealData import TrainerRealData

from NCAs.Util import set_seed


def run_block(state_structure:StateStructure, data:np.array, files_name:str, seed = None):
    if seed is not None:
        set_seed(seed)

    model = NCA_REAL(state_structure=state_structure)
    trained_model = TrainerRealData(data).train_center_finder(model=model, state_structure=state_structure, experiment_name=files_name)
    _folders.save_model(trained_model=trained_model, experiment_name=files_name)
    logger.success(f"Model {files_name} trained and saved")

    
if __name__ == '__main__':

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
