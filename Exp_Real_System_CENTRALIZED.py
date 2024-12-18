import sys

sys.path.append('NCAs')
sys.path.append('Environment')

import matplotlib.pyplot as plt
import imageio
import copy
import torch
import pickle
import numpy as np
from loguru import logger
import _config
import _folders
from NCAs.NCA_REAL_CENTRALIZED import NCA_REAL_CENT
from NCAs.TrainingRealCentralized import TrainerRealData
from NCAs.Util import set_seed, create_initial_states_real_data
from Environment.ContactBoard import ContactBoard

def run_block(data:np.array, model_name:str, seed = None):
    if seed is not None:
        set_seed(seed)

    model = NCA_REAL_CENT()
    trained_model = TrainerRealData(data).train_center_finder(model=model, experiment_name=model_name)



    _folders.save_model(trained_model=trained_model, model_name=model_name)
    logger.success(f"Model {model_name} trained and saved")


if __name__ == '__main__':
    experiment_name=f'Exp_RealSystem'
    _folders.set_experiment_folders(experiment_name)
    _config.set_parameters({
        'BOARD_SHAPE' :     [4,4],
        'TRAINING_STEPS' :  5000,
        'BATCH_SIZE' :      10,
    })
 
    NAMES = ['Calibrated',
             'Uncalibrated',
             ]

    for name in NAMES:
        train_data_path = f"Dataset/TrainData_{name}.pkl"
        test_data_path = f"Dataset/TestData_{name}.pkl"

        train_data = pickle.load(open(train_data_path, 'rb'))
        test_data = pickle.load(open(test_data_path, 'rb'))

        run_block(data=train_data, model_name= f'{name}_Centralized', seed=None)

        model = _folders.load_model(f'{name}_Centralized')

        # Test the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        # Convert all data to float32 upfront to avoid object types.
        CMX = test_data[:, 1].astype(np.float32)
        CMY = test_data[:, 2].astype(np.float32)
        VALUES = np.vstack(test_data[:, -1]).astype(np.float32)

        input_states = torch.from_numpy(VALUES).to(device)
        out = model(input_states)
        out = out.cpu().detach().numpy()

        # Calculate the error
        error = np.sqrt(np.mean((out - np.vstack([CMX, CMY]).T)))
        print(f"Model {name} has an error of {error}")
