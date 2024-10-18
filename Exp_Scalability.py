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
from NCAs.NCA_SCALE import NCA_SCALE
from NCAs.StateStructure import StateStructure
from NCAs.TrainingScalability import TrainingScalability
from NCAs.Util import set_seed, create_initial_states_real_data
from Environment.ContactBoard import ContactBoard

def run_block(state_structure:StateStructure, model_name:str, seed = None):
    if seed is not None:
        set_seed(seed)

    model = NCA_SCALE(state_structure=state_structure)
    trained_model = TrainingScalability().train_center_finder(model=model, state_structure=state_structure, experiment_name=model_name)
    _folders.save_model(trained_model=trained_model, model_name=model_name)
    logger.success(f"Model {model_name} trained and saved")

if __name__ == '__main__':

    experiment_name=f'Exp_Scalability'
    _folders.set_experiment_folders(experiment_name)
    _config.set_parameters({
        'BOARD_SHAPE' :     [4,4],
        'TRAINING_STEPS' :  100,
        'BATCH_SIZE' :      10,
    })

    state_structure = StateStructure(
                        estimation_dim  = 2,    
                        constant_dim    = 2,   
                        sensor_dim      = 1,   
                        hidden_dim      = 10)
    
    model_name = 'Scalability'
    run_block(state_structure=state_structure, model_name= model_name, seed=None)
    