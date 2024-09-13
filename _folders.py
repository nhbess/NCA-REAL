import json
import os
import pickle

import torch.nn as nn
from loguru import logger

import _config

# DEFAULT FOLDER EXPERIMENT STRUCTURE
DEFAULT_FOLDERS = {
    'MODELS': '__Models',
    #'SIMULATIONS': '__Simulations',
    'VISUALIZATIONS': '__Visualizations',
    'RESULTS': '__Results',
}

# FOLDER PATHS
MODELS_PATH =           None
#SIMULATIONS_PATH =      None
VISUALIZATIONS_PATH =   None
RESULTS_PATH =          None

def _folder_paths() -> dict:
    paths = {}
    for key in DEFAULT_FOLDERS.keys():
        paths[key] = globals()[f'{key}_PATH']
    return paths

def _set_folders(configuration: dict = None):
    for key, value in DEFAULT_FOLDERS.items():
        globals()[f'{key}_PATH'] = value

    if configuration is not None:
        for key, value in configuration.items():
            if key not in DEFAULT_FOLDERS:
                raise Exception(f'DEFAULT_FOLDERS does not have the attribute {key}')
            globals()[f'{key}_PATH'] = value

def _create_folders():
    for folder_path in _folder_paths().values():
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

# Set default and configure if needed
_set_folders()
#create_folders()

def set_experiment_folders(experiment_name:str):
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    _set_folders({
        'MODELS': f'{experiment_name}/{MODELS_PATH}',
        #'SIMULATIONS': f'{experiment_name}/{SIMULATIONS_PATH}',
        'VISUALIZATIONS': f'{experiment_name}/{VISUALIZATIONS_PATH}',
        'RESULTS': f'{experiment_name}/{RESULTS_PATH}',
    })
    _create_folders()



def save_model(trained_model:nn.Module, model_name:str ) -> None:
    path = f'{MODELS_PATH}/{model_name}.pkl'
    logger.info(f"Saving model to path {path}")
    with open(path, 'wb') as handle:
        pickle.dump(trained_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_model(model_name:str) -> nn.Module:
    path = f'{MODELS_PATH}/{model_name}.pkl'
    logger.info(f"Loading model from path {path}")
    with open(path, 'rb') as handle:
        model = pickle.load(handle)
    return model

def save_training_results(data, experiment_name):
    results = {'training_parameters': _config.training_parameters(),}
    results.update(data)

    result_path = os.path.join(RESULTS_PATH, f'{experiment_name}.json')
    print(f"Saving training results to path {result_path}")
    with open(result_path, 'w') as json_file:
        json.dump(results, json_file)

def save_training_plot(data, experiment_name):
    import matplotlib.pyplot as plt
    data = data['training_results']
    loss = [x['loss'] for x in data]
    y = range(len(loss))
    plt.plot(y, loss)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    #plt.yscale('log')
    image_path = os.path.join(VISUALIZATIONS_PATH, f'{experiment_name}.png')
    plt.savefig(image_path)
    #clear plot
    plt.clf()
    plt.close()
    #plt.show()