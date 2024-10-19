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
from NCAs.Util import set_seed, moving_contact_masks, create_initial_states, create_initial_states_real_data, create_initial_states_relative
from Environment.ContactBoard import ContactBoard

def run_block(state_structure:StateStructure, model_name:str, seed = None):
    if seed is not None:
        set_seed(seed)

    model = NCA_SCALE(state_structure=state_structure)
    trained_model = TrainingScalability().train_center_finder(model=model, state_structure=state_structure, experiment_name=model_name)
    _folders.save_model(trained_model=trained_model, model_name=model_name)
    logger.success(f"Model {model_name} trained and saved")


def some_visuals(model_name:str, id:int = 0):
    #load model
    model = _folders.load_model(model_name).eval()
    
    #board = ContactBoard(board_shape=_config.BOARD_SHAPE, tile_size=_config.TILE_SIZE, center=(0,0))
    
    initial_state = create_initial_states_relative(n_states=1, state_structure=state_structure, board_shape=_config.BOARD_SHAPE)[0]
    
    initial_state[..., state_structure.sensor_channels, :, :] = moving_contact_masks(1, 1, *_config.BOARD_SHAPE).unsqueeze(1)

    output_states:torch.Tensor = model(initial_state, np.random.randint(*_config.UPDATE_STEPS), return_frames=True)
    estimation_states = output_states[...,state_structure.estimation_channels, :, :]
    sensor_states = output_states[...,state_structure.sensor_channels, :, :]
    steps = estimation_states.shape[0]
    frames = []

    for i in range(steps):



        estimation = estimation_states[i].detach().cpu().numpy()
        estimation = estimation.reshape(2, -1)        
        
        sensor_state = sensor_states[i].detach().cpu().numpy()[0]

        fig, ax = plt.subplots()

        ax.imshow(sensor_state, cmap='grey', alpha=0.9)


        estimation_x = estimation[0].flatten()
        estimation_y = estimation[1].flatten()
        mestx = np.mean(estimation_x)
        mesty = np.mean(estimation_y)
        ax.scatter(estimation_x, estimation_y, c='orange', marker='x', label='Estimation')
        #ax.scatter(cmx, cmy, c='blue', marker='x', label='Real')
        ax.scatter(mestx, mesty, c='red', marker='x', label='Mean Estimation')

        #draw and arrow from the real to the mean estimation
        #ax.arrow(cmx, cmy, mestx-cmx, mesty-cmy, head_width=0.1, head_length=0.1, fc='grey', ec='grey')
        #draw text with the distance between the real and the mean estimation
        #ax.text(cmx + (mestx-cmx)/2, cmy + (mesty-cmy)/2, f'{np.linalg.norm([mestx-cmx, mesty-cmy]):.2f} mm', fontsize=12, color='black')
        # Convert the figure to a NumPy array
        #label
        ax.legend()
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        #imshow values
        
        # Close the figure to free memory
        plt.close(fig)

    # Create a GIF from the list of frames
    visual_path = f'{_folders.VISUALIZATIONS_PATH}/{model_name}_{id}.gif'
    imageio.mimsave(visual_path, frames, fps=60)



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
                        constant_dim    = 0,   
                        sensor_dim      = 1,   
                        hidden_dim      = 10)
    
    model_name = 'Scalability'
    run_block(state_structure=state_structure, model_name= model_name, seed=None)
    
    for i in range (5):
        some_visuals(model_name = model_name, id=i)