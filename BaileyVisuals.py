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
from NCAs.NCA import NCA_CenterFinder
from NCAs.NCA_REAL import NCA_REAL
from NCAs.StateStructure import StateStructure
from NCAs.TrainingRealData import TrainerRealData
from NCAs.Util import set_seed, create_initial_states_real_data
from Environment.ContactBoard import ContactBoard
import pandas as pd

def some_visuals(experiment_name:str, data:np.array, id:int = 0):
    #load model
    model = _folders.load_model(experiment_name).eval()
    CMX = data[:,1]
    CMY = data[:,2]
    VALUES = np.vstack(data[:,-1])
    #pick a random index
    idx = np.random.randint(0, len(data))
    cmx = CMX[idx]
    cmy = CMY[idx]
    values = VALUES[idx].reshape(_config.BOARD_SHAPE)
    
    board = ContactBoard(board_shape=_config.BOARD_SHAPE, tile_size=_config.TILE_SIZE, center=(0,0))
    X_pos,Y_pos = board.sensor_positions

    initial_state = create_initial_states_real_data(n_states=1, state_structure=state_structure, X=X_pos, Y=Y_pos)[0]
            
    # ASSIGN DATA TO INPUT STATES       
    initial_state[..., state_structure.sensor_channels, :, :] = torch.from_numpy(values).unsqueeze(0)
            
    output_states:torch.Tensor = model(initial_state, np.random.randint(*_config.UPDATE_STEPS), return_frames=True)
    estimation_states = output_states[...,state_structure.estimation_channels, :, :]

    steps = estimation_states.shape[0]
    frames = []
    for i in range(steps):
        estimation = estimation_states[i].detach().cpu().numpy()
        
        # Create a new figure for each frame
        fig, ax = plt.subplots()

        # Plot your board and estimation
        #board.plot()
        ax.imshow(values, cmap='viridis', alpha=0.9, extent=[-board.tile_size*board.board_shape[1]/2, board.tile_size*board.board_shape[1]/2, -board.tile_size*board.board_shape[0]/2, board.tile_size*board.board_shape[0]/2])


        estimation_x = estimation[0].flatten()
        estimation_y = estimation[1].flatten()
        mestx = np.mean(estimation_x)
        mesty = np.mean(estimation_y)
        ax.scatter(estimation_x, estimation_y, c='orange', marker='x', label='Estimation')
        ax.scatter(cmx, cmy, c='blue', marker='x', label='Real')
        ax.scatter(mestx, mesty, c='red', marker='x', label='Mean Estimation')

        #draw and arrow from the real to the mean estimation
        ax.arrow(cmx, cmy, mestx-cmx, mesty-cmy, head_width=0.1, head_length=0.1, fc='grey', ec='grey')
        #draw text with the distance between the real and the mean estimation
        ax.text(cmx + (mestx-cmx)/2, cmy + (mesty-cmy)/2, f'{np.linalg.norm([mestx-cmx, mesty-cmy]):.2f} mm', fontsize=12, color='black')
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
    visual_path = f'{_folders.VISUALIZATIONS_PATH}/{experiment_name}_{id}.gif'
    imageio.mimsave(visual_path, frames, fps=60)

def single_frame(experiment_name:str, data:np.array, id:int = 0):
    #load model
    model = _folders.load_model(experiment_name).eval()
    CMX = data[:,1]
    CMY = data[:,2]
    VALUES = np.vstack(data[:,-1])
    #pick a random index
    idx = np.random.randint(0, len(data))
    cmx = CMX[idx]
    cmy = CMY[idx]
    values = VALUES[idx].reshape(_config.BOARD_SHAPE)
    
    board = ContactBoard(board_shape=_config.BOARD_SHAPE, tile_size=_config.TILE_SIZE, center=(0,0))
    X_pos,Y_pos = board.sensor_positions

    initial_state = create_initial_states_real_data(n_states=1, state_structure=state_structure, X=X_pos, Y=Y_pos)[0]
            
    # ASSIGN DATA TO INPUT STATES       
    initial_state[..., state_structure.sensor_channels, :, :] = torch.from_numpy(values).unsqueeze(0)
            
    output_states:torch.Tensor = model(initial_state, np.random.randint(*_config.UPDATE_STEPS), return_frames=False)
    estimation_states = output_states[...,state_structure.estimation_channels, :, :]
    print(f'{estimation_states.shape}')

    
    estimation = estimation_states.detach().cpu().numpy()
        
    fig, ax = plt.subplots(figsize=(4,4))

    ax.imshow(values, cmap='viridis', alpha=0.9, extent=[-board.tile_size*board.board_shape[1]/2, board.tile_size*board.board_shape[1]/2, -board.tile_size*board.board_shape[0]/2, board.tile_size*board.board_shape[0]/2])
    estimation_x = estimation[0].flatten()
    estimation_y = estimation[1].flatten()
    mestx = np.mean(estimation_x)
    mesty = np.mean(estimation_y)
    ax.scatter(estimation_x, estimation_y, c='orange', marker='x', label='Estimation')
    ax.scatter(cmx, cmy, c='blue', marker='x', label='Real')
    ax.scatter(mestx, mesty, c='red', marker='x', label='Mean Estimation')
    
    ax.arrow(cmx, cmy, mestx-cmx, mesty-cmy, head_width=0.1, head_length=0.1, fc='grey', ec='grey')
    ax.text(cmx + (mestx-cmx)/2, cmy + (mesty-cmy)/2, f'error = {np.linalg.norm([mestx-cmx, mesty-cmy]):.2f} mm', fontsize=12, color='black')
    ax.legend()
    #axist titles 
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    visual_path = f'{_folders.VISUALIZATIONS_PATH}/{experiment_name}_frame_{id}.png'
    plt.savefig(visual_path, dpi=300, bbox_inches='tight')

if __name__ == '__main__':

    experiment_name=f'Exp_RealSystem'
    _folders.set_experiment_folders(experiment_name)
    _config.set_parameters({
        'BOARD_SHAPE' :     [4,4],
    })

    state_structure = StateStructure(
                        estimation_dim  = 2,    
                        constant_dim    = 2,   
                        sensor_dim      = 1,   
                        hidden_dim      = 10)
    
    
    MODEL = 'Calibrated' # OR 'Uncalibrated'
    MODEL = 'Uncalibrated' # OR 'Uncalibrated'

    ROOT = 'Dataset/Threshold0g/ShapeA/recordingA000'

    if MODEL == 'Calibrated':
        which_data = 'calibrated_sensor_data_all_recordings' 
    elif MODEL == 'Uncalibrated':
        which_data = 'uncalibrated_sensor_data'
    else:
        raise ValueError('Model not found')
    
    cm_estimation = pd.read_csv(ROOT + '/center_of_mass_estimate.csv')['CAMERA'].tolist()
    records = pd.read_csv(ROOT + f'/{which_data}.csv').iloc[:, 1:].values.tolist()    
    shape = ROOT[-4]

    print(f'{shape} {cm_estimation} {len(records)}')
    print(f'{records}')  

    sub_data = {
    'shape': [shape] * len(records),
    'cm_x': [cm_estimation[0]] * len(records),
    'cm_y': [cm_estimation[1]] * len(records),
    'values': list(records)
    }

    data = pd.DataFrame(sub_data).to_numpy()
    print(f'{data.shape}')

    # THEY ARE SAVED IN THE FOLDER EXP_REALSYSTEM IN THE VISUALIZATIONS FOLDER

    #this is a gif (no useful for now but my serve you in a presentation)
    some_visuals(MODEL, data, id=0)
    #this is a single frame 
    single_frame(MODEL, data, id=0)