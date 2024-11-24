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

def run_block(state_structure:StateStructure, data:np.array, model_name:str, seed = None):
    if seed is not None:
        set_seed(seed)

    model = NCA_REAL(state_structure=state_structure)
    trained_model = TrainerRealData(data).train_center_finder(model=model, state_structure=state_structure, experiment_name=model_name)
    _folders.save_model(trained_model=trained_model, model_name=model_name)
    logger.success(f"Model {model_name} trained and saved")

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

def _threshold_data(data:np.array, thresholds:float):
    shapes = [d[0] for d in data]
    values = [d[-1] for d in data]

    unique_shapes = np.unique(shapes)
    shape_values = {shape:[] for shape in unique_shapes}
    
    for d in data:
        shape_values[d[0]].extend(d[-1])
    
    for s in unique_shapes:
        vals = np.array(shape_values[s])
        #vals = vals[vals > 0]
        #plot histogram
        plt.hist(vals, bins=100, alpha=0.5, label=f'{s}')
    plt.legend()
    plt.show()
    return


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
    

    NAMES = ['Calibrated',
             'Uncalibrated',
             'Calibrated_0g',
             'Calibrated_25g',
             'Calibrated_40g',]
    
    LEN_DATA_BLOCK = 50

    for name in NAMES:
        data_path = f"Dataset/RealData_{name}.pkl"
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        _threshold_data(data, 0)
        sys.exit()
        data_blocks_indexes = np.arange(0, len(data), LEN_DATA_BLOCK)
        train_indexes = np.random.choice(data_blocks_indexes, int(len(data_blocks_indexes)*0.5), replace=False)
        test_indexes = np.array([i for i in data_blocks_indexes if i not in train_indexes])
        
        train_indexes.sort()
        test_indexes.sort()
        
        train_data = np.vstack([data[i:i+LEN_DATA_BLOCK] for i in train_indexes])
        print(name)
        print(train_data)
        print(train_data.shape)

        test_data = np.vstack([data[i:i+LEN_DATA_BLOCK] for i in test_indexes])
        
        with open(f"Dataset/TrainData_{name}.pkl", 'wb') as f:
            pickle.dump(train_data, f)
        with open(f"Dataset/TestData_{name}.pkl", 'wb') as f:
            pickle.dump(test_data, f)

        run_block(state_structure=state_structure, data=train_data, model_name= name, seed=None)

        if True:
            for i in range(5):
                some_visuals(name, test_data, i)