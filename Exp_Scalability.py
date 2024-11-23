import sys

sys.path.append('NCAs')
sys.path.append('Environment')
import json
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
from tqdm import tqdm

import _colors

def run_block(state_structure:StateStructure, model_name:str, seed = None):
    if seed is not None:
        set_seed(seed)

    model = NCA_SCALE(state_structure=state_structure)
    trained_model = TrainingScalability().train_center_finder(model=model, state_structure=state_structure, experiment_name=model_name)
    _folders.save_model(trained_model=trained_model, model_name=model_name)
    logger.success(f"Model {model_name} trained and saved")

def evaluate_scalability(state_structure:StateStructure, model_name:str, seed = None):
    if seed is not None:
        set_seed(seed)

    model = _folders.load_model(model_name).eval()    
    BATCH_SIZE = 50
    BOARDS = [4, 8, 10, 15, 20, 30, 40, 50, 60 , 70, 80, 90, 100]

    results = {}
    for board_shape in tqdm(BOARDS):
        initial_state = create_initial_states_relative(n_states=BATCH_SIZE, state_structure=state_structure, board_shape=[board_shape,board_shape])
        initial_state[..., state_structure.sensor_channels, :, :] = moving_contact_masks(n_movements = 1, batch_size=BATCH_SIZE, height=board_shape, width=board_shape).squeeze(0).unsqueeze(1)

        output_states:torch.Tensor = model(initial_state, np.random.randint(*_config.UPDATE_STEPS), return_frames=False) #only the last state
        estimation_states = output_states[...,state_structure.estimation_channels, :, :]
        sensor_states = output_states[...,state_structure.sensor_channels, :, :]

        H, W = estimation_states.shape[-2:]
        x_coords, y_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        estimations = (estimation_states + torch.stack([x_coords, y_coords], dim=0)*sensor_states).detach().cpu().numpy()

        sensor_masks = sensor_states.detach().cpu().numpy()
        x_coords = x_coords.detach().cpu().numpy()
        y_coords = y_coords.detach().cpu().numpy()

        distance_errors = []
        for i in range(BATCH_SIZE):
            mask = sensor_masks[i][0].flatten()
            cmx = np.mean(x_coords.flatten()[mask > 0])
            cmy = np.mean(y_coords.flatten()[mask > 0])
        
            ex = estimations[i][0].flatten()[mask > 0]
            ey = estimations[i][1].flatten()[mask > 0]

            estimated_cmx = np.mean(ex)
            estimated_cmy = np.mean(ey)
            
            distance_error = np.sqrt((estimated_cmx - cmx) ** 2 + (estimated_cmy - cmy) ** 2)
            distance_errors.append(distance_error)

        distance_errors = np.array(distance_errors)
        mean_distance_error = np.mean(distance_errors)
        stddev_distance_error = np.std(distance_errors)
        print(f'{board_shape}, {mean_distance_error}, {stddev_distance_error}')
        results[board_shape] = (mean_distance_error, stddev_distance_error)

    #save in json
    with open(f'{_folders.RESULTS_PATH}/{model_name}_results.json', 'w') as f:
        json.dump(results, f)

def some_visuals(model_name:str, board_shape:int, id:int = 0):
    model = _folders.load_model(model_name).eval()
    
    initial_state = create_initial_states_relative(n_states=1, state_structure=state_structure, board_shape=[board_shape, board_shape])[0]    
    initial_state[..., state_structure.sensor_channels, :, :] = moving_contact_masks(1, 1, board_shape, board_shape).unsqueeze(1)

    output_states:torch.Tensor = model(initial_state, np.random.randint(*_config.UPDATE_STEPS), return_frames=True)
    estimation_states = output_states[...,state_structure.estimation_channels, :, :]
    sensor_states = output_states[...,state_structure.sensor_channels, :, :]
    steps = estimation_states.shape[0]
    frames = []

    for i in range(steps):
        estimation = estimation_states[i].detach().cpu().numpy()
        H, W = estimation.shape[-2:]
        x_coords, y_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        estimation += torch.stack([x_coords, y_coords], dim=0).detach().cpu().numpy()
        estimation = estimation.reshape(2, -1)
        sensor_state = sensor_states[i].detach().cpu().numpy()[0].flatten()
        estimation_x, estimation_y = estimation
        mestx = np.mean(estimation_x[sensor_state > 0])
        mesty = np.mean(estimation_y[sensor_state > 0])
        sensor_x, sensor_y = np.where(sensor_state.reshape(H, W) > 0)  # Get coordinates where sensor state is active
        real_mestx = np.mean(sensor_x)
        real_mesty = np.mean(sensor_y)


        colors = ['orange' if sensor else 'grey' for sensor in sensor_state]

        # Prepare plot
        fig, ax = plt.subplots()
        ax.imshow(sensor_state.reshape(H, W).T, cmap='grey', alpha=0.9)
        ax.scatter(estimation_x, estimation_y, c=colors, marker='x', label='Estimation')
        ax.scatter(mestx, mesty, c='red', marker='x', label='Mean Estimation')
        ax.scatter(real_mestx, real_mesty, c='blue', marker='x', label='Real Center')
        
        #draw and arrow from the real to the mean estimation
        #ax.arrow(cmx, cmy, mestx-cmx, mesty-cmy, head_width=0.1, head_length=0.1, fc='grey', ec='grey')
        #draw text with the distance between the real and the mean estimation
        #ax.text(cmx + (mestx-cmx)/2, cmy + (mesty-cmy)/2, f'{np.linalg.norm([mestx-cmx, mesty-cmy]):.2f} mm', fontsize=12, color='black')
        # Convert the figure to a NumPy array
        #label
        ax.legend(loc='upper right')
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        
        plt.close(fig)

    # Create a GIF from the list of frames
    visual_path = f'{_folders.VISUALIZATIONS_PATH}/{model_name}_{id}.gif'
    imageio.mimsave(visual_path, frames, fps=60)

def plot_scalability(model_name:str):
    with open(f'{_folders.RESULTS_PATH}/{model_name}_results.json', 'rb') as f:
        results = json.load(f)

    boards = np.array(list(results.keys()))    
    mean_errors = np.array([results[board][0] for board in boards])
    std_errors = np.array([results[board][1] for board in boards])
    boards = boards.astype(int)

    plt.figure(figsize=_colors.FIG_SIZE)

    palette = _colors.create_palette(3)
    color = palette[0]

    #plor results in a line and std as fill_between
    plt.plot(boards, mean_errors, label='Mean Error', color=color)
    plt.fill_between(boards, mean_errors-std_errors, mean_errors+std_errors, alpha=0.2, label='Std Error', color=color)
    plt.xlabel('Board Shape [NxN]')
    plt.ylabel('Distance Error [Tiles]')
    #plt.title('Scalability')
    #ticks to board shapes
    #plt.xticks(boards)
    plt.savefig(f'{_folders.VISUALIZATIONS_PATH}/{model_name}_results.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':

    experiment_name=f'Exp_Scalability'
    _folders.set_experiment_folders(experiment_name)
    _config.set_parameters({
        'BOARD_SHAPE' :     [8,8],
        'TRAINING_STEPS' :  5000,
        'BATCH_SIZE' :      10,
    })

    state_structure = StateStructure(
                        estimation_dim  = 2,    
                        constant_dim    = 0,
                        sensor_dim      = 1,   
                        hidden_dim      = 10)
    
    model_name = 'Scalability'
    #run_block(state_structure=state_structure, model_name= model_name, seed=None)
    #evaluate_scalability(state_structure=state_structure, model_name=model_name, seed=None)
    plot_scalability(model_name=model_name)

    BOARD_SHAPES = [4, 10, 20]
    #for b in BOARD_SHAPES:
    #    some_visuals(model_name = model_name, board_shape = b, id=b)
