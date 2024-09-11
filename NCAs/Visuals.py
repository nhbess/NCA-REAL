import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from matplotlib.animation import FuncAnimation
from torch import nn

from StateStructure import StateStructure
from Util import calculate_mass_center, moving_contact_masks


def plot_sensor_sensible_states(init_state:torch.Tensor, state_structure:StateStructure):
    n = init_state.shape[0]
    board_shape = init_state.shape[-2:]
    if n == 1:
        init_state = init_state.squeeze(0)
        sensible = init_state[..., state_structure.estimation_channels, :, :]
        sensor = init_state[..., state_structure.sensor_channels, :, :]
        
        x_channel = np.array(sensible[0, :, :].tolist()).flatten()
        y_channel = np.array(sensible[1, :, :].tolist()).flatten()
        
        plt.imshow(sensor[0], cmap='gray', vmin=0, vmax=1)
        plt.scatter(x_channel*(board_shape[0]-1), y_channel*(board_shape[1]-1), color='red', marker='x')

        
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()
        return
    
    fig, axs = plt.subplots(1, n, figsize=(n * 2, 2))
    for i in range(n):
        sensible = init_state[i, ..., state_structure.estimation_channels, :, :]
        sensor = init_state[i, ..., state_structure.sensor_channels, :, :]
        axs[i].imshow(sensor[0], cmap='gray', vmin=0, vmax=1)
        x_channel = np.array(sensible[0, :, :].tolist()).flatten()
        y_channel = np.array(sensible[1, :, :].tolist()).flatten()

        axs[i].scatter(x_channel*(board_shape[0]-1), y_channel*(board_shape[1]-1), color='red', marker='x')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.tight_layout()
    plt.show()


def create_random_initial_states_dynamic(batch_size:int, state_structure:StateStructure, board_shape:tuple, n_movements:int = 1):
    '''
    Returns a tensor of shape (n_movements, batch_size, state_dim, height, width)
    '''
    
    contact_mask = moving_contact_masks(n_movements, batch_size, *board_shape, show=False)
    contact_mask = contact_mask.unsqueeze(2)
    init_state = torch.zeros(n_movements,batch_size,state_structure.state_dim, *board_shape)
    
    lowx = 0
    lowy = 0
    highx = lowx + board_shape[0]-1
    highy = lowy + board_shape[1]-1

    x, y = torch.meshgrid(torch.linspace(lowx, highx, board_shape[0]), torch.linspace(lowy, highy, board_shape[1]), indexing='ij')
    
    init_state[..., state_structure.constant_channels, :, :] = torch.stack([x, y], dim=0)
    init_state[..., state_structure.estimation_channels, :, :] = torch.stack([x, y], dim=0) 
    init_state[..., state_structure.sensor_channels, :, :] = contact_mask

    #plot_states(init_state, state_structure)
    return init_state


def simulate_center_finder(model:nn.Module, state_structure: StateStructure, board_shape:tuple, batch_size:int, n_movements:int, output_file_name:str = None, num_steps = 50) -> torch.Tensor:
    logger.info(f'Simulating {batch_size}')
    init_states = create_random_initial_states_dynamic(batch_size, state_structure, board_shape, n_movements)
    model = model.cpu()
    all_frames = []
    with torch.no_grad():
        for movement in range(n_movements):
            init_state = init_states[movement]
            simulated_frames = model.forward(input_state=init_state, num_steps= num_steps, return_frames = True)
            simulated_frames = simulated_frames.detach().numpy()
            all_frames.append(simulated_frames)
  
    np.save(output_file_name, all_frames)


def visualize_from_file(simulation_path: str, animation_path:str, state_structure: StateStructure):
    logger.info(f'Visualizing dynamic input frames from file: {simulation_path}')

    simulation = np.load(file=simulation_path)
    movements = simulation.shape[0]
    steps = simulation.shape[1]
    batches = simulation.shape[2]
    board_shape = simulation.shape[-2:]
    
    reshaped = simulation.reshape(movements*steps, batches, state_structure.state_dim, *board_shape)
    
    for b in range(batches):
        batch = reshaped[:,b, ...]
        sensible = batch[..., state_structure.estimation_channels, :, :]
        sensor = batch[..., state_structure.sensor_channels, :, :]
        constant = batch[..., state_structure.constant_channels, :, :]
        mass_centers = [calculate_mass_center(sensor[i, :, :, :], constant_state=constant[i, :, :, :]) for i in range(steps*movements)]    
        
        fig, ax = plt.subplots()
        def update(step):
            mass_center = mass_centers[step]
            ax.clear()
            ax.imshow(sensor[step, 0, :, :], cmap='gray', vmin=0, vmax=1)
            x_channel = np.array(sensible[step, 0, :, :].tolist()).flatten()/board_shape[0]
            y_channel = np.array(sensible[step, 1, :, :].tolist()).flatten()/board_shape[1]
            
            ax.scatter(y_channel*(board_shape[0]), x_channel*(board_shape[1]), color='red', marker='o')
            ax.scatter(mass_center[1]*(board_shape[0])/board_shape[0], mass_center[0]*(board_shape[1])/board_shape[1], color='blue', marker='x')

            ax.set_title(f'Step: {step + 1}')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()

        ani = FuncAnimation(fig, update, frames=steps*movements, repeat=False)
        path = animation_path.split('.')[0]
        path = f'{path}_{b}.gif'
        ani.save(f'{path}', writer='pillow', fps=20)
        plt.close()

def visualize_tensor(tensor: torch.Tensor, state_structure: StateStructure, show_channels:bool = False):
    batch_size = tensor.shape[0]
    channels = state_structure.state_dim - (state_structure.hidden_channels.stop - state_structure.hidden_channels.start)
    
    for batch_idx in range(batch_size):
        print(f'Generating video for batch {batch_idx} of {batch_size}')
        batch_tensor = tensor[batch_idx]  
        batch_tensor = batch_tensor.squeeze(0)
        sensible = batch_tensor[..., state_structure.estimation_channels, :, :]
        sensor = batch_tensor[..., state_structure.sensor_channels, :, :]
        constants = batch_tensor[..., state_structure.constant_channels, :, :]
        
        if not show_channels:
            fig, axs = plt.subplots(1, channels, figsize=(channels * 2,2))  # Adjust figsize based on the number of channels

            axs[0].imshow(sensor[0], cmap='gray', vmin=0, vmax=1)
            axs[0].set_title('Sensor state')
            axs[1].imshow(sensible[0], cmap='gray', vmin=0, vmax=1)
            axs[1].set_title('Sensible state x')
            axs[2].imshow(sensible[1], cmap='gray', vmin=0, vmax=1)
            axs[2].set_title('Sensible state y')
            axs[3].imshow(constants[0], cmap='gray', vmin=0, vmax=1)
            axs[3].set_title('Constant state x')
            axs[4].imshow(constants[1], cmap='gray', vmin=0, vmax=1)
            axs[4].set_title('Constant state y')

            for c in range(channels):
                axs[c].axis('off')
            
            plt.tight_layout()
            plt.show()
            return

        fig, axs = plt.subplots(2, channels, figsize=(channels * 2, 3))  # Adjust figsize based on the number of channels

        axs[0, 0].imshow(sensor[0], cmap='gray', vmin=0, vmax=1)
        axs[0, 0].set_title('Sensor state')
        axs[0, 1].imshow(sensible[0], cmap='gray', vmin=0, vmax=1)
        axs[0, 1].set_title('Sensible state x')
        axs[0, 2].imshow(sensible[1], cmap='gray', vmin=0, vmax=1)
        axs[0, 2].set_title('Sensible state y')
        axs[0, 3].imshow(constants[0], cmap='gray', vmin=0, vmax=1)
        axs[0, 3].set_title('Constant state x')
        axs[0, 4].imshow(constants[1], cmap='gray', vmin=0, vmax=1)
        axs[0, 4].set_title('Constant state y')

        channels_list = [sensor[0], sensible[0], sensible[1], constants[0], constants[1]    ]

        for c in range(channels):
            channel = channels_list[c]
            N, M = channel.shape
            table_data = [['{:.2f}'.format(channel[i, j]) for j in range(M)] for i in range(N)]

            axs[1, c].axis('off')
            axs[1, c].table(cellText=table_data, loc='center', cellLoc='center')
            axs[1, c].tables[0].auto_set_font_size(True)
            axs[1, c].set_aspect('equal')
            axs[1, c].tables[0].scale(1, 1.2)

        for ax in axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    pass