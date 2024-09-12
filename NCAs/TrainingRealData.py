import copy
import sys
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR

import _config
from ResultsHandler import ResultsHandler
from StateStructure import StateStructure
from tqdm import tqdm
from Environment.ContactBoard import ContactBoard

class TrainerRealData:
    def __init__(self, data:np.array) -> None:
        self.rh = ResultsHandler()
        self.data = data

    def _normalize_grads(self, model:nn.Module):
        for p in model.parameters():
            p.grad = p.grad/(p.grad.norm() + 1e-8) if p.grad is not None else p.grad

    def _loss_center_finder_all_frames(self, target_states:torch.Tensor, sensible_states:torch.Tensor, contact_masks:torch.Tensor) -> torch.Tensor:
        
        steps = sensible_states.shape[0]
        contact_masks = contact_masks.repeat(steps, 1, 1, 1, 1)
        target_states = target_states.repeat(steps, 1, 1, 1, 1)
       
        masked_sensible_state = sensible_states * contact_masks
        batch_losses = (target_states-masked_sensible_state).pow(2)
        batch_losses_list = batch_losses.reshape(_config.BATCH_SIZE, -1).mean(-1)
        loss = batch_losses_list.mean()

        return loss, batch_losses_list
    
    
    def create_initial_states(self, n_states:int, state_structure:StateStructure, X:np.array,Y:np.array) -> torch.Tensor:
        '''
        Return a tensor of shape [n_states, state_dim, board_shape[0], board_shape[1]].
        With constant values in the constant channels, and the coordinates in the estimation channels.
        '''
        pool = torch.zeros(n_states, state_structure.state_dim, _config.BOARD_SHAPE[0], _config.BOARD_SHAPE[1])
        x = torch.from_numpy(X)
        y = torch.from_numpy(Y)
        
        pool[..., state_structure.constant_channels, :, :] = torch.stack([x, y], dim=0)
        pool[..., state_structure.estimation_channels, :, :] = torch.stack([x, y], dim=0)
        
        return pool

    def train_center_finder(self, model:nn.Module, state_structure:StateStructure, experiment_name:str = None) -> nn.Module:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Training CF model on {device}")
        
        model = model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=_config.LEARNING_RATE, weight_decay=_config.WEIGHT_DECAY)
        scheduler = MultiStepLR(optimizer=optimizer, milestones=_config.MILESTONES, gamma=_config.GAMMA)
        
        
        board = ContactBoard(board_shape=_config.BOARD_SHAPE, tile_size=_config.TILE_SIZE, center=(0,0))
        X_pos,Y_pos = board.sensor_positions
        pool = self.create_initial_states(n_states=_config.POOL_SIZE, state_structure=state_structure, X=X_pos, Y=Y_pos).to(device)        
        
        logger.info(f"Training for {_config.TRAINING_STEPS} epochs")
        self.rh.set_training_start()


        #select _config.TRAINING_STEPS random indexes from the data
        if _config.TRAINING_STEPS > len(self.data):
            logger.error(f"Training steps are greater than the number of data points) must be less than {len(self.data)}")

        
        X = self.data[:,1]
        Y = self.data[:,2]
        S = np.vstack(self.data[:,-1])
        
        for step in tqdm(range(0,_config.TRAINING_STEPS)):

            pool_indexes = np.random.randint(_config.POOL_SIZE, size=[_config.BATCH_SIZE])
            input_states = pool[pool_indexes].to(device)

            indexes = np.random.randint(len(self.data), size=_config.BATCH_SIZE)
            x = X[indexes]
            y = Y[indexes]
            s = np.array([np.reshape(i, _config.BOARD_SHAPE) for i in S[indexes]])
            
            total_batch_losses_list = []
            total_losses = []
                            
            for state in range(_config.BATCH_SIZE):
                
                input_states[..., state_structure.sensor_channels, :, :] = torch.from_numpy(s[state])
                sensor_states = input_states[..., state_structure.sensor_channels, :, :]
                
                N, _, H, W = sensor_states.shape
                target_states = torch.ones(N, 2, H, W)
                for i in range(N):
                    target_states[i, 0] = x[i]
                    target_states[i, 1] = y[i]
    
                #print(f'target state shape: {target_states.shape}')


                #TODO: DO I need dead mask here in the forward pass how does it work since I am passing not a single but BatchSize things?
                states:torch.Tensor = model(input_states, np.random.randint(*_config.UPDATE_STEPS), return_frames=True)
                sensible_states = states[...,state_structure.estimation_channels, :, :] 
                         
                mov_loss, mov_batch_losses_list = self._loss_center_finder_all_frames(target_states, sensible_states, sensor_states)
                total_batch_losses_list.append(mov_batch_losses_list)
                total_losses.append(mov_loss)

             
            total_batch_losses_list = torch.stack(total_batch_losses_list).sum(dim=0)
            loss = torch.stack(total_losses).mean()
            
            loss.backward()
            self._normalize_grads(model)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            indexes_to_replace = torch.argsort(total_batch_losses_list, descending=True)[:int(.15*_config.BATCH_SIZE)]
            final_states = states.detach()[-1]
            empty_states = self.create_initial_states(len(indexes_to_replace), state_structure, X=X_pos, Y=Y_pos).to(device)
            final_states[indexes_to_replace] = empty_states
            pool[pool_indexes] = final_states

            self.rh.add_loss(loss)
            if step%(_config.TRAINING_STEPS//10) == 0 and step != 0:
                data = self.rh.data['training_results'][-1]
                l = data['loss']
                rt = data['time']
                print(f"loss: {round(l*1000,5)}\t\ttotal time: {round(rt,3)}")
        
        self.rh.save_training_results(experiment_name)
        self.rh.plot_loss(experiment_name)
        
        logger.info(f"Finished training model on {device}")
        trained_model = copy.deepcopy(model.cpu())
        return trained_model
    
if __name__ == '__main__':
    pass