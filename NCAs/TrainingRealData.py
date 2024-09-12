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
from Util import create_initial_states_real_data
class TrainerRealData:
    def __init__(self, data:np.array) -> None:
        self.rh = ResultsHandler()
        self.data = data

    def _normalize_grads(self, model:nn.Module):
        for p in model.parameters():
            p.grad = p.grad/(p.grad.norm() + 1e-8) if p.grad is not None else p.grad

    def _loss_center_finder_all_frames(self, estimation_states:torch.Tensor, cmx:np.array, cmy:np.array) -> torch.Tensor:
        #._loss_center_finder_all_frames(estimation_states, cmx, cmy)
        steps = estimation_states.shape[0]
        cmx = torch.from_numpy(np.array(cmx, dtype=float)) # [B]
        cmy = torch.from_numpy(np.array(cmy, dtype=float)) # [B]

        cmx_tensor = torch.ones(_config.BOARD_SHAPE).repeat(_config.BATCH_SIZE, 1, 1) # [B, H, W]
        cmy_tensor = torch.ones(_config.BOARD_SHAPE).repeat(_config.BATCH_SIZE, 1, 1) # [B, H, W]

        target_x = cmx_tensor * cmx.unsqueeze(1).unsqueeze(2) # [B, H, W]
        target_y = cmy_tensor * cmy.unsqueeze(1).unsqueeze(2) # [B, H, W]
        target_x_repeated = target_x.unsqueeze(0).repeat(steps, 1, 1, 1) # [S, B, H, W]
        target_y_repeated = target_y.unsqueeze(0).repeat(steps, 1, 1, 1) # [S, B, H, W]
        target_x_repeated = target_x_repeated.unsqueeze(2) # [S, B, 1, H, W]
        target_y_repeated = target_y_repeated.unsqueeze(2) # [S, B, 1, H, W]
        target_combined = torch.cat((target_x_repeated, target_y_repeated), dim=2)

        
        batch_losses = (estimation_states-target_combined).pow(2)
        batch_losses_list = batch_losses.reshape(_config.BATCH_SIZE, -1).mean(-1)
        loss = batch_losses_list.mean()
        return loss, batch_losses_list
    

    def train_center_finder(self, model:nn.Module, state_structure:StateStructure, experiment_name:str = None) -> nn.Module:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Training CF model on {device}")
        
        model = model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=_config.LEARNING_RATE, weight_decay=_config.WEIGHT_DECAY)
        scheduler = MultiStepLR(optimizer=optimizer, milestones=_config.MILESTONES, gamma=_config.GAMMA)
        
        # CREATE POOL
        board = ContactBoard(board_shape=_config.BOARD_SHAPE, tile_size=_config.TILE_SIZE, center=(0,0))
        X_pos,Y_pos = board.sensor_positions
        pool = create_initial_states_real_data(n_states=_config.POOL_SIZE, state_structure=state_structure, X=X_pos, Y=Y_pos).to(device)        
        
        logger.info(f"Training for {_config.TRAINING_STEPS} epochs")
        self.rh.set_training_start()


        # PREPARE DATA
        if _config.TRAINING_STEPS > len(self.data):
            logger.error(f"Training steps are greater than the number of data points) must be less than {len(self.data)}")

        CMX = self.data[:,1]
        CMY = self.data[:,2]
        VALUES = np.vstack(self.data[:,-1])
        
        for step in tqdm(range(0,_config.TRAINING_STEPS)):
        #for step in range(0,_config.TRAINING_STEPS):
            # SELECT BATCH
            pool_indexes = np.random.randint(_config.POOL_SIZE, size=[_config.BATCH_SIZE])
            input_states = pool[pool_indexes].to(device)

            # SELECT BATCH DATA 
            indexes = np.random.randint(len(self.data), size=_config.BATCH_SIZE)
            cmx = CMX[indexes]
            cmy = CMY[indexes]
            values = np.array([np.reshape(i, _config.BOARD_SHAPE) for i in VALUES[indexes]])
            
            # ASSIGN DATA TO INPUT STATES       
            input_states[..., state_structure.sensor_channels, :, :] = torch.from_numpy(values).unsqueeze(1)
            
            # UPDATE MODEL
            states:torch.Tensor = model(input_states, np.random.randint(*_config.UPDATE_STEPS), return_frames=True)
            estimation_states = states[...,state_structure.estimation_channels, :, :]
            
            # CALCULATE LOSS
            loss, batch_losses_list = self._loss_center_finder_all_frames(estimation_states, cmx, cmy)
            
            loss.backward()
            self._normalize_grads(model)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            indexes_to_replace = torch.argsort(batch_losses_list, descending=True)[:int(.15*_config.BATCH_SIZE)]
            final_states = states.detach()[-1]
            empty_states = create_initial_states_real_data(len(indexes_to_replace), state_structure, X=X_pos, Y=Y_pos).to(device)
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