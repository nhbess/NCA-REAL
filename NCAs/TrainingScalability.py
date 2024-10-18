import copy

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR

import _config
from ResultsHandler import ResultsHandler
from StateStructure import StateStructure
from Util import create_initial_states, get_target_tensor, moving_contact_masks
from tqdm import tqdm

class TrainingScalability:
    def __init__(self) -> None:
        self.rh = ResultsHandler()

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
    
    def _create_dead_mask(self, device: torch.device) -> torch.Tensor:
        dead_percent = np.random.randint(*_config.DEAD_PERCENTAGE) / 100
        dead_mask = torch.ones(_config.BOARD_SHAPE, device=device)
        dead_mask = dead_mask.flatten()
        dead_mask[:int(dead_mask.shape[0] * dead_percent)] = 0    
        dead_mask = dead_mask[torch.randperm(dead_mask.shape[0])]
        dead_mask = dead_mask.reshape(_config.BOARD_SHAPE)
        return dead_mask

    def train_center_finder(self, model:nn.Module, state_structure:StateStructure, experiment_name:str = None, robustness:bool = False) -> nn.Module:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Training CF model on {device}")
        
        model = model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=_config.LEARNING_RATE, weight_decay=_config.WEIGHT_DECAY)
        scheduler = MultiStepLR(optimizer=optimizer, milestones=_config.MILESTONES, gamma=_config.GAMMA)

        pool = create_initial_states(_config.POOL_SIZE, state_structure, _config.BOARD_SHAPE).to(device)
        
        
        
        
        logger.info(f"Training for {_config.TRAINING_STEPS} epochs")
        self.rh.set_training_start()

        for step in tqdm(range(1,_config.TRAINING_STEPS+1)):

            pool_indexes = np.random.randint(_config.POOL_SIZE, size=[_config.BATCH_SIZE])
            input_states = pool[pool_indexes].to(device)
            #pool_indexes = list(np.random.randint(config.POOL_SIZE, size=[config.BATCH_SIZE]))
            #input_states = pool[pool_indexes].to(device)



            contact_masks = moving_contact_masks(_config.NUM_MOVEMENTS, _config.BATCH_SIZE, *_config.BOARD_SHAPE).to(device)
            
            total_batch_losses_list = []
            total_losses = []
                            
            for movement in range(_config.NUM_MOVEMENTS):
                #print(input_states[0][0])
                dead_input_states = input_states
                dead_input_states[..., state_structure.sensor_channels, :, :] = contact_masks[movement].unsqueeze(1)
                
                
                sensor_states = dead_input_states[..., state_structure.sensor_channels, :, :]
                constant_states = dead_input_states[..., state_structure.constant_channels, :, :]

                #TODO: I am not saving anything back into the pool!

                target_states = get_target_tensor(sensor_states, constant_states).to(device)

                #TODO: DO I need dead mask here in the forward pass how does it work since I am passing not a single but BatchSize things?
                states:torch.Tensor = model(dead_input_states, np.random.randint(*_config.UPDATE_STEPS), return_frames=True, dead_mask=None)
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
            empty_states = create_initial_states(len(indexes_to_replace), state_structure, _config.BOARD_SHAPE).to(device)
            final_states[indexes_to_replace] = empty_states
            pool[pool_indexes] = final_states

            self.rh.add_loss(loss)
            if step%(_config.TRAINING_STEPS//10) == 0 and step != 0:
                data = self.rh.data['training_results'][-1]
                l = data['loss']
                rt = data['time']
                print(f"loss: {round(l*1000,5)}\t\ttotal time: {round(rt,3)}")
        
        self.rh.save_training_results(experiment_name)
        logger.info(f"Finished training model on {device}")
        trained_model = copy.deepcopy(model.cpu())
        return trained_model