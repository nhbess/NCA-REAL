import time
import torch
import _folders

class ResultsHandler:
    def __init__(self) -> None:
        self.training_start = None
        self._data = {}

    @property
    def data(self):
        return self._data
    
    def add_data(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self._data:
                self._data[key] = []
            self._data[key].append(value)

    def set_training_start(self):
        self.training_start = time.time()
        
    def add_loss(self, loss:torch.Tensor):
        training_results = {
            'time': time.time() - self.training_start,
            'loss': loss.item(),
        }
        self.add_data(training_results = training_results)
        
    def save_training_results(self, experiment_name:str):
        experiment_name = 'Training_' + experiment_name
        _folders.save_training_results(self.data, experiment_name)

    def plot_loss(self, experiment_name:str):
        _folders.save_training_plot(self.data, experiment_name)
