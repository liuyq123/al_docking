from typing import Optional, List

import torch.nn.functional as F

from src.gnns import GCNModel, GATModel

class ModelCreator:
    """
    Instantiate a model.
    """

    def __init__(self, 
                 model_config: dict) -> None:

        self.config = model_config

    def get_model(self):

        if 'activation' in self.config['params']:
            self.config['params']['activation'] = eval(self.config['params']['activation'])

        models = {
            "gcn": GCNModel,
            "gat": GATModel
        }
        
        model_name = self.config['name']

        return models[model_name](**self.config['params'])
