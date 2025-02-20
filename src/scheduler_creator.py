from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LinearLR, SequentialLR

class SchedulerCreator:
    """
    Instantiate a scheduler.
    """

    def __init__(self, optimization_config):

        self.config = optimization_config

    def get_scheduler(self, optimizer):

        schedulers = {
            "cosine": CosineAnnealingLR,
            "exponential": ExponentialLR,
            "linear": LinearLR
        }
        
        scheduler_name = self.config['scheduler']
        warmup_steps = self.config['warmup_steps']

        if scheduler_name == 'exponential':
            import math
            gamma = math.pow(self.config['scheduler_params']['decay_rate'], 
                            1 / self.config['scheduler_params']['decay_steps'])
            self.config['scheduler_params'] = {'gamma': gamma}

        if warmup_steps != 0:
            scheduler1 = LinearLR(optimizer, 0.001, 1.0, warmup_steps)
            scheduler2 = schedulers[scheduler_name](optimizer, **self.config['scheduler_params'])
            
            scheduler = SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[warmup_steps])
        else:
            scheduler = schedulers[scheduler_name](optimizer, **self.config['scheduler_params'])

        return scheduler