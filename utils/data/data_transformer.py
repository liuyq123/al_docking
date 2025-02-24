import torch

class DataTransformer:
    def __init__(self, transformation_strategy):
        self.transformation_strategy = transformation_strategy
        self.params = {}
    
    def transform(self, dataset):
        """
        Transform the labels of self.dataset.

        Parameters
        ----------
        transformation_strategy: str
            The strategy used to transform the labels. If the strategy is
            "normalization", then the new label will be y' = (y - mean(Y)) / std(Y).
            If the strategy is "exponential_transformation", then the new label will
            be y' = 1 - exp(-alpha * y / abs(min(Y))).

        Returns
        -------
            dataset
        """
        if self.transformation_strategy == 'normalization':
            if len(self.params) == 0:
                std, mean = torch.std_mean(dataset[1]['labels'], dim=0)
                self.params['std'] = std
                self.params['mean'] = mean
            else:
                std = self.params['std']
                mean = self.params['mean']

            dataset[1]['labels'] = (dataset[1]['labels'] - mean) / std

        elif self.transformation_strategy == 'exponential_transformation':
            if len(self.params) == 0:
                minimum = torch.abs(torch.min(dataset[1]['labels'][:]))
                self.params['minimum'] = minimum
            else:
                minimum = self.params['minimum']

            dataset[1]['labels'] = 1 - torch.exp(-3 * dataset[1]['labels'] / minimum)
        
        return dataset
