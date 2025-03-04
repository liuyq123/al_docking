import glob
from typing import List
import random
import pickle
import itertools

import dgl

import torch
from torch.utils.data import IterableDataset

class GraphIterableDataset(IterableDataset):
    def __init__(self, 
                 paths, 
                 mode, 
                 shuffle=False, 
                 batch_size=1, 
                 data_transformer=None):
        """
        Parameters
        ----------
        paths: List[str]
            Path(s) to the dataset(s) to load.
        by_shard: bool
            Whether to load all data into memory.
        shuffle: bool
            Whether to shuffle the data.
        batch_size: int
            The number of data points in one batch.
        data_transformer:
        mode: str
            Whether or not to use this class in training mode or prediction mode
        """
        super(GraphIterableDataset).__init__()
        self.paths = paths
        self.mode = mode
        self.dataset = self.load_data()

        if shuffle:
            self.shuffle()
        
        if data_transformer is not None:
            self.transform(data_transformer)
        
        if batch_size > 1:
            self.batch(batch_size)
        
        if self.mode == 'prediction':
            self.n_shards = len(glob.glob(paths + '/*'))
    
    def __getitem__(self, index):
        pass

    def __iter__(self):
        """
        If the mode is training, then the iterator will give a batch of graphs 
        and its corresponding labels. Otherwise, it will return a list of graphs.
        """
        if self.mode == 'training':
            graphs = self.dataset[0]
            labels = self.dataset[1]
            return zip(graphs, labels)
        elif self.mode == 'prediction':
            return self.load_data()

    def load_data(self):
        """
        Load data into the memory. If number of path(s) is greater than one, 
        than data from thoses sources will be merged together. 

        Parameters
        ----------
        paths: List[str]
            A list of strings that point to the location of the data to load.

        Returns
        -------
        dataset:
            A list of dgl graphs with labels (if provided).
        """
        if self.mode == 'training':
            merged_dataset = ([],{'labels':None})

            graphs_to_merge = []
            labels_to_merge = []

            for path in self.paths:

                shards = glob.glob(path)

                for i in range(len(shards)):
                    graphs = dgl.load_graphs(path + '/shard{}'.format(i+1))

                    graphs_to_merge.extend(graphs[0])
                    labels_to_merge.extend(graphs[1]['labels'])

            merged_dataset[0].extend(graphs_to_merge)
            merged_dataset[1]['labels'] = torch.stack(labels_to_merge, 0)
            
            return merged_dataset

        elif self.mode == 'prediction':
            def _load(file):
                with open (file, 'rb') as f:
                    dataset = pickle.load(f)
                return dataset[0]

            def load_generator(files):
                for file in files:
                    yield _load(file)

            n_shards = len(glob.glob(self.paths + '/*'))
            files = [self.paths + '/shard{}'.format(i+1) for i in range(n_shards)]

            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:  
                return load_generator(files)
            else:
                worker_total_num = torch.utils.data.get_worker_info().num_workers
                worker_id = torch.utils.data.get_worker_info().id

                return itertools.islice(map(_load, iter(files)), worker_id, None, worker_total_num)

    def shuffle(self):
        """
        Shuffle the dataset.

        Parameters
        ----------
        dataset:
            The dataset to shuffle.

        Returns
        -------
        shuffled_dataset:
            Shuffled dataset.
        """

        from operator import itemgetter 

        shuffled_indices = list(range(len(self.dataset[0])))
        random.shuffle(shuffled_indices)

        shuffled_0 = itemgetter(*shuffled_indices)(self.dataset[0])
        shuffled_1 = self.dataset[1]['labels'][torch.tensor(shuffled_indices), :]

        self.dataset = (shuffled_0, {'labels': shuffled_1})
    
    def batch(self, batch_size):
        """
        Make the dataset into batches.

        Parameters
        ----------
        dataset:
            The dataset to batch.

        Returns
        -------
        batched_dataset:
            batched dataset.
        """

        batched_features = []
        for i in range(0, len(self.dataset[0]), batch_size):
            batched_features.append(dgl.batch(self.dataset[0][i:i+batch_size]))
        
        all_labels = self.dataset[1]['labels']
        batched_labels = []
        for i in range(0, len(all_labels), batch_size):
            batched_labels.append(all_labels[i:i+batch_size])
        
        self.dataset = (batched_features, batched_labels)

    def transform(self, data_transformer):
        """
        Transform the labels of self.dataset.

        Parameters
        ----------
        data_transformer:

        Returns
        -------
            None
        """
        self.dataset = data_transformer.transform(self.dataset)
