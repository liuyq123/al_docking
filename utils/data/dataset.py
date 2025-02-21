import glob
from typing import List
import random

import dgl

import torch
from torch.utils.data import IterableDataset

class GraphIterableDataset(IterableDataset):
    def __init__(self, paths, shuffle, batch_size, data_transformer):
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
        """
        super(GraphIterableDataset).__init__()
        self.dataset = self.load_data(paths)

        if shuffle:
            self.shuffle()
        
        if  is not None:
            self.transform(data_transformer)
        
        if batch_size > 1:
            self.batch(batch_size)
    
    def __getitem__(self, index):
        pass

    def __iter__(self):
        """
        If the dataset has labels, then the iterator will give a batch of graphs 
        and its corresponding labels. Otherwise, it will only give a batch of graphs.
        
        """
        graphs = self.dataset[0]
        labels = self.dataset[1]

        if labels is not None:
            return zip(graphs, labels)
        else:
            return iter(graphs)

    def load_data(self, paths: List[str]):
        """
        Load data into the memory. If number of path(s) is greater than one, 
        than data from thoses sources will be merged together. 

        Parameters
        ----------
        paths: List[str]
            A list of strings that point to the location of the data to load.

        Returns
        -------
        merged_dataset:
            A list of dgl graphs with labels (if provided).
        """

        merged_dataset = ([],{'labels':None})

        graphs_to_merge = []
        labels_to_merge = []

        for path in paths:

            shards = glob.glob(path)

            for i in range(len(shards)):

                graphs = dgl.load_graphs(path + '/shard{}'.format(i+1))

                graphs_to_merge.extend(graphs[0])
                labels_to_merge.extend(graphs[1]['labels'])
                print('labels', graphs[1]['labels'].shape)

        merged_dataset[0].extend(graphs_to_merge)
        merged_dataset[1]['labels'] = torch.stack(labels_to_merge, 0)
            
        return merged_dataset
    
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
        for i in range(0, len(dataset[0]), batch_size):
            batched_features.append(dgl.batch(dataset[0][i:i+batch_size]))
        
        all_labels = dataset[1]['labels']
        batched_labels = []
        for i in range(0, len(all_labels), batch_size):
            batched_labels.append(all_labels[i:i+batch_size])
        return (batched_features, batched_labels)

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
        
