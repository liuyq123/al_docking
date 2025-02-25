import argparse
import os
import deepchem as dc
import dgl
import pickle 
import torch 
import numpy as np

from utils import load_data

def convert(file_path, output_path, batch_size, shard_size):

    dataset = load_data(file_path)

    os.makedirs(output_path, exist_ok=True) 

    if batch_size > 1:
        
        batch_counter = 0
        shard_counter = 0
        shard_idx = 1
        
        graphs = []
        batched_graphs = []

        for x, y, w, id in dataset.itersamples():

            graphs.append(x.to_dgl_graph(self_loop=True))

            batch_counter += 1
            shard_counter += 1

            if batch_counter >= batch_size:

                batch_counter = 0
                batched_graphs.append(dgl.batch(graphs))
                graphs = []

            if shard_counter >= shard_size:

                shard_counter = 0

                with open (output_path + '/shard{}'.format(shard_idx), 'wb') as file:
                    pickle.dump((batched_graphs, ), file)
                
                batched_graphs = []

                shard_idx += 1

        with open (output_path + '/shard{}'.format(shard_idx), 'wb') as file:
            batched_graphs.append(dgl.batch(graphs))
            pickle.dump((batched_graphs, ), file)

    elif batch_size == 1:

        graphs = []

        if shard_size != -1:

            shard_counter = 0
            shard_idx = 1

            for x, y, w, id in dataset.itersamples():

                graphs.append(x.to_dgl_graph(self_loop=True))
                shard_counter += 1

                if shard_counter >= shard_size:

                    shard_counter = 0

                    with open (output_path + '/shard{}'.format(shard_idx), 'wb') as file:
                        pickle.dump((graphs, ), file)
                    
                    shard_idx += 1
            with open (output_path + '/shard{}'.format(shard_idx), 'wb') as file:
                        pickle.dump((graphs, ), file)

        else:
            labels = []

            for x, y, w, id in dataset.itersamples():

                graphs.append(x.to_dgl_graph(self_loop=True))
                labels.append(y)

            dgl.save_graphs(output_path + '/shard1', graphs, {"labels":torch.FloatTensor(np.array(labels))})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path")
    parser.add_argument("--output_path")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--shard_size", type=int)
    
    args = parser.parse_args()

    convert(args.file_path, args.output_path, args.batch_size, args.shard_size)

if __name__ == "__main__":
    main()
        