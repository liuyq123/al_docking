import argparse
import os

import dgl
import glob
import pickle
import numpy as np
import polars as pl
import pyarrow as pa
import torch

from utils.utils import yaml_parser
from src.model_creator import ModelCreator

def predict(config: dict, 
            featurized_data_path: str, 
            raw_data_path: str,
            id_field: str,
            batch_size: int,
            output_dir: str,
            output_name: str) -> None:
    """
    Make predictions with the trained model.

    Parameters
    ----------
    config (dict): A dictionary contains the parameters need to restore the model. 
        Can be loaded from the same yaml file used for training.
    featurized_data_path (str): Path to the featurized data.
    raw_data_path (str): Path to the corresponding csv file.
    id_field (str): The name of the id column.
    batch_size (int): The batch size used for prediciton. Set batch size to 1 when it is already batched.
    output_dir (str): The output directory for the predictions. 
    output_name (str): The name of the output file.

    Returns
    -------
    None
    """
    if torch.cuda.is_available():
       device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = ModelCreator(config['model']).get_model()
    checkpoint = torch.load(config['model']['best_ckpt'] + '/best.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model = model.to(device)
    model.eval()
    
    n_shards = len(glob.glob(featurized_data_path + '/*'))
    pred_arrs = []
    for i in range(n_shards):
        with open (featurized_data_path + '/shard{}'.format(i+1), 'rb') as f:
            dataset = pickle.load(f)
            dataset = dataset[0]
            
        output = []
        for i in range(0, len(dataset)):
            inputs = dataset[i].to(device)
            output_values = model(inputs)
            output_values = output_values.detach().cpu().numpy()
            output.append(output_values)
            
        shard_prediction = np.concatenate(output, axis=0)
        pred_arrs.extend(shard_prediction)
    predictions = np.concatenate(pred_arrs, axis=0)

    if raw_data_path[-7:] == 'parquet':
        df = pl.scan_parquet(raw_data_path)
    else:
        df = pl.scan_csv(raw_data_path)
    output = df.with_columns(pl.Series(name="preds", values=predictions)).collect(streaming=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output.write_parquet(output_dir + '/' + output_name + '.parquet')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--featurized_data_path")
    parser.add_argument("--raw_data_path")
    parser.add_argument("--id_field")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--output_dir")
    parser.add_argument("--output_name")
    
    args = parser.parse_args()

    config = yaml_parser(args.config)
    
    predict(config, args.featurized_data_path, args.raw_data_path, args.id_field, args.batch_size, args.output_dir, args.output_name)

if __name__ == "__main__":
    main()