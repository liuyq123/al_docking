import deepchem as dc
from deepchem.data import DiskDataset

import yaml

def load_data(datasets: str) -> DiskDataset:

    if isinstance(datasets, list):
        dataset = merge_dataset(datasets)
    else:
        dataset = DiskDataset(datasets)

    return dataset

def merge_dataset(datasets) -> DiskDataset:

        datasets_list = []

        for dataset in datasets:

            disk_dataset = DiskDataset(dataset)
            datasets_list.append(disk_dataset)

        merged_dataset = DiskDataset.merge(datasets_list)

        return merged_dataset

def yaml_parser(config_file) -> dict:
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config