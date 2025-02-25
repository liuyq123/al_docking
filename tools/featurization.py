import argparse
import deepchem as dc
from typing import List

def featurization(file_path: str, 
                  output_dir: str, 
                  id_field: str, 
                  tasks: List[str],
                  use_chirality: bool,
                  use_edges: bool):
    """
    Featurize smiles strings. 
    Please refer to https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html#molgraphconvfeaturizer for more details.

    Parameters
    ----------
    file_path (str): Path to the csv file. 
    output_dir (str): Path to the output directory. 
    id_field (str): The name of the id column. 
    tasks (List(str)): The name(s) of the task column(s).
    use_edges (bool): Whether or not edge features will be generated.

    Returns
    -------
    None
    """
    loader = dc.data.CSVLoader(tasks=tasks,  
                           id_field=id_field, 
                           feature_field="smiles", 
                           featurizer=dc.feat.MolGraphConvFeaturizer(use_chirality=use_chirality, use_edges=use_edges))
    dataset = loader.create_dataset(inputs=file_path,
                                    data_dir=output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path")
    parser.add_argument("--output_dir")
    parser.add_argument("--id_field", default="zincid")
    parser.add_argument("--tasks", nargs='+', default=[])
    parser.add_argument("--use_chirality", action="store_true", default=False)
    parser.add_argument("--use_edges", action="store_true", default=False)

    args = parser.parse_args()

    featurization(args.file_path, args.output_dir, args.id_field, args.tasks, args.use_chirality, args.use_edges)

if __name__ == "__main__":
    main()