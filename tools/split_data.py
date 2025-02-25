import argparse
import os

from typing import Optional

import numpy as np
import pandas as pd

def split_data(file_path: str, 
               output_dir: str,
               num_splits: Optional[int] = None, 
               frac_splits: Optional[float] = None) -> None:
    """
    Split the data to `num_splits` files.

    Parameters
    ----------
    file_path (str): The path of the csv file to split.
    num_splits (int, optional): Split the file into `num_splits` files of same size.
    frac_splits (float, optional): Split the file into two files. One is `frac_splits` of the orginal file,
        and the other is of 1 - `frac_splits`.
    output_dir (str): The output directory for the splitted csv files.

    Returns
    -------
    None
    """
    df = pd.read_csv(file_path)

    np.random.seed(42)

    n_rows = df.shape[0]
    arr = np.linspace(0, n_rows-1, num=n_rows, dtype=int)
    np.random.shuffle(arr)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if frac_splits:
        num = n_rows * frac_splits
        split1 = df.iloc[arr[:num]]
        split2 = df.iloc[arr[num:]]
        split1.to_csv(output_dir + '/fold{}.csv'.format(1), index=False)
        split2.to_csv(output_dir + '/fold{}.csv'.format(2), index=False)
    else: 
        splits = np.array_split(arr, num_splits)
        i = 1
        for split in splits:
            df.iloc[split.tolist()].to_csv(output_dir + '/fold{}.csv'.format(i), index=False)
            i += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path")
    parser.add_argument("--num_splits", type=int, nargs='?', default=None)
    parser.add_argument("--frac_splits", type=int, nargs='?', default=None)
    parser.add_argument("--output_dir")

    args = parser.parse_args()

    split_data(args.file_path, args.output_dir, args.num_splits, args.frac_splits)

if __name__ == "__main__":
    main()