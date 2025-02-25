import argparse
import csv
import numpy as np
import os 
import pandas as pd
import glob
from pyspark.sql import SparkSession
import pyspark.sql.functions as F 
from pyspark.sql import DataFrame
from typing import List

def computing_recall(predictions_path: str, 
                     preds_field: str,
                     score_field: str, 
                     id_field: str, 
                     field: str, 
                     pct_list: List[float], 
                     top_n: int, 
                     top_n_path: str,
                     ground_truth_path: str, 
                     mode='percentile') -> List[int]:
    """
    Compute prediction recall.

    Parameters
    ----------
    preds (Dataframe): Prediction dataframe, which should have a column for id and a columns for predictions.
    score_field (str): The name of the score column.
    id_field (str): The name of the id column.
    field (str): The name of column used for computing recall.
    pct_list (List[float]): The percentage cutoffs of top ML predictions used for evaluation. 
    top_n (int): The number of real tops of interest. 
    ground_truth (Dataframe): Ground truth dataframe, which should have a column for id and a column for docking scores.
    mode ({'sorting', 'percentile'}, default 'percentile'): Sorting mode obtaining top ML predictions by sorting the whole dataset, 
        while percentile mode computes an approximated percentile and then selects data points within that range. 

    Returns
    -------
    List[int]
        The recalls of different percentage cutoffs as provided by pct_list.
    """

    spark = SparkSession.builder \
    .master('local[*]') \
    .config("spark.driver.memory", '64g') \
    .config("spark.executor.memory", '32g') \
    .config("spark.driver.maxResultSize", '16g') \
    .config("spark.worker.cleanup.enabled", "true") \
    .appName('recall') \
    .getOrCreate()

    preds = spark.read.parquet(predictions_path)
    preds = preds.withColumn(preds_field, preds[preds_field].cast("float"))

    if ground_truth_path[-7:] == 'parquet' or len(glob.glob(ground_truth_path + "/*.parquet")) != 0:
        ground_truth = spark.read.parquet(ground_truth_path)
    else: 
        ground_truth = spark.read.option("header", True).csv(ground_truth_path)

    ground_truth = ground_truth.dropDuplicates([id_field])
    ground_truth = ground_truth.withColumn(score_field, ground_truth[score_field].cast("float"))
    
    if os.path.exists(top_n_path):
        true_top_n = spark.read.option("header", True).parquet(top_n_path)
    else:
        true_top_n = ground_truth.orderBy(ground_truth[score_field].asc()).limit(top_n)
        true_top_n = true_top_n.select(id_field, score_field)

        true_top_n.write.option("header", True).parquet(top_n_path)

    recalls = []

    if mode == 'sorting':
        df_sorted = preds.orderBy(preds[field].asc())

    for pct in pct_list:

        if mode == 'sorting':
            cutoff = round(preds.count() * pct)
            top_preds = df_sorted.limit(cutoff)
        
        else: 
            threshold = preds.select(
                F.percentile_approx(field, pct, 5000).alias("quantiles"))

            top_preds = preds.filter(preds[field] < threshold.head()[0])
    
        top_preds = top_preds.select(id_field, field)

        overlap = true_top_n.join(top_preds, [id_field], "left")
        recall = overlap.where(F.col(field).isNotNull()).count()
        
        recalls.append(recall)
    
    spark.stop()

    return recalls


def csv_writer(lines: List[tuple],
               output_path: str, 
               header: tuple = None):
    """
    Write lines to a csv file.

    Parameters
    ----------
    lines (List[tuple]): The lines to be written.
    output_path (str): The path to the output file. If it doesn't exist, a new file will be created. 
        Otherwise new lines will be added to the existing file.
    header (tuple, optional): The header of the csv file.

    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'a+') as out_file:
        writer = csv.writer(out_file)
        if header:
            writer.writerow(header)
        
        for line in lines:
            writer.writerow(line)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--predictions_path", help="Path to the prediction parquet file(s).")
    parser.add_argument("--id_field")
    parser.add_argument("--score_field")
    parser.add_argument("--preds_field", help="The field used for computing recall.")
    parser.add_argument("--ground_truth_path", help="Path to the ground truth csv file(s).")
    parser.add_argument("--pct_list", nargs='+', type=float)
    parser.add_argument("--top_n", type=int)
    parser.add_argument("--top_n_path")
    parser.add_argument("--experiment_name",
                        help="The name of the experiment, which will be in the first column of the output csv file.")
    parser.add_argument("--output")

    args = parser.parse_args()

    recalls = computing_recall(args.predictions_path, args.preds_field, args.score_field, args.id_field, args.preds_field, args.pct_list, args.top_n, args.top_n_path, args.ground_truth_path)


    lines = [tuple([args.experiment_name] + recalls)]

    if not os.path.isfile(args.output):
        header = tuple(['experiment_name'] + args.pct_list)
        csv_writer(lines, args.output, header=header)
    else:
        csv_writer(lines, args.output)
    
if __name__ == "__main__":
    main()