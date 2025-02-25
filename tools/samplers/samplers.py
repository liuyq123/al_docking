import glob
import os

from typing import Optional

from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

class Sampler(object):
    """
    Abstract base class for samplers
    """

    def __init__(self,
                 score_field: Optional[str] = None) -> None:
        """

        Parameters
        ----------
        score_field (float): The name of the docking score column. 

        Returns
        -------
        None
        """

        self.spark = None
        self.score_field = score_field
    
    def create_spark_session(self, 
                             spark_config: dict) -> None:
        self.spark = SparkSession.builder \
        .master('local[*]') \
        .config('spark.driver.memory', spark_config['spark.driver.memory']) \
        .config('spark.executor.memory', spark_config['spark.executor.memory']) \
        .config('spark.driver.maxResultSize', spark_config['spark.driver.maxResultSize']) \
        .config('spark.worker.cleanup.enabled', "true") \
        .appName('sample data') \
        .getOrCreate()
        
    
    def create_dataframe(self, 
                         file_path: str,
                         prev_samples_path: Optional[str] = None,
                         id_field: Optional[str] = None,
                         sampling_cutoff: Optional[float] = None) -> DataFrame:
        """
        Create a dataframe for the data to be sampled. 

        Parameters
        ----------
        file_path (str): Path to the csv/parquet file(s). It can either be a single file, or a directory. 
            If it's a directory, then all the files in that directory will be read. 
        prev_samples_path (str, optional, default None): Path to previous samples. Provide it to exclude 
            those samples from the current sampling process.
        id_field (str, optional, default None): The name of the column used as id. Need to provide it 
            when prev_samples_path is not None.
        sampling_cutoff (float, optional, default None): Only sample data points with a score smaller than this value.

        Returns
        -------
        DataFrame
            
        """
        if file_path[-7:] == 'parquet' or len(glob.glob(file_path + "/*.parquet")) != 0:
            data = self.spark.read.parquet(file_path)
        else: 
            data = self.spark.read.option("header", True).csv(file_path)
        
        if prev_samples_path is not None:
            prev_samples = self.spark.read.option("header", True).csv(prev_samples_path)
            prev_samples = prev_samples[[id_field]]
            data = data.join(prev_samples, [id_field], "left_outer") \
                       .where(prev_samples[id_field].isNull()) \
                       .select([F.col(c) for c in data.columns])

        if self.score_field:
            data = data.withColumn(self.score_field, data[self.score_field].cast("float"))

        if sampling_cutoff is not None:
            data = data.filter(data[self.score_field] < sampling_cutoff)
        
        return data

    def sample(self, 
               data: DataFrame,
               sort_top: bool,
               num_samples: Optional[int] = None, 
               frac_samples: Optional[float] = None) -> DataFrame:
        """
        Sample data based on some stratigies.

        Parameters
        ----------
        data (DataFrame): The dataset to sample.
        num_samples (int, optional, default None): The number of rows to sample.  
        frac_samples (float, optional, default None): The fraction of rows to sample. 
            When both `num_samples` and `frac_samples` are provided, `num_samples` will be used.

        Returns
        -------
        DataFrame
            Samples.
        """
        pass

    def save_samples(self,
                     samples: DataFrame, 
                     output_path: str) -> None:
        """
        Write samples to a csv file.

        Parameters
        ----------
        samples (DataFrame): The samples dataframe to write.
        output_dir (str): Path to the output directory. 

        Returns
        -------
        None
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        samples.toPandas().to_csv(output_path, index=False)

        # samples.write.option("header", True).parquet(output_path)
    
    def close_spark_session(self) -> None:
        if self.spark is not None:
            self.spark.stop()


class GreedySampler(Sampler):
    def __init__(self, 
                 score_field: str) -> None:
        super().__init__(score_field)
    
    def create_spark_session(self, 
                             spark_config: dict) -> None:
        return super().create_spark_session(spark_config)
    
    def create_dataframe(self, 
                         file_path: str, 
                         prev_samples_path: Optional[str] = None, 
                         id_field: Optional[str] = None, 
                         sampling_cutoff: Optional[float] = None) -> DataFrame:
        return super().create_dataframe(file_path, prev_samples_path, id_field, sampling_cutoff)
    
    def sample(self, 
               data: DataFrame,
               sort_top: Optional[bool] = False,
               num_samples: Optional[int] = None, 
               frac_samples: Optional[float] = None) -> DataFrame:
        """
        Sample the data points with the lowest score.
        """
        if sort_top:
            top_data = data.filter(data[self.score_field] < -8)
            data_sorted = top_data.orderBy(top_data[self.score_field].asc())
        else:
            data_sorted = data.orderBy(data[self.score_field].asc())

        if num_samples:
            samples = data_sorted.limit(num_samples)
        else:
            num_samples = round(data.count() * frac_samples)
            samples = data_sorted.limit(num_samples)
        
        return samples


class RandomSampler(Sampler):
    def __init__(self, 
                 score_field: str) -> None:
        super().__init__(score_field)
    
    def sample(self, 
               data: DataFrame,
               num_samples: Optional[int] = None, 
               frac_samples: Optional[float] = None) -> DataFrame:
        """
        Sample data points randomly.
        """
        if num_samples:
            samples = data.sample(fraction=num_samples/data.count())
        else:
            samples = data.sample(fraction=frac_samples)
        
        return samples


class YangSampler(Sampler):
    """
    Implement the sampling strategy proposed by this paper: https://pubs.acs.org/doi/10.1021/acs.jctc.1c00810,
    which is choosing the most uncertain 0.1% from the top 5%. The implementation has some slight differeces to 
    the original one. Instead of the most uncetain 0.1%, the most uncertain `num_samples` or `frac_samples` 
    data points are chosen from the top 5%.
    """
    def __init__(self, 
                 score_field: str) -> None:
        super().__init__(score_field)
    
    def sample(self, 
               data: DataFrame,
               uncertainty_field: str, 
               num_samples: Optional[int] = None, 
               frac_samples: Optional[float] = None) -> DataFrame:

        score_cutoff = data.select(
                F.percentile_approx(self.score_field, 0.05, 5000).alias("score_cutoff"))
        top_score = data.filter(data[self.score_field] < score_cutoff.head()["score_cutoff"])

        sorted_top_score = top_score.orderBy(top_score[uncertainty_field].desc())

        if num_samples:
            samples = sorted_top_score.limit(num_samples)
        else:
            samples = sorted_top_score.limit(round(frac_samples * data.count()))
        
        return samples