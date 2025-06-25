#!/usr/bin/env python3
"""
DataFrame Sampler Tool

This tool reads parquet files from S3 buckets, samples distinct IDs,
and creates before/after datasets for comparison testing.
"""

import sys
import logging
import pandas as pd
import dask.dataframe as dd
import numpy as np
from pathlib import Path
from typing import List
import warnings
import time

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class DataFrameSampler:
    """
    A tool for sampling data from S3 parquet files and creating before/after datasets.
    """

    def __init__(self, log_file: str = None):
        """
        Initialize the sampler tool with logging configuration.

        Args:
            log_file (str): Path to log file. If None, uses default filename.
        """
        self.setup_logging(log_file)
        self.logger = logging.getLogger(__name__)

    def setup_logging(self, log_file: str = None):
        """
        Set up logging to both file and console with timestamps.

        Args:
            log_file (str): Path to log file
        """
        if log_file is None:
            log_file = "df_sampler.log"

        # Create logs directory if it doesn't exist
        log_dir = Path(log_file).parent
        log_dir.mkdir(exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(log_file, mode="w"),
                logging.StreamHandler(sys.stdout),
            ],
        )

        print(f"Logging initialized. Log file: {log_file}")

    def read_parquet_from_s3(self, s3_path: str, dataset_name: str) -> pd.DataFrame:
        """
        Read parquet files from S3 using Dask and convert to pandas DataFrame.

        Args:
            s3_path (str): S3 path containing parquet files
            dataset_name (str): Name for logging purposes

        Returns:
            pd.DataFrame: Combined dataframe from all parquet files
        """
        self.logger.info(f"Starting to read {dataset_name} data from S3: {s3_path}")

        try:
            # Read parquet files using Dask
            self.logger.info(f"Reading parquet files with Dask from S3: {s3_path}")
            dask_df = dd.read_parquet(s3_path)

            # Log basic info about the Dask dataframe
            self.logger.info(f"{dataset_name} - Dask DataFrame info:")
            self.logger.info(f"  - Columns: {list(dask_df.columns)}")
            self.logger.info(f"  - Number of partitions: {dask_df.npartitions}")

            # Convert to pandas DataFrame
            self.logger.info(f"Converting {dataset_name} Dask DataFrame to pandas...")
            pandas_df = dask_df.compute()

            # Convert all column names to lowercase
            original_columns = list(pandas_df.columns)
            pandas_df.columns = pandas_df.columns.str.lower()
            lowercase_columns = list(pandas_df.columns)

            # Log column name changes if any occurred
            if original_columns != lowercase_columns:
                self.logger.info(
                    f"Converted column names to lowercase for {dataset_name}"
                )
                for orig, lower in zip(original_columns, lowercase_columns):
                    if orig != lower:
                        self.logger.info(f"  - '{orig}' -> '{lower}'")

            # Log pandas DataFrame info
            self.logger.info(f"{dataset_name} - Pandas DataFrame info:")
            self.logger.info(f"  - Shape: {pandas_df.shape}")
            self.logger.info(
                f"  - Memory usage: {pandas_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            )
            self.logger.info(f"  - Data types:\n{pandas_df.dtypes}")

            return pandas_df

        except Exception as e:
            self.logger.error(
                f"Error reading {dataset_name} data from {s3_path}: {str(e)}"
            )
            raise

    def get_distinct_ids(
        self, df: pd.DataFrame, sample_size: int, id_column: str = "id"
    ) -> List:
        """
        Get a random sample of distinct IDs from the dataframe.

        Args:
            df (pd.DataFrame): Source dataframe
            sample_size (int): Number of distinct IDs to sample
            id_column (str): Name of the ID column (default: 'id')

        Returns:
            List: List of sampled distinct IDs
        """
        self.logger.info(
            f"Sampling {sample_size} distinct IDs from column '{id_column}'"
        )

        if id_column not in df.columns:
            raise ValueError(
                f"ID column '{id_column}' not found in dataframe. Available columns: {list(df.columns)}"
            )

        # Get distinct IDs
        distinct_ids = df[id_column].unique()
        total_distinct = len(distinct_ids)

        self.logger.info(f"Found {total_distinct} distinct IDs in the dataset")

        if sample_size > total_distinct:
            self.logger.warning(
                f"Requested sample size ({sample_size}) is larger than available distinct IDs ({total_distinct})"
            )
            self.logger.warning(f"Using all available distinct IDs ({total_distinct})")
            sample_size = total_distinct

        # Randomly sample IDs
        np.random.seed(42)  # For reproducible results
        sampled_ids = np.random.choice(distinct_ids, size=sample_size, replace=False)

        self.logger.info(f"Successfully sampled {len(sampled_ids)} distinct IDs")
        self.logger.info(f"Sample ID range: {min(sampled_ids)} to {max(sampled_ids)}")

        return sampled_ids.tolist()

    def filter_dataframe_by_ids(
        self, df: pd.DataFrame, ids: List, id_column: str = "id"
    ) -> pd.DataFrame:
        """
        Filter dataframe to include only rows with specified IDs.

        Args:
            df (pd.DataFrame): Source dataframe
            ids (List): List of IDs to filter by
            id_column (str): Name of the ID column (default: 'id')

        Returns:
            pd.DataFrame: Filtered dataframe
        """
        self.logger.info(f"Filtering dataframe by {len(ids)} IDs")

        if id_column not in df.columns:
            raise ValueError(
                f"ID column '{id_column}' not found in dataframe. Available columns: {list(df.columns)}"
            )

        # Filter dataframe
        filtered_df = df[df[id_column].isin(ids)].copy()

        self.logger.info(f"Filtered dataframe shape: {filtered_df.shape}")
        self.logger.info(f"Original dataframe shape: {df.shape}")

        return filtered_df

    def save_parquet(self, df: pd.DataFrame, output_path: str, dataset_name: str):
        """
        Save dataframe as parquet file.

        Args:
            df (pd.DataFrame): Dataframe to save
            output_path (str): Output file path
            dataset_name (str): Name for logging purposes
        """
        self.logger.info(f"Saving {dataset_name} to: {output_path}")

        try:
            # Create output directory if it doesn't exist
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save as parquet
            df.to_parquet(output_path, index=False, engine="pyarrow")

            self.logger.info(f"Successfully saved {dataset_name}")
            self.logger.info(f"  - File: {output_path}")
            self.logger.info(f"  - Shape: {df.shape}")
            self.logger.info(
                f"  - File size: {Path(output_path).stat().st_size / 1024**2:.2f} MB"
            )

        except Exception as e:
            self.logger.error(f"Error saving {dataset_name} to {output_path}: {str(e)}")
            raise

    def create_sample_datasets(
        self,
        original_s3_path: str,
        new_s3_path: str,
        sample_size: int,
        id_column: str = "id",
        before_output: str = "before_df.parquet",
        after_output: str = "after_df.parquet",
    ):
        """
        Main method to create sample datasets from S3 parquet files.

        Args:
            original_s3_path (str): S3 path for original dataset
            new_s3_path (str): S3 path for new dataset
            sample_size (int): Number of distinct IDs to sample
            id_column (str): Name of the ID column (default: 'id')
            before_output (str): Output path for before dataset
            after_output (str): Output path for after dataset
        """
        start_time = time.time()
        self.logger.info("Starting dataset sampling process")
        self.logger.info(f"Original dataset S3 path: {original_s3_path}")
        self.logger.info(f"New dataset S3 path: {new_s3_path}")
        self.logger.info(f"Sample size: {sample_size}")
        self.logger.info(f"ID column: {id_column}")

        try:
            # Step 1: Read original dataset from S3
            self.logger.info("=" * 50)
            self.logger.info("STEP 1: Reading original dataset from S3")
            self.logger.info("=" * 50)
            original_df = self.read_parquet_from_s3(
                original_s3_path, "original_dataset"
            )

            # Step 2: Sample distinct IDs
            self.logger.info("=" * 50)
            self.logger.info("STEP 2: Sampling distinct IDs")
            self.logger.info("=" * 50)
            sampled_ids = self.get_distinct_ids(original_df, sample_size, id_column)

            # Step 3: Create before dataset (filtered original)
            self.logger.info("=" * 50)
            self.logger.info("STEP 3: Creating before dataset")
            self.logger.info("=" * 50)
            before_df = self.filter_dataframe_by_ids(
                original_df, sampled_ids, id_column
            )
            self.save_parquet(before_df, before_output, "before_dataset")

            # Step 4: Read new dataset from S3
            self.logger.info("=" * 50)
            self.logger.info("STEP 4: Reading new dataset from S3")
            self.logger.info("=" * 50)
            new_df = self.read_parquet_from_s3(new_s3_path, "new_dataset")

            # Step 5: Create after dataset (filtered new dataset with same IDs)
            self.logger.info("=" * 50)
            self.logger.info("STEP 5: Creating after dataset")
            self.logger.info("=" * 50)
            after_df = self.filter_dataframe_by_ids(new_df, sampled_ids, id_column)
            self.save_parquet(after_df, after_output, "after_dataset")

            # Summary
            total_time = time.time() - start_time
            self.logger.info("=" * 50)
            self.logger.info("SAMPLING COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 50)
            self.logger.info(f"Total processing time: {total_time:.2f} seconds")
            self.logger.info(f"Sampled {len(sampled_ids)} distinct IDs")
            self.logger.info(f"Before dataset saved to: {before_output}")
            self.logger.info(f"After dataset saved to: {after_output}")
            self.logger.info(f"Before dataset shape: {before_df.shape}")
            self.logger.info(f"After dataset shape: {after_df.shape}")

        except Exception as e:
            self.logger.error(f"Error during sampling process: {str(e)}")
            raise


# Configuration - modify these for your specific use case
ORIGINAL_S3_PATH = "s3://your-bucket/original-data/"
NEW_S3_PATH = "s3://your-bucket/new-data/"
SAMPLE_SIZE = 1000
ID_COLUMN = (
    "id"  # Change this to your ID column name: customer_id, car_id, user_id, etc.
)
BEFORE_OUTPUT = "before_df.parquet"
AFTER_OUTPUT = "after_df.parquet"
LOG_FILE = None  # Will use default: df_sampler.log


def main():
    """
    Main function to run the DataFrame sampler tool.
    """

    # Create sampler tool
    sampler = DataFrameSampler(log_file=LOG_FILE)

    # Display the configuration
    print(f"DataFrame Sampler Configuration:")
    print(f"  Original S3 Path: {ORIGINAL_S3_PATH}")
    print(f"  New S3 Path: {NEW_S3_PATH}")
    print(f"  Sample Size: {SAMPLE_SIZE}")
    print(f"  ID Column: {ID_COLUMN}")
    print(f"  Before Output: {BEFORE_OUTPUT}")
    print(f"  After Output: {AFTER_OUTPUT}")
    print()

    try:
        # Run sampling
        sampler.create_sample_datasets(
            original_s3_path=ORIGINAL_S3_PATH,
            new_s3_path=NEW_S3_PATH,
            sample_size=SAMPLE_SIZE,
            id_column=ID_COLUMN,
            before_output=BEFORE_OUTPUT,
            after_output=AFTER_OUTPUT,
        )

        print("Sampling completed successfully!")
        sys.exit(0)

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
