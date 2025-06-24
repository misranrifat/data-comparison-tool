#!/usr/bin/env python3
"""
Comprehensive Data Comparison Tool

This tool compares two datasets stored in S3 as parquet files.
It uses Dask for efficient reading and pandas for comparison operations.
"""

import sys
import logging
import pandas as pd
import numpy as np
import dask.dataframe as dd
from pathlib import Path
import argparse
from typing import Tuple, List
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class DataComparisonTool:
    """
    A comprehensive tool for comparing two datasets from S3 parquet files.
    """

    def __init__(self, log_file: str = None):
        """
        Initialize the comparison tool with logging configuration.

        Args:
            log_file (str): Path to log file. If None, uses timestamp-based filename.
        """
        self.setup_logging(log_file)
        self.logger = logging.getLogger(__name__)
        self.differences = []

    def setup_logging(self, log_file: str = None):
        """
        Set up logging to both file and console with timestamps.

        Args:
            log_file (str): Path to log file
        """
        if log_file is None:
            log_file = "data_comparison.log"

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

    def detect_file_type_and_location(self, path: str) -> Tuple[str, str]:
        """
        Detect if path is local/S3 and file type (CSV/Parquet).

        Args:
            path (str): File or directory path

        Returns:
            Tuple[str, str]: (location_type, file_type) where:
                location_type: 'local' or 's3'
                file_type: 'csv' or 'parquet'
        """
        # Determine location
        if path.startswith("s3://"):
            location_type = "s3"
        else:
            location_type = "local"

        # Determine file type from path or extension
        path_lower = path.lower()
        if path_lower.endswith(".csv") or "/csv/" in path_lower or "csv" in path_lower:
            file_type = "csv"
        elif (
            path_lower.endswith(".parquet")
            or "/parquet/" in path_lower
            or "parquet" in path_lower
        ):
            file_type = "parquet"
        else:
            # Default to parquet if unclear
            file_type = "parquet"
            self.logger.warning(
                f"Could not determine file type from path '{path}', defaulting to parquet"
            )

        self.logger.info(f"Detected: {location_type} {file_type} for path: {path}")
        return location_type, file_type

    def read_data(self, path: str, dataset_name: str) -> pd.DataFrame:
        """
        Read CSV or Parquet files from local filesystem or S3 using Dask and convert to pandas DataFrame.

        Args:
            path (str): Path containing data files (local or S3)
            dataset_name (str): Name for logging purposes

        Returns:
            pd.DataFrame: Combined dataframe from all files
        """
        self.logger.info(f"Starting to read {dataset_name} data from: {path}")

        try:
            # Detect file type and location
            location_type, file_type = self.detect_file_type_and_location(path)

            # Read files using Dask based on type and location
            self.logger.info(
                f"Reading {file_type} files with Dask from {location_type}: {path}"
            )

            if file_type == "parquet":
                dask_df = dd.read_parquet(path)
            elif file_type == "csv":
                # For CSV files, we need to handle potential schema differences
                # Read with assume_missing=True to handle varying schemas
                dask_df = dd.read_csv(path, assume_missing=True)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

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
                        self.logger.info(f"  - '{orig}' → '{lower}'")

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
                f"Error reading {dataset_name} data from {path}: {str(e)}"
            )
            raise

    def get_common_columns(self, df1: pd.DataFrame, df2: pd.DataFrame) -> List[str]:
        """
        Get intersection of columns between two dataframes.

        Args:
            df1 (pd.DataFrame): First dataframe
            df2 (pd.DataFrame): Second dataframe

        Returns:
            List[str]: List of common column names
        """
        columns1 = set(df1.columns)
        columns2 = set(df2.columns)

        common_columns = list(columns1.intersection(columns2))
        only_in_df1 = list(columns1 - columns2)
        only_in_df2 = list(columns2 - columns1)

        self.logger.info(f"Column comparison:")
        self.logger.info(f"  - Total columns in before_df: {len(columns1)}")
        self.logger.info(f"  - Total columns in after_df: {len(columns2)}")
        self.logger.info(f"  - Common columns: {len(common_columns)}")
        self.logger.info(f"  - Columns only in before_df: {len(only_in_df1)}")
        self.logger.info(f"  - Columns only in after_df: {len(only_in_df2)}")

        if only_in_df1:
            self.logger.warning(f"Columns only in before_df: {only_in_df1}")
            self.differences.append(
                {
                    "type": "column_difference",
                    "description": "Columns only in before_df",
                    "details": only_in_df1,
                }
            )

        if only_in_df2:
            self.logger.warning(f"Columns only in after_df: {only_in_df2}")
            self.differences.append(
                {
                    "type": "column_difference",
                    "description": "Columns only in after_df",
                    "details": only_in_df2,
                }
            )

        return common_columns

    def identify_column_types(
        self, df: pd.DataFrame, columns: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Identify numerical and non-numerical columns.

        Args:
            df (pd.DataFrame): DataFrame to analyze
            columns (List[str]): List of columns to analyze

        Returns:
            Tuple[List[str], List[str]]: (numerical_columns, non_numerical_columns)
        """
        numerical_columns = []
        non_numerical_columns = []

        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numerical_columns.append(col)
            else:
                non_numerical_columns.append(col)

        self.logger.info(f"Column type analysis:")
        self.logger.info(
            f"  - Numerical columns ({len(numerical_columns)}): {numerical_columns}"
        )
        self.logger.info(
            f"  - Non-numerical columns ({len(non_numerical_columns)}): {non_numerical_columns}"
        )

        return numerical_columns, non_numerical_columns

    def compare_shapes(self, before_df: pd.DataFrame, after_df: pd.DataFrame):
        """
        Compare the shapes of both dataframes.

        Args:
            before_df (pd.DataFrame): Before dataframe
            after_df (pd.DataFrame): After dataframe
        """
        self.logger.info(f"Shape comparison:")
        self.logger.info(f"  - before_df shape: {before_df.shape}")
        self.logger.info(f"  - after_df shape: {after_df.shape}")

        if before_df.shape != after_df.shape:
            self.logger.warning("DataFrames have different shapes!")
            self.differences.append(
                {
                    "type": "shape_difference",
                    "description": "DataFrames have different shapes",
                    "before_shape": before_df.shape,
                    "after_shape": after_df.shape,
                }
            )
        else:
            self.logger.info("✓ DataFrames have identical shapes")

    def compare_numerical_columns(
        self,
        before_df: pd.DataFrame,
        after_df: pd.DataFrame,
        numerical_columns: List[str],
        tolerance: float = 0.01,
    ) -> pd.DataFrame:
        """
        Compare numerical columns with specified tolerance.

        Args:
            before_df (pd.DataFrame): Before dataframe
            after_df (pd.DataFrame): After dataframe
            numerical_columns (List[str]): List of numerical columns
            tolerance (float): Tolerance for numerical comparison

        Returns:
            pd.DataFrame: DataFrame containing differences
        """
        self.logger.info(f"Comparing numerical columns with tolerance: {tolerance}")

        numerical_differences = []

        for col in numerical_columns:
            self.logger.info(f"Comparing numerical column: {col}")

            # Convert to float for consistent comparison (handles int, bool, etc.)
            original_type = before_df[col].dtype
            try:
                before_col = before_df[col].astype(float).fillna(0)
                after_col = after_df[col].astype(float).fillna(0)
                if original_type != "float64":
                    self.logger.info(
                        f"Converted column '{col}' from {original_type} to float64 for comparison"
                    )
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Could not convert column '{col}' to float: {e}")
                # Fallback to original handling
                before_col = before_df[col].fillna(0)
                after_col = after_df[col].fillna(0)

            # Compare with tolerance
            diff_mask = ~np.isclose(before_col, after_col, atol=tolerance, rtol=0)

            if diff_mask.any():
                diff_count = diff_mask.sum()
                self.logger.warning(f"Found {diff_count} differences in column '{col}'")

                # Get indices where differences occur
                diff_indices = before_df.index[diff_mask].tolist()

                for idx in diff_indices:
                    before_val = before_df.loc[idx, col]
                    after_val = after_df.loc[idx, col]

                    # Calculate difference safely
                    try:
                        if pd.notna(before_val) and pd.notna(after_val):
                            diff_value = abs(float(before_val) - float(after_val))
                        else:
                            diff_value = "NaN difference"
                    except (ValueError, TypeError):
                        diff_value = "Cannot calculate difference"

                    numerical_differences.append(
                        {
                            "row_index": idx,
                            "column": col,
                            "before_value": before_val,
                            "after_value": after_val,
                            "difference": diff_value,
                            "type": "numerical",
                        }
                    )

                self.differences.append(
                    {
                        "type": "numerical_column_differences",
                        "column": col,
                        "difference_count": diff_count,
                        "total_rows": len(before_df),
                    }
                )
            else:
                self.logger.info(f"✓ No differences found in numerical column '{col}'")

        return pd.DataFrame(numerical_differences)

    def compare_non_numerical_columns(
        self,
        before_df: pd.DataFrame,
        after_df: pd.DataFrame,
        non_numerical_columns: List[str],
    ) -> pd.DataFrame:
        """
        Compare non-numerical columns for exact matches.

        Args:
            before_df (pd.DataFrame): Before dataframe
            after_df (pd.DataFrame): After dataframe
            non_numerical_columns (List[str]): List of non-numerical columns

        Returns:
            pd.DataFrame: DataFrame containing differences
        """
        self.logger.info("Comparing non-numerical columns for exact matches")

        non_numerical_differences = []

        for col in non_numerical_columns:
            self.logger.info(f"Comparing non-numerical column: {col}")

            # Handle NaN values by converting to string
            before_col = before_df[col].astype(str)
            after_col = after_df[col].astype(str)

            # Compare for exact matches
            diff_mask = before_col != after_col

            if diff_mask.any():
                diff_count = diff_mask.sum()
                self.logger.warning(f"Found {diff_count} differences in column '{col}'")

                # Get indices where differences occur
                diff_indices = before_df.index[diff_mask].tolist()

                for idx in diff_indices:
                    non_numerical_differences.append(
                        {
                            "row_index": idx,
                            "column": col,
                            "before_value": before_df.loc[idx, col],
                            "after_value": after_df.loc[idx, col],
                            "difference": "exact_match_failed",
                            "type": "non_numerical",
                        }
                    )

                self.differences.append(
                    {
                        "type": "non_numerical_column_differences",
                        "column": col,
                        "difference_count": diff_count,
                        "total_rows": len(before_df),
                    }
                )
            else:
                self.logger.info(
                    f"✓ No differences found in non-numerical column '{col}'"
                )

        return pd.DataFrame(non_numerical_differences)

    def save_differences_to_csv(
        self, differences_df: pd.DataFrame, output_file: str = None
    ):
        """
        Save differences to a CSV file.

        Args:
            differences_df (pd.DataFrame): DataFrame containing all differences
            output_file (str): Output CSV file path
        """
        if output_file is None:
            output_file = "data_differences.csv"

        if not differences_df.empty:
            differences_df.to_csv(output_file, index=False)
            self.logger.info(f"Differences saved to: {output_file}")
            self.logger.info(f"Total differences found: {len(differences_df)}")
        else:
            self.logger.info("No differences found - no CSV file created")

    def generate_summary_report(self):
        """
        Generate a summary report of all comparisons.
        """
        self.logger.info("=" * 50)
        self.logger.info("COMPARISON SUMMARY REPORT")
        self.logger.info("=" * 50)

        if not self.differences:
            self.logger.info("✓ NO DIFFERENCES FOUND - DataFrames are identical!")
        else:
            self.logger.warning(f"Found {len(self.differences)} types of differences:")
            for i, diff in enumerate(self.differences, 1):
                self.logger.warning(
                    f"{i}. {diff['type']}: {diff.get('description', 'See details above')}"
                )

    def compare_datasets(
        self,
        before_path: str,
        after_path: str,
        tolerance: float = 0.01,
        output_csv: str = None,
    ) -> bool:
        """
        Main method to compare two datasets.

        Args:
            before_path (str): Path for before dataset (local or S3, CSV or Parquet)
            after_path (str): Path for after dataset (local or S3, CSV or Parquet)
            tolerance (float): Tolerance for numerical comparisons
            output_csv (str): Output CSV file for differences

        Returns:
            bool: True if datasets are identical, False otherwise
        """
        self.logger.info("Starting comprehensive data comparison")
        self.logger.info(f"Before dataset: {before_path}")
        self.logger.info(f"After dataset: {after_path}")
        self.logger.info(f"Numerical tolerance: {tolerance}")

        try:
            # Read datasets
            before_df = self.read_data(before_path, "before_df")
            after_df = self.read_data(after_path, "after_df")

            # Compare shapes
            self.compare_shapes(before_df, after_df)

            # Get common columns
            common_columns = self.get_common_columns(before_df, after_df)

            if not common_columns:
                self.logger.error("No common columns found between datasets!")
                return False

            # Check if DataFrames have different number of rows
            if before_df.shape[0] != after_df.shape[0]:
                self.logger.error(
                    f"Cannot perform row-by-row comparison: DataFrames have different number of rows "
                    f"(before: {before_df.shape[0]}, after: {after_df.shape[0]})"
                )
                self.logger.error(
                    "Row-by-row comparison requires identical number of rows in both datasets"
                )
                return False

            # Filter datasets to common columns only
            before_df_common = before_df[common_columns].copy()
            after_df_common = after_df[common_columns].copy()

            # Identify column types
            numerical_columns, non_numerical_columns = self.identify_column_types(
                before_df_common, common_columns
            )

            # Compare numerical columns
            numerical_diffs = pd.DataFrame()
            if numerical_columns:
                numerical_diffs = self.compare_numerical_columns(
                    before_df_common, after_df_common, numerical_columns, tolerance
                )

            # Compare non-numerical columns
            non_numerical_diffs = pd.DataFrame()
            if non_numerical_columns:
                non_numerical_diffs = self.compare_non_numerical_columns(
                    before_df_common, after_df_common, non_numerical_columns
                )

            # Combine all differences
            all_differences = pd.concat(
                [numerical_diffs, non_numerical_diffs], ignore_index=True
            )

            # Save differences to CSV
            self.save_differences_to_csv(all_differences, output_csv)

            # Generate summary report
            self.generate_summary_report()

            # Return True if no differences found
            return len(self.differences) == 0

        except Exception as e:
            self.logger.error(f"Error during comparison: {str(e)}")
            raise


def main():
    """
    Main function to run the data comparison tool.
    """
    # Hardcoded paths - modify these for your specific use case
    # Can be local paths or S3 paths, CSV or Parquet files
    BEFORE_PATH = "sample_data/parquet/before_data.parquet"
    AFTER_PATH = "sample_data/parquet/after_data.parquet"

    parser = argparse.ArgumentParser(description="Comprehensive Data Comparison Tool")
    parser.add_argument(
        "--before-path",
        default=BEFORE_PATH,
        help=f"Path for before dataset - local or S3, CSV or Parquet (default: {BEFORE_PATH})",
    )
    parser.add_argument(
        "--after-path",
        default=AFTER_PATH,
        help=f"Path for after dataset - local or S3, CSV or Parquet (default: {AFTER_PATH})",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Tolerance for numerical comparisons (default: 0.01)",
    )
    parser.add_argument("--output-csv", help="Output CSV file for differences")
    parser.add_argument("--log-file", help="Log file path")

    args = parser.parse_args()

    # Create comparison tool
    tool = DataComparisonTool(log_file=args.log_file)

    # Display the paths being used
    print(f"Using data paths:")
    print(f"  Before: {args.before_path}")
    print(f"  After:  {args.after_path}")
    print()

    try:
        # Run comparison
        datasets_identical = tool.compare_datasets(
            before_path=args.before_path,
            after_path=args.after_path,
            tolerance=args.tolerance,
            output_csv=args.output_csv,
        )

        # Exit with appropriate code
        sys.exit(0 if datasets_identical else 1)

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(2)


if __name__ == "__main__":
    main()
