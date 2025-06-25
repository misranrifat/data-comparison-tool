#!/usr/bin/env python3
"""
Unit tests for the DataComparisonTool class.

This module contains comprehensive tests for all methods in the DataComparisonTool class,
including data reading, comparison operations, and file output functionality.
"""

import logging
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from data_comparison_tool import ID_COLUMN, DataComparisonTool


class TestDataComparisonTool(unittest.TestCase):
    """Test cases for DataComparisonTool class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.test_log_file = os.path.join(self.temp_dir, "test.log")

        # Initialize the tool
        self.tool = DataComparisonTool(log_file=self.test_log_file)

        # Create sample test data
        self.sample_data_before = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
                "age": [25, 30, 35, 40, 45],
                "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
                "department": ["IT", "HR", "Finance", "IT", "HR"],
                "active": [True, False, True, False, True],
                "score": [85.5, 92.3, 78.1, 88.7, 95.2],
            }
        )

        self.sample_data_after = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "name": [
                    "Alice",
                    "Bob",
                    "Charlie",
                    "David",
                    "Eva",
                ],  # Changed Eve to Eva
                "age": [25, 30, 35, 40, 45],
                "salary": [
                    50000.0,
                    60000.0,
                    70000.0,
                    80000.0,
                    91000.0,
                ],  # Changed 90000 to 91000
                "department": [
                    "IT",
                    "Marketing",
                    "Finance",
                    "IT",
                    "HR",
                ],  # Changed HR to Marketing
                "active": [True, False, True, False, True],
                "score": [85.5, 92.3, 78.1, 88.7, 96.2],  # Changed 95.2 to 96.2
            }
        )

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up temporary files
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init(self):
        """Test DataComparisonTool initialization."""
        tool = DataComparisonTool()
        self.assertIsNotNone(tool.logger)
        self.assertEqual(tool.differences, [])

    def test_setup_logging(self):
        """Test logging setup functionality."""
        # Test that log file is created
        self.assertTrue(os.path.exists(self.test_log_file))

        # Test that logger is configured
        self.assertIsNotNone(self.tool.logger)

        # Test that the logger name is correct
        self.assertEqual(self.tool.logger.name, "data_comparison_tool")

    def test_detect_file_type_and_location(self):
        """Test file type and location detection."""
        # Test S3 parquet
        location, file_type = self.tool.detect_file_type_and_location(
            "s3://bucket/data.parquet"
        )
        self.assertEqual(location, "s3")
        self.assertEqual(file_type, "parquet")

        # Test S3 CSV
        location, file_type = self.tool.detect_file_type_and_location(
            "s3://bucket/data.csv"
        )
        self.assertEqual(location, "s3")
        self.assertEqual(file_type, "csv")

        # Test local parquet
        location, file_type = self.tool.detect_file_type_and_location(
            "./data/file.parquet"
        )
        self.assertEqual(location, "local")
        self.assertEqual(file_type, "parquet")

        # Test local CSV
        location, file_type = self.tool.detect_file_type_and_location("./data/file.csv")
        self.assertEqual(location, "local")
        self.assertEqual(file_type, "csv")

        # Test path with parquet folder
        location, file_type = self.tool.detect_file_type_and_location("./data/parquet/")
        self.assertEqual(location, "local")
        self.assertEqual(file_type, "parquet")

        # Test path with csv folder
        location, file_type = self.tool.detect_file_type_and_location("./data/csv/")
        self.assertEqual(location, "local")
        self.assertEqual(file_type, "csv")

    @patch("data_comparison_tool.dd.read_parquet")
    def test_read_data_parquet(self, mock_read_parquet):
        """Test reading parquet data."""
        # Mock dask dataframe
        mock_dask_df = MagicMock()
        mock_dask_df.columns = ["ID", "Name", "Age"]
        mock_dask_df.npartitions = 2
        mock_dask_df.compute.return_value = pd.DataFrame(
            {"ID": [1, 2, 3], "Name": ["Alice", "Bob", "Charlie"], "Age": [25, 30, 35]}
        )
        mock_read_parquet.return_value = mock_dask_df

        # Test reading parquet data
        result = self.tool.read_data("test.parquet", "test_dataset")

        # Verify the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(
            list(result.columns), ["id", "name", "age"]
        )  # Should be lowercase
        self.assertEqual(len(result), 3)
        mock_read_parquet.assert_called_once_with("test.parquet")

    @patch("data_comparison_tool.dd.read_csv")
    def test_read_data_csv(self, mock_read_csv):
        """Test reading CSV data."""
        # Mock dask dataframe
        mock_dask_df = MagicMock()
        mock_dask_df.columns = ["ID", "Name", "Age"]
        mock_dask_df.npartitions = 1
        mock_dask_df.compute.return_value = pd.DataFrame(
            {"ID": [1, 2, 3], "Name": ["Alice", "Bob", "Charlie"], "Age": [25, 30, 35]}
        )
        mock_read_csv.return_value = mock_dask_df

        # Test reading CSV data
        result = self.tool.read_data("test.csv", "test_dataset")

        # Verify the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(
            list(result.columns), ["id", "name", "age"]
        )  # Should be lowercase
        self.assertEqual(len(result), 3)
        mock_read_csv.assert_called_once_with("test.csv", assume_missing=True)

    def test_get_common_columns(self):
        """Test getting common columns between dataframes."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        df2 = pd.DataFrame({"b": [7, 8], "c": [9, 10], "d": [11, 12]})

        common_columns = self.tool.get_common_columns(df1, df2)

        self.assertEqual(set(common_columns), {"b", "c"})
        self.assertEqual(
            len(self.tool.differences), 2
        )  # Should record column differences

    def test_identify_column_types(self):
        """Test identification of numerical and non-numerical columns."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "salary": [50000.0, 60000.0, 70000.0],
                "active": [True, False, True],
            }
        )

        columns = ["id", "name", "age", "salary", "active"]
        numerical, non_numerical = self.tool.identify_column_types(df, columns)

        self.assertIn("id", numerical)
        self.assertIn("age", numerical)
        self.assertIn("salary", numerical)
        self.assertIn("active", numerical)
        self.assertIn("name", non_numerical)

    def test_compare_shapes(self):
        """Test shape comparison between dataframes."""
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2], "b": [4, 5]})

        # Test different shapes
        self.tool.compare_shapes(df1, df2)
        self.assertEqual(len(self.tool.differences), 1)
        self.assertEqual(self.tool.differences[0]["type"], "shape_difference")

        # Reset differences and test same shapes
        self.tool.differences = []
        df3 = pd.DataFrame({"a": [1, 2, 3], "c": [7, 8, 9]})
        self.tool.compare_shapes(df1, df3)
        self.assertEqual(len(self.tool.differences), 0)

    def test_compare_numerical_columns(self):
        """Test numerical column comparison."""
        # Create test data with known differences
        before_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "value1": [10.0, 20.0, 30.0],
                "value2": [100.0, 200.0, 300.0],
            }
        )
        after_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "value1": [10.0, 20.5, 30.0],  # Difference in row 2
                "value2": [100.0, 200.0, 305.0],  # Difference in row 3
            }
        )

        numerical_columns = ["value1", "value2"]
        differences_df = self.tool.compare_numerical_columns(
            before_df, after_df, numerical_columns, tolerance=0.01
        )

        # Should find 2 differences
        self.assertEqual(len(differences_df), 2)
        self.assertEqual(differences_df.iloc[0]["id"], 2)  # First difference at id=2
        self.assertEqual(differences_df.iloc[0]["column"], "value1")
        self.assertEqual(differences_df.iloc[1]["id"], 3)  # Second difference at id=3
        self.assertEqual(differences_df.iloc[1]["column"], "value2")

    def test_compare_numerical_columns_with_tolerance(self):
        """Test numerical column comparison with tolerance."""
        before_df = pd.DataFrame({"id": [1, 2], "value": [10.0, 20.0]})
        after_df = pd.DataFrame(
            {"id": [1, 2], "value": [10.005, 20.02]}  # Small differences
        )

        # Test with tight tolerance - should find differences
        differences_df = self.tool.compare_numerical_columns(
            before_df, after_df, ["value"], tolerance=0.001
        )
        self.assertEqual(len(differences_df), 2)

        # Test with loose tolerance - should find no differences
        self.tool.differences = []  # Reset
        differences_df = self.tool.compare_numerical_columns(
            before_df, after_df, ["value"], tolerance=0.1
        )
        self.assertEqual(len(differences_df), 0)

    def test_compare_non_numerical_columns(self):
        """Test non-numerical column comparison."""
        before_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "department": ["IT", "HR", "Finance"],
            }
        )
        after_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charles"],  # Difference in row 3
                "department": ["IT", "Marketing", "Finance"],  # Difference in row 2
            }
        )

        non_numerical_columns = ["name", "department"]
        differences_df = self.tool.compare_non_numerical_columns(
            before_df, after_df, non_numerical_columns
        )

        # Should find 2 differences
        self.assertEqual(len(differences_df), 2)

        # Check the differences
        name_diff = differences_df[differences_df["column"] == "name"].iloc[0]
        dept_diff = differences_df[differences_df["column"] == "department"].iloc[0]

        self.assertEqual(name_diff["id"], 3)
        self.assertEqual(name_diff["before_value"], "Charlie")
        self.assertEqual(name_diff["after_value"], "Charles")

        self.assertEqual(dept_diff["id"], 2)
        self.assertEqual(dept_diff["before_value"], "HR")
        self.assertEqual(dept_diff["after_value"], "Marketing")

    def test_compare_columns_without_id_column(self):
        """Test comparison when ID column doesn't exist."""
        # Create dataframes without ID column
        before_df = pd.DataFrame({"name": ["Alice", "Bob"], "value": [10.0, 20.0]})
        after_df = pd.DataFrame({"name": ["Alice", "Bob"], "value": [10.5, 20.0]})

        # Test numerical comparison - should use row index
        differences_df = self.tool.compare_numerical_columns(
            before_df, after_df, ["value"], tolerance=0.01
        )

        self.assertEqual(len(differences_df), 1)
        self.assertEqual(differences_df.iloc[0]["id"], 0)  # Should be row index
        self.assertEqual(differences_df.iloc[0]["column"], "value")

    def test_save_differences_to_csv(self):
        """Test saving differences to CSV file."""
        # Create test differences dataframe
        differences_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "column": ["name", "salary", "department"],
                "before_value": ["Alice", 50000, "IT"],
                "after_value": ["Alicia", 51000, "Engineering"],
                "difference": [
                    "exact_match_failed",
                    "not_calculated",
                    "exact_match_failed",
                ],
                "type": ["non_numerical", "numerical", "non_numerical"],
            }
        )

        output_file = os.path.join(self.temp_dir, "test_differences.csv")

        # Test saving differences
        self.tool.save_differences_to_csv(differences_df, output_file)

        # Verify file was created and has correct content
        self.assertTrue(os.path.exists(output_file))

        saved_df = pd.read_csv(output_file)
        self.assertEqual(len(saved_df), 3)
        self.assertEqual(
            list(saved_df.columns),
            ["id", "column", "before_value", "after_value", "difference", "type"],
        )

    def test_save_differences_to_csv_with_limit(self):
        """Test saving differences to CSV with row limit."""
        # Create large differences dataframe
        differences_df = pd.DataFrame(
            {
                "id": range(1500),
                "column": ["test"] * 1500,
                "before_value": range(1500),
                "after_value": range(1500, 3000),
                "difference": ["not_calculated"] * 1500,
                "type": ["numerical"] * 1500,
            }
        )

        output_file = os.path.join(self.temp_dir, "test_differences_limited.csv")

        # Test saving with default limit (1000 rows)
        self.tool.save_differences_to_csv(differences_df, output_file)

        # Verify only first 1000 rows were saved
        saved_df = pd.read_csv(output_file)
        self.assertEqual(len(saved_df), 1000)

    def test_save_differences_to_csv_empty(self):
        """Test saving empty differences dataframe."""
        empty_df = pd.DataFrame()
        output_file = os.path.join(self.temp_dir, "test_empty.csv")

        # Should not create file for empty dataframe
        self.tool.save_differences_to_csv(empty_df, output_file)
        self.assertFalse(os.path.exists(output_file))

    def test_generate_summary_report(self):
        """Test summary report generation."""
        # Test with no differences
        with patch.object(self.tool.logger, "info") as mock_info:
            self.tool.generate_summary_report()
            mock_info.assert_any_call(
                "NO DIFFERENCES FOUND - DataFrames are identical!"
            )

        # Test with differences
        self.tool.differences = [
            {"type": "shape_difference", "description": "Different shapes"},
            {"type": "column_difference", "description": "Different columns"},
        ]

        with patch.object(self.tool.logger, "warning") as mock_warning:
            self.tool.generate_summary_report()
            mock_warning.assert_any_call("Found 2 types of differences:")

    @patch("data_comparison_tool.DataComparisonTool.read_data")
    def test_compare_datasets_different_row_counts(self, mock_read_data):
        """Test dataset comparison with different row counts."""
        # Mock dataframes with different row counts
        before_df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        after_df = pd.DataFrame({"id": [1, 2], "value": [10, 20]})

        mock_read_data.side_effect = [before_df, after_df]

        # Should return False due to different row counts
        result = self.tool.compare_datasets("before.csv", "after.csv")
        self.assertFalse(result)

    @patch("data_comparison_tool.DataComparisonTool.read_data")
    def test_compare_datasets_no_common_columns(self, mock_read_data):
        """Test dataset comparison with no common columns."""
        # Mock dataframes with no common columns
        before_df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        after_df = pd.DataFrame({"c": [1, 2, 3], "d": [10, 20, 30]})

        mock_read_data.side_effect = [before_df, after_df]

        # Should return False due to no common columns
        result = self.tool.compare_datasets("before.csv", "after.csv")
        self.assertFalse(result)

    @patch("data_comparison_tool.DataComparisonTool.read_data")
    def test_compare_datasets_identical(self, mock_read_data):
        """Test dataset comparison with identical datasets."""
        # Mock identical dataframes
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        mock_read_data.side_effect = [df.copy(), df.copy()]

        # Should return True for identical datasets
        result = self.tool.compare_datasets("before.csv", "after.csv")
        self.assertTrue(result)

    @patch("data_comparison_tool.DataComparisonTool.read_data")
    def test_compare_datasets_with_differences(self, mock_read_data):
        """Test dataset comparison with differences."""
        before_df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        after_df = pd.DataFrame(
            {"id": [1, 2, 3], "value": [10, 25, 30]}
        )  # Difference in row 2

        mock_read_data.side_effect = [before_df, after_df]

        # Should return False due to differences
        result = self.tool.compare_datasets("before.csv", "after.csv")
        self.assertFalse(result)
        self.assertGreater(len(self.tool.differences), 0)

    def test_id_column_configuration(self):
        """Test that ID column configuration is respected."""
        # This test verifies that the global ID_COLUMN variable is used
        self.assertIsNotNone(ID_COLUMN)
        self.assertIsInstance(ID_COLUMN, str)

        # Test with custom ID column name
        original_id_column = ID_COLUMN

        # Create test data with custom ID column
        before_df = pd.DataFrame(
            {
                "customer_id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "value": [10, 20, 30],
            }
        )
        after_df = pd.DataFrame(
            {
                "customer_id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charles"],  # Difference
                "value": [10, 20, 30],
            }
        )

        # Temporarily change ID_COLUMN
        import data_comparison_tool

        data_comparison_tool.ID_COLUMN = "customer_id"

        try:
            differences_df = self.tool.compare_non_numerical_columns(
                before_df, after_df, ["name"]
            )

            # Should use customer_id as the identifier
            self.assertEqual(len(differences_df), 1)
            self.assertEqual(differences_df.iloc[0]["id"], 3)

        finally:
            # Restore original ID_COLUMN
            data_comparison_tool.ID_COLUMN = original_id_column


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.tool = DataComparisonTool()

    def tearDown(self):
        """Clean up after integration tests."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow with sample data."""
        # Create sample parquet files
        before_data = pd.DataFrame(
            {
                "id": range(1, 101),
                "name": [f"Person_{i}" for i in range(1, 101)],
                "age": np.random.randint(20, 70, 100),
                "salary": np.random.uniform(30000, 100000, 100),
                "department": np.random.choice(["IT", "HR", "Finance"], 100),
                "active": np.random.choice([True, False], 100),
            }
        )

        # Create after data with some differences
        after_data = before_data.copy()
        after_data.loc[10, "name"] = "Modified_Person"  # Change one name
        after_data.loc[20, "salary"] = (
            after_data.loc[20, "salary"] + 5000
        )  # Change one salary
        after_data.loc[30, "department"] = "Marketing"  # Change one department

        # Save to temporary parquet files
        before_file = os.path.join(self.temp_dir, "before.parquet")
        after_file = os.path.join(self.temp_dir, "after.parquet")

        before_data.to_parquet(before_file, index=False)
        after_data.to_parquet(after_file, index=False)

        # Run comparison
        output_csv = os.path.join(self.temp_dir, "differences.csv")

        with patch("data_comparison_tool.dd.read_parquet") as mock_read:
            # Mock dask read_parquet to return our test data
            mock_dask_df_before = MagicMock()
            mock_dask_df_before.columns = before_data.columns.tolist()
            mock_dask_df_before.npartitions = 1
            mock_dask_df_before.compute.return_value = before_data

            mock_dask_df_after = MagicMock()
            mock_dask_df_after.columns = after_data.columns.tolist()
            mock_dask_df_after.npartitions = 1
            mock_dask_df_after.compute.return_value = after_data

            mock_read.side_effect = [mock_dask_df_before, mock_dask_df_after]

            result = self.tool.compare_datasets(
                before_path=before_file, after_path=after_file, output_csv=output_csv
            )

        # Verify results
        self.assertFalse(result)  # Should find differences
        self.assertTrue(os.path.exists(output_csv))

        # Check differences CSV
        differences_df = pd.read_csv(output_csv)
        self.assertGreater(len(differences_df), 0)
        self.assertIn("id", differences_df.columns)
        self.assertIn("column", differences_df.columns)
        self.assertIn("before_value", differences_df.columns)
        self.assertIn("after_value", differences_df.columns)


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests

    # Run the tests
    unittest.main(verbosity=2)
