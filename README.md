# Comprehensive Data Comparison Tool

A robust Python tool for comparing two datasets stored as CSV or Parquet files in local filesystem or S3. The tool uses Dask for efficient reading of large datasets and pandas for detailed comparison operations.

## Features

- **Flexible Data Reading**: Uses Dask to read CSV or Parquet files from local filesystem or S3
- **Auto-Detection**: Automatically detects file type (CSV/Parquet) and location (local/S3)
- **Column Intersection**: Compares only common columns between datasets
- **Numerical Comparison**: Uses numpy with configurable tolerance (default: 0.01) for numerical columns
- **Exact String Matching**: Performs exact matches for non-numerical columns
- **Shape Validation**: Compares DataFrame shapes and row counts
- **Comprehensive Logging**: Logs to both console and timestamped log files
- **Difference Export**: Saves all differences to a CSV file for detailed analysis
- **Summary Report**: Provides a comprehensive summary of all comparisons

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make the script executable (optional):
```bash
chmod +x data_comparison_tool.py
```

## Usage

### Basic Usage (Using Hardcoded Paths)

```bash
python data_comparison_tool.py
```

The script has hardcoded paths that you can modify in the script:
- `BEFORE_PATH = "s3://your-bucket/before-data-folder/"`
- `AFTER_PATH = "s3://your-bucket/after-data-folder/"`

### Override Hardcoded Paths

```bash
# Compare S3 Parquet files
python data_comparison_tool.py \
    --before-path s3://custom-bucket/before-data/ \
    --after-path s3://custom-bucket/after-data/

# Compare local CSV files
python data_comparison_tool.py \
    --before-path ./data/before_data.csv \
    --after-path ./data/after_data.csv

# Compare local directory with CSV files
python data_comparison_tool.py \
    --before-path ./data/before/*.csv \
    --after-path ./data/after/*.csv

# Mix local and S3
python data_comparison_tool.py \
    --before-path ./local/before.parquet \
    --after-path s3://bucket/after-data/
```

### Advanced Usage

```bash
python data_comparison_tool.py \
    --before-path s3://custom-bucket/before-data/ \
    --after-path ./local/after-data/ \
    --tolerance 0.001 \
    --output-csv my_differences.csv \
    --log-file my_comparison.log
```

### Parameters

- `--before-path`: Path for "before" dataset - local or S3, CSV or Parquet (optional, uses hardcoded default)
- `--after-path`: Path for "after" dataset - local or S3, CSV or Parquet (optional, uses hardcoded default)
- `--tolerance`: Numerical tolerance for comparisons (default: 0.01)
- `--output-csv`: Custom filename for differences CSV output
- `--log-file`: Custom filename for log file

### Supported Path Formats

#### Local Paths:
- Single file: `./data/file.csv` or `./data/file.parquet`
- Directory: `./data/csv_files/` or `./data/parquet_files/`
- Glob pattern: `./data/*.csv` or `./data/*.parquet`

#### S3 Paths:
- Directory: `s3://bucket/folder/`
- Specific path: `s3://bucket/path/to/files/`

### File Type Auto-Detection

The tool automatically detects file types based on:
- File extensions (`.csv`, `.parquet`)
- Path patterns (`/csv/`, `/parquet/`)
- Keywords in path (`csv`, `parquet`)

### Modifying Hardcoded Paths

Edit the following lines in `data_comparison_tool.py`:

```python
# Hardcoded paths - modify these for your specific use case
# Can be local paths or S3 paths, CSV or Parquet files
BEFORE_PATH = "s3://your-bucket/before-data-folder/"
AFTER_PATH = "s3://your-bucket/after-data-folder/"
```

## Output Files

### Log File
- Static log file: `data_comparison.log` (overwrites on each run)
- Contains detailed information about:
  - Data loading progress
  - Column comparisons
  - Shape comparisons
  - Difference counts per column
  - Summary report

### Differences CSV
- Static CSV file: `data_differences.csv` (overwrites on each run)
- Contains columns:
  - `row_index`: Index of the differing row
  - `column`: Column name where difference was found
  - `before_value`: Value in the before dataset
  - `after_value`: Value in the after dataset
  - `difference`: Calculated difference (for numerical) or 'exact_match_failed' (for strings)
  - `type`: 'numerical' or 'non_numerical'

## AWS Configuration

Ensure your AWS credentials are configured for S3 access:

```bash
# Using AWS CLI
aws configure

# Or using environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=your_region
```

## Example Output

```
Using data paths:
  Before: ./sample_data/csv/before_data.csv
  After:  ./sample_data/csv/after_data.csv

Logging initialized. Log file: data_comparison.log
2023-12-01 14:30:22 - INFO - Starting comprehensive data comparison
2023-12-01 14:30:22 - INFO - Before dataset: ./sample_data/csv/before_data.csv
2023-12-01 14:30:22 - INFO - After dataset: ./sample_data/csv/after_data.csv
2023-12-01 14:30:22 - INFO - Numerical tolerance: 0.01
2023-12-01 14:30:22 - INFO - Detected: local csv for path: ./sample_data/csv/before_data.csv
2023-12-01 14:30:22 - INFO - Reading csv files with Dask from local: ./sample_data/csv/before_data.csv
2023-12-01 14:30:25 - INFO - before_df - Pandas DataFrame info:
2023-12-01 14:30:25 - INFO -   - Shape: (1000, 7)
2023-12-01 14:30:25 - INFO -   - Memory usage: 0.05 MB
2023-12-01 14:30:25 - INFO - Detected: local csv for path: ./sample_data/csv/after_data.csv
2023-12-01 14:30:25 - INFO - Reading csv files with Dask from local: ./sample_data/csv/after_data.csv
2023-12-01 14:30:28 - INFO - after_df - Pandas DataFrame info:
2023-12-01 14:30:28 - INFO -   - Shape: (1000, 7)
2023-12-01 14:30:28 - INFO -   - Memory usage: 0.05 MB
2023-12-01 14:30:28 - INFO - ✓ DataFrames have identical shapes
2023-12-01 14:30:28 - INFO - Column comparison:
2023-12-01 14:30:28 - INFO -   - Common columns: 7
2023-12-01 14:30:28 - INFO - ✓ No differences found in numerical column 'salary'
2023-12-01 14:30:29 - WARNING - Found 50 differences in column 'name'
2023-12-01 14:30:29 - WARNING - Found 50 differences in column 'department'
2023-12-01 14:30:29 - INFO - Differences saved to: data_differences.csv
2023-12-01 14:30:29 - INFO - Total differences found: 100
2023-12-01 14:30:29 - INFO - ==================================================
2023-12-01 14:30:29 - INFO - COMPARISON SUMMARY REPORT
2023-12-01 14:30:29 - INFO - ==================================================
2023-12-01 14:30:29 - WARNING - Found 2 types of differences:
2023-12-01 14:30:29 - WARNING - 1. non_numerical_column_differences: See details above
2023-12-01 14:30:29 - WARNING - 2. non_numerical_column_differences: See details above
```

## Testing with Sample Data

To test the tool with sample data:

1. Create sample data:
```bash
python create_sample_data.py
```

2. Test with CSV files:
```bash
python data_comparison_tool.py --before-path sample_data/csv/before_data.csv --after-path sample_data/csv/after_data.csv
```

3. Test with Parquet files:
```bash
python data_comparison_tool.py --before-path sample_data/parquet/before_data.parquet --after-path sample_data/parquet/after_data.parquet
```

## Exit Codes

- `0`: Datasets are identical
- `1`: Differences found between datasets
- `2`: Fatal error occurred

## Requirements

- Python 3.7+
- pandas >= 1.5.0
- numpy >= 1.21.0
- dask[complete] >= 2023.1.0
- s3fs >= 2023.1.0
- pyarrow >= 10.0.0
- fsspec >= 2023.1.0

## Error Handling

The tool includes comprehensive error handling for:
- S3 connectivity issues
- Invalid parquet files
- Memory limitations
- Column type mismatches
- Missing AWS credentials

## Performance Considerations

- Uses Dask for memory-efficient reading of large datasets
- Processes data in chunks to handle datasets larger than available RAM
- Optimized for datasets with millions of rows
- Memory usage is logged for monitoring purposes

## Troubleshooting

### Common Issues

1. **AWS Credentials**: Ensure AWS credentials are properly configured
2. **S3 Permissions**: Verify read permissions for the S3 buckets/paths
3. **Memory Issues**: For very large datasets, consider increasing system memory or using a machine with more RAM
4. **Parquet Format**: Ensure files are valid parquet format and readable by pyarrow

### Debug Mode

For additional debugging information, you can modify the logging level in the script from `INFO` to `DEBUG`. 