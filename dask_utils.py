import dask
from dask.dataframe import read_parquet, read_csv

class DaskUtils:
    @staticmethod
    def read_parquet(path, **kwargs):
        """Read a Parquet file into a Dask DataFrame."""
        return read_parquet(path, **kwargs)

    @staticmethod
    def read_csv(path, **kwargs):
        """Read a CSV file into a Dask DataFrame."""
        return read_csv(path, **kwargs)

    @staticmethod
    def get_row_count(df):
        """Return the number of rows in the DataFrame."""
        return df.shape[0].compute()

    @staticmethod
    def get_col_count(df):
        """Return the number of columns in the DataFrame."""
        return len(df.columns)

    @staticmethod
    def get_column_names(df):
        """Return a list of column names."""
        return list(df.columns)

    @staticmethod
    def get_distinct_count(df, col):
        """Return the number of distinct values in a column."""
        return df[col].nunique().compute()

    @staticmethod
    def get_distinct_values(df, col, npartitions=1):
        """Return all distinct values in a column as a list."""
        return df[col].drop_duplicates().compute().to_list()

    @staticmethod
    def filter_rows(df, expr):
        """Filter rows based on a boolean expression string."""
        return df.query(expr)

    @staticmethod
    def save_parquet(df, path, **kwargs):
        """Save Dask DataFrame to a Parquet file."""
        df.to_parquet(path, **kwargs)

    @staticmethod
    def save_csv(df, path, **kwargs):
        """Save Dask DataFrame to CSV files."""
        df.to_csv(path, **kwargs)

    @staticmethod
    def head(df, n=5):
        """Return the first n rows as a pandas DataFrame."""
        return df.head(n, compute=True)

    @staticmethod
    def describe(df):
        """Return descriptive statistics as a pandas DataFrame."""
        return df.describe().compute()

    @staticmethod
    def value_counts(df, col, sort=True):
        """Return counts of unique values in a column."""
        vc = df[col].value_counts().compute()
        return vc.sort_values(ascending=False) if sort else vc

    @staticmethod
    def memory_usage(df):
        """Get estimated memory usage in bytes."""
        return df.memory_usage(deep=True).compute().sum()

    @staticmethod
    def info(df):
        """Print DataFrame info summary (like pandas .info())."""
        print(f"Dask DataFrame with {df.npartitions} partitions")
        print(f"Columns: {list(df.columns)}")
        print(df.dtypes)
        try:
            total_rows = df.shape[0].compute()
        except Exception:
            total_rows = "Unknown (possibly too large or lazy)"
        print(f"Total rows: {total_rows}")

    @staticmethod
    def dask_to_pandas(df, max_rows=None):
        """
        Convert a Dask DataFrame to a pandas DataFrame.
        Optionally limit the number of rows converted (recommended for large data).
        """
        if max_rows is not None:
            # Safer: grab a slice first to avoid OOM
            return df.head(max_rows, compute=True)
        else:
            # Warning: This will attempt to load the entire Dask DataFrame into memory!
            print("WARNING: You are about to load the entire Dask DataFrame into memory. Proceed only if you are sure it fits!")
            return df.compute()

# --- Example Usage ---
if __name__ == "__main__":
    # Change the paths and columns as needed for your use-case
    utils = DaskUtils()
    df = utils.read_parquet("your_big_file.parquet")
    print("Rows:", utils.get_row_count(df))
    print("Columns:", utils.get_col_count(df))
    print("Column Names:", utils.get_column_names(df))
    print("Distinct in 'id':", utils.get_distinct_count(df, "id"))
    print("Memory usage:", utils.memory_usage(df), "bytes")
    utils.info(df)

    # Dask to pandas (first 10,000 rows)
    pdf = utils.dask_to_pandas(df, max_rows=10000)
    print(pdf.head())