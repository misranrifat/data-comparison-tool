import pandas as pd
import numpy as np
import logging
import sys

def setup_logger(log_path='compare_dataframes.log', level=logging.INFO):
    # Set up a logger that writes to file and stdout
    logger = logging.getLogger("DataFrameComparator")
    logger.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    # File handler
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(formatter)
    # Stream handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    # Avoid duplicate logs
    if not logger.hasHandlers():
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger

def compare_dataframes(df1, df2, float_tol=1e-8, save_csv=True, csv_path="df_mismatches.csv", logger=None):
    result = []

    logger.info("Starting DataFrame comparison...")

    # 1. Check shape
    if df1.shape != df2.shape:
        msg = f"Shape mismatch: df1 {df1.shape}, df2 {df2.shape}"
        result.append(msg)
        logger.warning(msg)
    else:
        logger.info(f"Shape match: {df1.shape}")

    # 2. Check column names/order
    if not all(df1.columns == df2.columns):
        msg = f"Column mismatch.\ndf1 columns: {list(df1.columns)}\ndf2 columns: {list(df2.columns)}"
        result.append(msg)
        logger.warning(msg)
    else:
        logger.info("Column names and order match.")

    # 3. Check column dtypes
    dtype_diff = {}
    for col in df1.columns.intersection(df2.columns):
        if df1[col].dtype != df2[col].dtype:
            dtype_diff[col] = (df1[col].dtype, df2[col].dtype)
            logger.warning(f"Column '{col}' dtype mismatch: df1={df1[col].dtype}, df2={df2[col].dtype}")
    if dtype_diff:
        msg = "Column dtype mismatches: " + str(dtype_diff)
        result.append(msg)
        logger.warning(msg)
    else:
        logger.info("All matching columns have same dtypes.")

    # 4. Thorough value check
    min_len = min(len(df1), len(df2))
    unequal = pd.DataFrame(False, index=range(min_len), columns=df1.columns)
    logger.info("Checking each column for value mismatches...")
    for col in df1.columns.intersection(df2.columns):
        s1 = df1[col].iloc[:min_len]
        s2 = df2[col].iloc[:min_len]
        if np.issubdtype(s1.dtype, np.number) and np.issubdtype(s2.dtype, np.number):
            unequal[col] = ~np.isclose(s1, s2, atol=float_tol, equal_nan=True)
        else:
            unequal[col] = ~(s1 == s2) & ~(s1.isna() & s2.isna())
    diff_locs = np.where(unequal)
    mismatch_rows = []
    if len(diff_locs[0]) > 0:
        rows = diff_locs[0]
        cols = [df1.columns[i] for i in diff_locs[1]]
        logger.info(f"Found {len(rows)} mismatched cells.")
        for r, c in zip(rows, cols):
            mismatch_rows.append({
                "row_index": r,
                "column": c,
                "df1_value": df1.at[r, c],
                "df2_value": df2.at[r, c]
            })
        num_to_save = min(1000, len(mismatch_rows))
        msg = f"Found {len(mismatch_rows)} cell mismatches. Saving first {num_to_save} to '{csv_path}'."
        result.append(msg)
        logger.warning(msg)
        if save_csv and num_to_save > 0:
            mismatch_df = pd.DataFrame(mismatch_rows[:num_to_save])
            mismatch_df.to_csv(csv_path, index=False)
            logger.info(f"Saved mismatches to '{csv_path}'.")
    else:
        logger.info("No cell mismatches found.")

    # 5. Extra rows in either DataFrame
    if len(df1) > min_len:
        msg = f"Extra rows in df1: {len(df1) - min_len}"
        result.append(msg)
        logger.warning(msg)
    if len(df2) > min_len:
        msg = f"Extra rows in df2: {len(df2) - min_len}"
        result.append(msg)
        logger.warning(msg)

    if not result:
        output = "DataFrames are identical (within tolerance for floats)."
        logger.info(output)
    else:
        output = "\n".join(result)

    return output

def main():
    # Setup logging
    logger = setup_logger()
    logger.info("Script started.")

    # Demo: Load or create two DataFrames here. In reality, load from CSV/DB/whatever.
    df1 = pd.DataFrame({
        "id": [1, 2, 3],
        "price": [10.2, 11.0, 12.1],
        "product": ["Apple", "Orange", "Banana"],
        "available": [True, False, True],
        "date_added": pd.to_datetime(["2023-05-01", "2023-05-02", "2023-05-03"])
    })

    df2 = pd.DataFrame({
        "id": [1, 2, 3],
        "price": [10.2, 11.0, 13.1],   # changed last price
        "product": ["Apple", "Orange", "Banana"],
        "available": [True, False, False],  # changed last availability
        "date_added": pd.to_datetime(["2023-05-01", "2023-05-02", "2023-05-03"])
    })

    # Compare DataFrames
    summary = compare_dataframes(
        df1, df2, 
        float_tol=1e-5, 
        save_csv=True, 
        csv_path="df_mismatches.csv", 
        logger=logger
    )

    logger.info("Comparison summary:")
    logger.info("\n" + summary)

    logger.info("Script finished.")

if __name__ == "__main__":
    main()