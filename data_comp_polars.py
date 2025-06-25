import polars as pl
import numpy as np
import logging
import sys

def setup_logger(log_path='compare_polars_dataframes.log', level=logging.INFO):
    logger = logging.getLogger("PolarsDataFrameComparator")
    logger.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger

def compare_polars_dataframes(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    float_tol=1e-8,
    save_csv=True,
    csv_path="polars_df_mismatches.csv",
    logger=None,
    limit=1000,
):
    logger.info("Starting Polars DataFrame comparison...")

    # 1. Check schema: column names, order, dtypes
    if df1.columns != df2.columns:
        logger.warning(f"Column name/order mismatch!\ndf1: {df1.columns}\ndf2: {df2.columns}")
    else:
        logger.info("Column names and order match.")

    dtype1 = [df1.schema[c] for c in df1.columns]
    dtype2 = [df2.schema[c] for c in df2.columns]
    if dtype1 != dtype2:
        logger.warning(f"Column type mismatch!\ndf1: {dtype1}\ndf2: {dtype2}")
    else:
        logger.info("Column dtypes match.")

    # 2. Row count
    n1, n2 = df1.height, df2.height
    if n1 != n2:
        logger.warning(f"Row count mismatch: df1={n1}, df2={n2}")
    else:
        logger.info(f"Row count matches: {n1}")

    min_len = min(n1, n2)
    mismatch_rows = []
    mismatch_count = 0

    # 3. Value comparison
    for col in df1.columns:
        s1 = df1[col][:min_len]
        s2 = df2[col][:min_len]

        dtype = df1.schema[col]
        if dtype in [pl.Float32, pl.Float64]:
            unequal = ~np.isclose(s1.to_numpy(), s2.to_numpy(), atol=float_tol, equal_nan=True)
            # Also check for mismatched nulls
            nulls1 = s1.is_null()
            nulls2 = s2.is_null()
            unequal |= (nulls1 != nulls2).to_numpy()
        else:
            # String, Int, Bool, Date, etc.
            unequal = (s1 != s2).to_numpy()
            nulls1 = s1.is_null().to_numpy()
            nulls2 = s2.is_null().to_numpy()
            both_null = nulls1 & nulls2
            unequal = unequal & (~both_null)

        mismatch_indices = np.where(unequal)[0]
        if len(mismatch_indices) > 0:
            for idx in mismatch_indices:
                mismatch_rows.append({
                    "row_index": idx,
                    "column": col,
                    "df1_value": s1[idx],
                    "df2_value": s2[idx],
                })
                mismatch_count += 1
                if mismatch_count >= limit:
                    break
        if mismatch_count >= limit:
            break

    if mismatch_rows:
        logger.warning(f"Found {len(mismatch_rows)} mismatches. Saving up to {limit} to {csv_path}")
        import pandas as pd
        pd.DataFrame(mismatch_rows[:limit]).to_csv(csv_path, index=False)
        logger.info(f"Saved mismatches to {csv_path}")
    else:
        logger.info("No mismatches found!")

    if n1 > min_len:
        logger.warning(f"Extra rows in df1: {n1 - min_len}")
    if n2 > min_len:
        logger.warning(f"Extra rows in df2: {n2 - min_len}")

def main():
    logger = setup_logger()

    # Example: Create two Polars DataFrames
    df1 = pl.DataFrame({
        "id": [1, 2, 3],
        "price": [10.2, 11.0, 12.1],
        "product": ["Apple", "Orange", "Banana"],
        "available": [True, False, True],
        "date_added": [pl.Date("2023-05-01"), pl.Date("2023-05-02"), pl.Date("2023-05-03")],
    })

    df2 = pl.DataFrame({
        "id": [1, 2, 3],
        "price": [10.2, 11.0, 13.1],   # changed last price
        "product": ["Apple", "Orange", "Banana"],
        "available": [True, False, False],  # changed last availability
        "date_added": [pl.Date("2023-05-01"), pl.Date("2023-05-02"), pl.Date("2023-05-03")],
    })

    compare_polars_dataframes(
        df1, df2,
        float_tol=1e-5,
        save_csv=True,
        csv_path="polars_df_mismatches.csv",
        logger=logger,
        limit=1000
    )

    logger.info("Done.")

if __name__ == "__main__":
    main()