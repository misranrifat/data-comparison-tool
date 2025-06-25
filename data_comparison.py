import pandas as pd
import numpy as np

def compare_dataframes(df1, df2, float_tol=1e-8, verbose=True, save_csv=True, csv_path="df_mismatches.csv"):
    result = []

    # 1. Check shape
    if df1.shape != df2.shape:
        result.append(f"Shape mismatch: df1 {df1.shape}, df2 {df2.shape}")

    # 2. Check column names/order
    if not all(df1.columns == df2.columns):
        result.append(f"Column mismatch.\ndf1 columns: {list(df1.columns)}\ndf2 columns: {list(df2.columns)}")

    # 3. Check column dtypes
    dtype_diff = {}
    for col in df1.columns.intersection(df2.columns):
        if df1[col].dtype != df2[col].dtype:
            dtype_diff[col] = (df1[col].dtype, df2[col].dtype)
    if dtype_diff:
        result.append("Column dtype mismatches: " + str(dtype_diff))

    # 4. Thorough value check
    min_len = min(len(df1), len(df2))
    unequal = pd.DataFrame(False, index=range(min_len), columns=df1.columns)
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
        for r, c in zip(rows, cols):
            mismatch_rows.append({
                "row_index": r,
                "column": c,
                "df1_value": df1.at[r, c],
                "df2_value": df2.at[r, c]
            })
        num_to_save = min(1000, len(mismatch_rows))
        result.append(f"Found {len(mismatch_rows)} cell mismatches.\nSaving first {num_to_save} to '{csv_path}'.")

        # Save to CSV if required
        if save_csv and num_to_save > 0:
            mismatch_df = pd.DataFrame(mismatch_rows[:num_to_save])
            mismatch_df.to_csv(csv_path, index=False)
    else:
        result.append("No cell mismatches found.")

    # 5. Extra rows in either DataFrame
    if len(df1) > min_len:
        result.append(f"Extra rows in df1: {len(df1) - min_len}")
    if len(df2) > min_len:
        result.append(f"Extra rows in df2: {len(df2) - min_len}")

    if not result:
        output = "DataFrames are identical (within tolerance for floats)."
    else:
        output = "\n".join(result)

    if verbose:
        print(output)
    return output

# Example usage:
# compare_dataframes(df1, df2)