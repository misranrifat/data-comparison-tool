import psycopg2
import pandas as pd
import numpy as np
import logging
import sys

def setup_logger(log_path='compare_postgres_tables.log', level=logging.INFO):
    logger = logging.getLogger("PostgresTableComparator")
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

def fetch_schema_and_data(conn, query, logger, chunk_size=50000):
    # Fetch schema
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM ({query}) AS sub LIMIT 0;")
        colnames = [desc[0] for desc in cur.description]
        coltypes = [desc[1] for desc in cur.description]
    logger.info(f"Fetched schema: {colnames}")
    # Fetch data in chunks
    df = pd.DataFrame()
    for chunk in pd.read_sql_query(query, conn, chunksize=chunk_size):
        df = pd.concat([df, chunk], ignore_index=True)
        logger.info(f"Fetched {len(df)} rows so far...")
    return colnames, coltypes, df

def compare_postgres_tables(
    conn, query1, query2, 
    pk=None, float_tol=1e-8, 
    save_csv=True, csv_path="pg_table_mismatches.csv", 
    logger=None, limit=1000
):
    logger.info("Starting Postgres table comparison...")

    # 1. Fetch schema and data
    cols1, types1, df1 = fetch_schema_and_data(conn, query1, logger)
    cols2, types2, df2 = fetch_schema_and_data(conn, query2, logger)
    
    # 2. Schema check
    if cols1 != cols2:
        logger.warning(f"Column mismatch:\nt1: {cols1}\nt2: {cols2}")
    else:
        logger.info("Column names and order match.")

    if types1 != types2:
        logger.warning(f"Type mismatch:\nt1: {types1}\nt2: {types2}")
    else:
        logger.info("Column types match.")

    # 3. Row count check
    n1, n2 = len(df1), len(df2)
    if n1 != n2:
        logger.warning(f"Row count mismatch: t1={n1}, t2={n2}")
    else:
        logger.info(f"Row count matches: {n1}")

    # 4. Align on PK (or on row order if PK is None)
    if pk is not None and pk in df1.columns and pk in df2.columns:
        logger.info(f"Sorting on PK: {pk}")
        df1 = df1.sort_values(pk).reset_index(drop=True)
        df2 = df2.sort_values(pk).reset_index(drop=True)
    else:
        logger.info("Sorting by row order (no PK provided)")

    min_len = min(len(df1), len(df2))
    mismatch_rows = []
    mismatch_count = 0
    for col in cols1:
        s1 = df1[col][:min_len]
        s2 = df2[col][:min_len]
        # Guess at numeric dtype: int, float, decimal
        if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
            unequal = ~np.isclose(s1.fillna(np.nan), s2.fillna(np.nan), atol=float_tol, equal_nan=True)
        else:
            unequal = (s1 != s2) & ~(s1.isna() & s2.isna())
        mismatch_indices = np.where(unequal)[0]
        for idx in mismatch_indices:
            pk_val = df1.iloc[idx][pk] if pk and pk in df1.columns else idx
            mismatch_rows.append({
                "row_index_or_pk": pk_val,
                "column": col,
                "t1_value": s1.iloc[idx],
                "t2_value": s2.iloc[idx]
            })
            mismatch_count += 1
            if mismatch_count >= limit:
                break
        if mismatch_count >= limit:
            break

    if mismatch_rows:
        logger.warning(f"Found {len(mismatch_rows)} mismatches. Saving up to {limit} to {csv_path}")
        pd.DataFrame(mismatch_rows[:limit]).to_csv(csv_path, index=False)
        logger.info(f"Saved mismatches to {csv_path}")
    else:
        logger.info("No mismatches found!")

    if n1 > min_len:
        logger.warning(f"Extra rows in t1: {n1 - min_len}")
    if n2 > min_len:
        logger.warning(f"Extra rows in t2: {n2 - min_len}")

def main():
    # --- Fill in your DB credentials ---
    conn = psycopg2.connect(
        dbname='YOUR_DB', user='YOUR_USER',
        password='YOUR_PASS', host='YOUR_HOST', port=5432
    )
    logger = setup_logger()

    # Example: compare two tables by query
    query1 = "SELECT * FROM public.table1"
    query2 = "SELECT * FROM public.table2"
    pk = "id"  # Set to None if no PK, or the name of the PK column

    compare_postgres_tables(
        conn, query1, query2,
        pk=pk, float_tol=1e-5,
        save_csv=True, csv_path="pg_table_mismatches.csv",
        logger=logger, limit=1000
    )

    logger.info("Done.")
    conn.close()

if __name__ == "__main__":
    main()