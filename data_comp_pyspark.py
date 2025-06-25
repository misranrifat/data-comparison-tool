from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import logging
import sys
import os

def setup_logger(log_path='compare_spark_dataframes.log', level=logging.INFO):
    logger = logging.getLogger("SparkDataFrameComparator")
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

def compare_spark_dataframes(df1, df2, float_tol=1e-8, save_csv=True, csv_path="spark_df_mismatches.csv", logger=None, limit=1000):
    logger.info("Starting Spark DataFrame comparison...")

    # 1. Schema (columns, order, types)
    schema1 = [(f.name, f.dataType) for f in df1.schema.fields]
    schema2 = [(f.name, f.dataType) for f in df2.schema.fields]
    if schema1 != schema2:
        logger.warning(f"Schema mismatch!\ndf1: {schema1}\ndf2: {schema2}")
    else:
        logger.info("Schema matches.")

    # 2. Row count
    n1, n2 = df1.count(), df2.count()
    if n1 != n2:
        logger.warning(f"Row count mismatch: df1={n1}, df2={n2}")
    else:
        logger.info(f"Row count matches: {n1}")

    # 3. Add row numbers to keep order (PySpark DataFrames are unordered by default)
    w = F.window().partitionBy()
    df1_with_idx = df1.withColumn("_row_id", F.monotonically_increasing_id())
    df2_with_idx = df2.withColumn("_row_id", F.monotonically_increasing_id())

    # Because `monotonically_increasing_id` is not guaranteed to produce the same ids for matching rows,
    # we need a deterministic row id. We'll use zipWithIndex on RDDs:
    df1_with_idx = df1.rdd.zipWithIndex().toDF()
    df1_with_idx = df1_with_idx.select(
        [F.col("_1." + f.name).alias(f.name) for f in df1.schema.fields] +
        [F.col("_2").alias("_row_id")]
    )
    df2_with_idx = df2.rdd.zipWithIndex().toDF()
    df2_with_idx = df2_with_idx.select(
        [F.col("_1." + f.name).alias(f.name) for f in df2.schema.fields] +
        [F.col("_2").alias("_row_id")]
    )

    # 4. Join on _row_id, only for the shortest DataFrame
    min_len = min(n1, n2)
    df1_trim = df1_with_idx.filter(F.col("_row_id") < min_len)
    df2_trim = df2_with_idx.filter(F.col("_row_id") < min_len)

    joined = df1_trim.alias("df1").join(df2_trim.alias("df2"), on="_row_id")

    # 5. For each column, compare values and create mismatch indicators
    mismatch_exprs = []
    mismatch_cols = []
    for f in df1.schema.fields:
        col = f.name
        dtype = str(f.dataType)
        if dtype.startswith("DoubleType") or dtype.startswith("FloatType") or dtype.startswith("DecimalType"):
            expr = (F.abs(F.col(f"df1.{col}") - F.col(f"df2.{col}")) > float_tol) | (
                    F.col(f"df1.{col}").isNull() ^ F.col(f"df2.{col}").isNull())
        else:
            expr = (F.col(f"df1.{col}") != F.col(f"df2.{col}")) & (
                    ~(F.col(f"df1.{col}").isNull() & F.col(f"df2.{col}").isNull()))
        mismatch_exprs.append(expr.alias(f"{col}_mismatch"))
        mismatch_cols.append(col)

    mismatch_df = joined.select(
        F.col("_row_id"),
        *[F.col(f"df1.{col}").alias(f"{col}_df1") for col in mismatch_cols],
        *[F.col(f"df2.{col}").alias(f"{col}_df2") for col in mismatch_cols],
        *mismatch_exprs
    )

    # 6. Unpivot mismatches and collect the first N
    mismatches = []
    mismatch_rows = mismatch_df.collect()
    count = 0
    for row in mismatch_rows:
        row_id = row["_row_id"]
        for col in mismatch_cols:
            if row[f"{col}_mismatch"]:
                mismatches.append({
                    "row_index": row_id,
                    "column": col,
                    "df1_value": row[f"{col}_df1"],
                    "df2_value": row[f"{col}_df2"]
                })
                count += 1
                if count >= limit:
                    break
        if count >= limit:
            break

    if mismatches:
        logger.warning(f"Found {len(mismatches)} mismatches. Saving up to {limit} to CSV: {csv_path}")
        import pandas as pd
        pd.DataFrame(mismatches[:limit]).to_csv(csv_path, index=False)
        logger.info(f"Saved mismatches to {csv_path}")
    else:
        logger.info("No mismatches found!")

    # 7. Extra rows
    if n1 > min_len:
        logger.warning(f"Extra rows in df1: {n1 - min_len}")
    if n2 > min_len:
        logger.warning(f"Extra rows in df2: {n2 - min_len}")

def main():
    # Spark session
    spark = SparkSession.builder.appName("SparkDataFrameComparator").getOrCreate()
    logger = setup_logger()

    # Example DataFrames
    df1 = spark.createDataFrame([
        (1, 10.2, "Apple", True, "2023-05-01"),
        (2, 11.0, "Orange", False, "2023-05-02"),
        (3, 12.1, "Banana", True, "2023-05-03"),
    ], ["id", "price", "product", "available", "date_added"])

    df2 = spark.createDataFrame([
        (1, 10.2, "Apple", True, "2023-05-01"),
        (2, 11.0, "Orange", False, "2023-05-02"),
        (3, 13.1, "Banana", False, "2023-05-03"),  # changed price and available
    ], ["id", "price", "product", "available", "date_added"])

    compare_spark_dataframes(
        df1, df2, float_tol=1e-5, save_csv=True,
        csv_path="spark_df_mismatches.csv", logger=logger, limit=1000
    )

    logger.info("Done.")
    spark.stop()

if __name__ == "__main__":
    main()