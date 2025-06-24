#!/usr/bin/env python3
"""
Helper script to create sample data for testing the data comparison tool.
"""

import pandas as pd
import numpy as np
import os


def create_sample_data():
    """Create sample CSV and Parquet files for testing."""

    # Create sample data
    np.random.seed(42)

    # Before dataset
    before_data = {
        "id": range(1, 1001),
        "name": [f"Person_{i}" for i in range(1, 1001)],
        "age": np.random.randint(18, 80, 1000),
        "salary": np.random.normal(50000, 15000, 1000),
        "department": np.random.choice(["HR", "IT", "Finance", "Marketing"], 1000),
        "active": np.random.choice([True, False], 1000),
        "score": np.random.uniform(0, 100, 1000),
    }

    # After dataset (with some differences)
    # Create deep copy to avoid reference issues
    after_data = {
        "id": list(before_data["id"]),
        "name": list(before_data["name"]),
        "age": before_data["age"].copy(),
        "salary": before_data["salary"].copy(),
        "department": list(before_data["department"]),
        "active": before_data["active"].copy(),
        "score": before_data["score"].copy(),
    }

    # Introduce numerical differences
    after_data["salary"] = after_data["salary"] + np.random.normal(
        0, 100, 1000
    )  # Small salary changes
    after_data["score"] = after_data["score"] + np.random.normal(
        0, 0.5, 1000
    )  # Small score changes

    # Change some names (non-numerical differences)
    for i in range(0, 50):
        after_data["name"][i] = f"Updated_Person_{i+1}"

    # Change some departments (non-numerical differences)
    for i in range(100, 150):
        after_data["department"][i] = "Sales"

    # Add more non-numerical differences
    # Change more names with different patterns
    for i in range(200, 250):
        after_data["name"][i] = f"Modified_{before_data['name'][i]}"

    # Change departments for different ranges
    for i in range(300, 320):
        after_data["department"][i] = "Legal"

    for i in range(400, 430):
        after_data["department"][i] = "Operations"

    # Add some completely different names
    for i in range(500, 520):
        after_data["name"][i] = f"NewEmployee_{i}"

    before_df = pd.DataFrame(before_data)
    after_df = pd.DataFrame(after_data)

    # Create directories
    os.makedirs("sample_data", exist_ok=True)
    os.makedirs("sample_data/csv", exist_ok=True)
    os.makedirs("sample_data/parquet", exist_ok=True)

    # Save as CSV
    before_df.to_csv("sample_data/csv/before_data.csv", index=False)
    after_df.to_csv("sample_data/csv/after_data.csv", index=False)

    # Save as Parquet
    before_df.to_parquet("sample_data/parquet/before_data.parquet", index=False)
    after_df.to_parquet("sample_data/parquet/after_data.parquet", index=False)

    print("Sample data created successfully!")
    print("Files created:")
    print("  - sample_data/csv/before_data.csv")
    print("  - sample_data/csv/after_data.csv")
    print("  - sample_data/parquet/before_data.parquet")
    print("  - sample_data/parquet/after_data.parquet")
    print()
    print("Test commands:")
    print("# Compare CSV files:")
    print(
        "python data_comparison_tool.py --before-path sample_data/csv/before_data.csv --after-path sample_data/csv/after_data.csv"
    )
    print()
    print("# Compare Parquet files:")
    print(
        "python data_comparison_tool.py --before-path sample_data/parquet/before_data.parquet --after-path sample_data/parquet/after_data.parquet"
    )


if __name__ == "__main__":
    create_sample_data()
