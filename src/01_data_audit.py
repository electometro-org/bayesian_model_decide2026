#!/usr/bin/env python3

"""
01_data_audit.py

Initial audit of a CSV or Parquet dataset.

Usage
-----
python src/01_data_audit.py data/your_dataset.parquet
python src/01_data_audit.py data/processed/data_preprocessed.csv
"""

from pathlib import Path
import argparse

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit a CSV or Parquet dataset."
    )

    parser.add_argument(
        "file",
        type=Path,
        help="Path to the CSV or Parquet dataset."
    )

    return parser.parse_args()


def main():

    args = parse_args()
    data_file = args.file

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not data_file.exists():
        raise FileNotFoundError(
            f"Data file does not exist: {data_file}"
        )

    if not data_file.is_file():
        raise ValueError(
            f"Input path is not a file: {data_file}"
        )

    suffix = data_file.suffix.lower()

    if suffix not in [".parquet", ".csv"]:
        raise ValueError(
            f"Expected a CSV or Parquet file, got: {data_file}"
        )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("=" * 70)
    print("BAYESIAN REGRESSION PROJECT — DATA AUDIT")
    print("=" * 70)

    print("\nData file:")
    print(f"  {data_file.resolve()}")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    if suffix == ".parquet":
        df = pd.read_parquet(data_file)
    elif suffix == ".csv":
        df = pd.read_csv(data_file)

    print("Data loaded successfully.")

    # --------------------------------------------------------
    # Dataset dimensions
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET DIMENSIONS")
    print("=" * 70)

    n_rows, n_columns = df.shape

    print(f"Rows:    {n_rows:,}")
    print(f"Columns: {n_columns:,}")

    # --------------------------------------------------------
    # First observations
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FIRST 5 OBSERVATIONS")
    print("=" * 70)

    print(df.head().to_string())

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VARIABLES")
    print("=" * 70)

    for number, column in enumerate(df.columns, start=1):
        print(f"{number:3}. {column}")

    # --------------------------------------------------------
    # Data types
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA TYPES")
    print("=" * 70)

    dtype_summary = pd.DataFrame({
        "variable": df.columns,
        "dtype": df.dtypes.astype(str).values,
    })

    print(dtype_summary.to_string(index=False))

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MISSING VALUES")
    print("=" * 70)

    missing_summary = pd.DataFrame({
        "missing": df.isna().sum(),
        "missing_pct": (
            df.isna().mean() * 100
        ).round(2),
    })

    missing_summary = missing_summary.sort_values(
        "missing",
        ascending=False,
    )

    if missing_summary["missing"].sum() == 0:
        print("No missing values detected.")
    else:
        print(
            missing_summary[
                missing_summary["missing"] > 0
            ].to_string()
        )

    # --------------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DUPLICATE ROWS")
    print("=" * 70)

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate rows: {duplicate_count:,}")

    # --------------------------------------------------------
    # Numerical variables
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("NUMERICAL VARIABLES")
    print("=" * 70)

    numeric_columns = (
        df.select_dtypes(include="number")
        .columns
        .tolist()
    )

    print(
        f"Number of numerical variables: "
        f"{len(numeric_columns)}"
    )

    for column in numeric_columns:
        print(f"  - {column}")

    # --------------------------------------------------------
    # Non-numerical variables
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("NON-NUMERICAL VARIABLES")
    print("=" * 70)

    non_numeric_columns = (
        df.select_dtypes(exclude="number")
        .columns
        .tolist()
    )

    print(
        f"Number of non-numerical variables: "
        f"{len(non_numeric_columns)}"
    )

    for column in non_numeric_columns:
        print(f"  - {column}")

    # print(
    #     df["demographics_region"]
    #     .value_counts(dropna=False)
    #     .sort_index()
    #     .to_string()
    # )

    # --------------------------------------------------------
    # Unique values
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("UNIQUE VALUES")
    print("=" * 70)

    unique_summary = pd.DataFrame({
        "variable": df.columns,
        "n_unique": [
            df[column].nunique(dropna=True)
            for column in df.columns
        ],
    })

    print(
        unique_summary
        .sort_values("n_unique")
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Numerical summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("NUMERICAL SUMMARY STATISTICS")
    print("=" * 70)

    if numeric_columns:

        numerical_summary = (
            df[numeric_columns]
            .describe()
            .T
        )

        print(numerical_summary.to_string())

    else:
        print("No numerical variables found.")

    # --------------------------------------------------------
    # Pandas info
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PANDAS INFO")
    print("=" * 70)

    df.info()

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print("\nNo changes were made to the dataset.")


if __name__ == "__main__":
    main()