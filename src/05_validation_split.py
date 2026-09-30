from pathlib import Path

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
TEST_SIZE = 0.20

TARGETS = {
    "gender": "demographics_gender",
    "region": "demographics_region",
    "education": "demographics_education",
    "age": "demographics_age",
}


def load_data(data_file: Path) -> pd.DataFrame:
    """Load a CSV or Parquet dataset."""

    suffix = data_file.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(data_file)

    if suffix == ".csv":
        return pd.read_csv(data_file)

    raise ValueError(
        f"Unsupported file type: {suffix}. "
        "Use .parquet or .csv."
    )


def create_classification_split(
    df: pd.DataFrame,
    target_column: str,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create an 80/20 stratified train/test split.

    Observations with a missing target are excluded before splitting.
    """

    valid_df = df.loc[
        df[target_column].notna()
    ].copy()

    train_df, test_df = train_test_split(
        valid_df,
        test_size=TEST_SIZE,
        random_state=random_seed,
        stratify=valid_df[target_column],
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df.insert(
        0,
        "original_row_index",
        train_df.index,
    )

    test_df.insert(
        0,
        "original_row_index",
        test_df.index,
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df


def create_age_split(
    df: pd.DataFrame,
    target_column: str,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create an 80/20 random train/test split for age.

    Observations with a missing or invalid age below 18 are
    excluded before splitting.
    """

    age = pd.to_numeric(
        df[target_column],
        errors="coerce",
    )

    valid_df = df.loc[
        age.notna() & (age >= 18)
    ].copy()

    train_df, test_df = train_test_split(
        valid_df,
        test_size=TEST_SIZE,
        random_state=random_seed,
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df.insert(
        0,
        "original_row_index",
        train_df.index,
    )

    test_df.insert(
        0,
        "original_row_index",
        test_df.index,
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df


def save_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    target_name: str,
) -> None:
    """Save train and test datasets."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_file = (
        output_dir
        / f"{target_name}_train.csv"
    )

    test_file = (
        output_dir
        / f"{target_name}_test.csv"
    )

    train_df.to_csv(
        train_file,
        index=False,
    )

    test_df.to_csv(
        test_file,
        index=False,
    )

    print(f"Saved training data: {train_file}")
    print(f"Saved test data:     {test_file}")


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Create reproducible train/test splits "
            "for Bayesian model validation."
        )
    )

    parser.add_argument(
        "data_file",
        type=Path,
        help=(
            "Preprocessed CSV or Parquet dataset."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/validation_splits"
        ),
        help=(
            "Directory where train/test datasets "
            "will be saved."
        ),
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Creating validation splits")
    print("=" * 70)

    print(f"\nInput data: {args.data_file}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Test size: {TEST_SIZE:.0%}")

    df = load_data(
        args.data_file
    )

    print(
        f"\nDataset dimensions: "
        f"{df.shape[0]:,} rows × {df.shape[1]} columns"
    )

    for target_name, target_column in TARGETS.items():

        print("\n" + "-" * 70)
        print(f"Target: {target_name}")
        print(f"Column: {target_column}")

        if target_name == "age":
            train_df, test_df = create_age_split(
                df,
                target_column,
                RANDOM_SEED,
            )
        else:
            train_df, test_df = create_classification_split(
                df,
                target_column,
                RANDOM_SEED,
            )

        print(
            f"Valid observations: "
            f"{len(train_df) + len(test_df):,}"
        )

        print(
            f"Training observations: "
            f"{len(train_df):,}"
        )

        print(
            f"Test observations: "
            f"{len(test_df):,}"
        )

        #TODO: Get distributions of unique values in target (train and split)

        save_split(
            train_df,
            test_df,
            args.output_dir,
            target_name,
        )

    print("\n" + "=" * 70)
    print("Validation splits created successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
