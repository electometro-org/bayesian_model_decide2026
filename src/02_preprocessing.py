from pathlib import Path
import argparse
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEMOGRAPHIC_COLUMNS = [
    "demographics_gender",
    "demographics_region",
    "demographics_education",
    "demographics_age",
]
CATEGORICAL_TARGETS = [
    "demographics_gender",
    "demographics_region",
    "demographics_education",
    "demographics_age",
]
POSITION_COLUMNS = [f"responses_t{i}_1" for i in range(1, 21)]
IMPORTANCE_COLUMNS = [f"responses_t{i}_2" for i in range(1, 21)]

# Raw position coding
POSITION_MAP_POSITIONS = {
    0.0: -1,
    0.5: 0,
    1.0: 1,
}
POSITION_MAP_IMPORTANCE = {
    1.0: 0,
    2.0: 1,
}

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def combine_thesis_response(position, importance):
    """
    Combine thesis position and importance into an experimental
    numerical score normalized to [-1, +1].

    Position:
        -1 = oppose
         0 = neutral
        +1 = support

    Importance:
        0 = not important
        1 = important

    Combined score:
        oppose + not important -> -0.5
        oppose + important     -> -1.0
        neutral                ->  0.0
        support + not important -> +0.5
        support + important      -> +1.0
    """

    if pd.isna(position) or pd.isna(importance):
        return pd.NA

    position = int(position)
    importance = int(importance)

    if position not in {-1, 0, 1}:
        return pd.NA

    if importance not in {0, 1}:
        return pd.NA

    if position == 0:
        return 0.0

    if importance == 0:
        return position * 0.5

    return float(position)

def mode_impute_predictor(df, columns): 
    """ Impute missing values in predictor columns using the most 
    frequent observed value (mode). 
    
    The imputation value for each variable is printed and returned for transparency. 
    """ 
    imputation_values = {} 
    
    for column in columns: 
        if column not in df.columns: 
            continue 
        
        missing_before = df[column].isna().sum() 
        
        if missing_before == 0: 
            continue 

        mode_values = df[column].mode(dropna=True) 
            
        if mode_values.empty: 
            raise ValueError( f"Cannot impute {column}: no observed values." ) 
        
        mode_value = mode_values.iloc[0]

        df[column] = df[column].fillna(mode_value)

        imputation_values[column] = { 
            "mode": mode_value, 
            "missing_before": int(missing_before), 
            "missing_after": int(df[column].isna().sum()), 
        } 
        
    return df, imputation_values

def preprocess_data(df, min_demographics_age):
    """
    Apply preprocessing transformations to the dataframe.
    """

    df = df.copy()

    # -----------------------------------------------------
    # 1. Correct demographic data types
    # -----------------------------------------------------

    # demographics_age should be numeric
    df["demographics_age"] = pd.to_numeric(df["demographics_age"], errors="coerce")

    # Demographic categorical variables
    for column in ["demographics_gender", "demographics_region", "demographics_education"]:
        df[column] = df[column].astype("category")

    df["demographics_region"] = (
        df["demographics_region"]
        .replace({
            "Lima": "Lima Metropolitana",
        })
        .astype("string")
    )

    # -----------------------------------------------------
    # 2. Clean implausible demographics_ages
    # -----------------------------------------------------

    invalid_demographics_age = df["demographics_age"].notna() & (df["demographics_age"] < min_demographics_age)

    n_invalid_demographics_age = invalid_demographics_age.sum()

    # Do NOT delete the respondent.
    # Set invalid demographics_age to missing instead.
    df.loc[invalid_demographics_age, "demographics_age"] = pd.NA

    # -----------------------------------------------------
    # 3. Correct survey-response data types
    # -----------------------------------------------------

    for column in POSITION_COLUMNS:
        if column in df.columns:
            df[column] = (
                pd.to_numeric(df[column], errors="coerce")
                .map(POSITION_MAP_POSITIONS)
                .astype("Int64")
                .astype("category")
            )

    for column in IMPORTANCE_COLUMNS:
        if column in df.columns:
            df[column] = (
                pd.to_numeric(df[column], errors="coerce")
                .map(POSITION_MAP_IMPORTANCE)
                .astype("Int64")
                .astype("category")
            )

    # Convert response variables to categorical
    for column in POSITION_COLUMNS + IMPORTANCE_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype("category")

    # -----------------------------------------------------
    # 3.5. Mode imputation for survey predictors
    # -----------------------------------------------------
    predictor_columns = [column for column in (POSITION_COLUMNS + IMPORTANCE_COLUMNS) if column in df.columns ]

    df, imputation_values = mode_impute_predictor(df, predictor_columns)
    

    # -----------------------------------------------------
    # 4. Create combined thesis variables
    # -----------------------------------------------------

    combined_columns = []

    for i in range(1, 21):
        position_column = f"responses_t{i}_1"
        importance_column = f"responses_t{i}_2"
        combined_column = f"responses_t{i}_combined"

        if position_column not in df.columns:
            continue

        if importance_column not in df.columns:
            continue

        df[combined_column] = [
            combine_thesis_response(position, importance)
            for position, importance
            in zip(
                df[position_column],
                df[importance_column]
            )
        ]

        # df[combined_column] = df[combined_column].astype("category")

        combined_columns.append(combined_column)

    # -----------------------------------------------------
    # Return processed data + summary
    # -----------------------------------------------------

    summary = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "invalid_demographics_ages_set_to_missing": int(n_invalid_demographics_age),
        "combined_variables_created": len(combined_columns),
    }

    return df, summary


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Preprocess survey data and save Parquet + CSV."
    )

    parser.add_argument(
        "file",
        type=Path,
        help="Path to the raw Parquet file"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed files "
             "(default: data/processed)"
    )

    parser.add_argument(
        "--min-demographics_age",
        type=int,
        default=18,
        help="Minimum valid demographics_age. demographics_ages below this are set to missing "
             "(default: 18)"
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    if not args.file.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {args.file}"
        )

    if not args.file.is_file():
        raise ValueError(
            f"Input path is not a file: {args.file}"
        )

    if args.file.suffix.lower() != ".parquet":
        raise ValueError(
            "Input file must be a .parquet file."
        )

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    print(f"\nReading: {args.file}")

    df = pd.read_parquet(args.file)

    print(f"Original dimensions: {df.shape}")

    # -----------------------------------------------------
    # Preprocess
    # -----------------------------------------------------

    processed_df, summary = preprocess_data(
        df,
        min_demographics_age=args.min_demographics_age
    )

    # -----------------------------------------------------
    # Create output directory
    # -----------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    parquet_path = (
        args.output_dir / "data_preprocessed.parquet"
    )

    csv_path = (
        args.output_dir / "data_preprocessed.csv"
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    processed_df.to_parquet(
        parquet_path,
        index=False
    )

    processed_df.to_csv(
        csv_path,
        index=False
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\nPreprocessing complete.")
    print("-" * 50)

    print(f"Rows: {summary['n_rows']}")
    print(f"Columns: {summary['n_columns']}")

    print(
        "Invalid demographics_ages set to missing: "
        f"{summary['invalid_demographics_ages_set_to_missing']}"
    )

    print(
        "Combined thesis variables created: "
        f"{summary['combined_variables_created']}"
    )

    print("\nOutput files:")
    print(f"  Parquet: {parquet_path}")
    print(f"  CSV:     {csv_path}")


if __name__ == "__main__":
    main()