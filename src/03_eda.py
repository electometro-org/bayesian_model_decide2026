# src/03_eda.py

from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Configuration
# ============================================================

DEMOGRAPHIC_COLUMNS = [
    "demographics_gender",
    "demographics_region",
    "demographics_education",
    "demographics_age",
]

POSITION_COLUMNS = [
    f"responses_t{i}_1"
    for i in range(1, 21)
]

IMPORTANCE_COLUMNS = [
    f"responses_t{i}_2"
    for i in range(1, 21)
]

COMBINED_COLUMNS = [
    f"responses_t{i}_combined"
    for i in range(1, 21)
]


# ============================================================
# Helpers
# ============================================================

def load_data(data_file: Path) -> pd.DataFrame:
    """Load CSV or Parquet input."""

    suffix = data_file.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(data_file)

    if suffix == ".csv":
        return pd.read_csv(data_file)

    raise ValueError(
        f"Unsupported file type: {suffix}. "
        "Please provide a .csv or .parquet file."
    )


def save_bar_plot(
    counts: pd.Series,
    title: str,
    output_file: Path,
    xlabel: str = "",
    ylabel: str = "Count",
    rotation: int = 0,
):
    """Create and save a bar plot from value counts."""

    fig, ax = plt.subplots(figsize=(10, 6))

    counts.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if rotation is not None:
        ax.tick_params(axis="x", rotation=rotation)

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def save_histogram(
    series: pd.Series,
    title: str,
    output_file: Path,
    xlabel: str,
):
    """Create and save a histogram."""

    fig, ax = plt.subplots(figsize=(10, 6))

    series.dropna().plot(
        kind="hist",
        bins=30,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def save_boxplot(
    series: pd.Series,
    title: str,
    output_file: Path,
    ylabel: str,
):
    """Create and save a boxplot."""

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.boxplot(series.dropna())

    ax.set_title(title)
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


# ============================================================
# Target EDA
# ============================================================

def analyze_categorical_target(
    df: pd.DataFrame,
    column: str,
    output_dir: Path,
):
    """Analyze a categorical demographic target."""

    counts = (
        df[column]
        .dropna()
        .value_counts()
    )

    proportions = (
        df[column]
        .dropna()
        .value_counts(normalize=True)
        .rename("proportion")
    )

    summary = pd.concat(
        [
            counts.rename("count"),
            proportions,
        ],
        axis=1,
    )

    summary.to_csv(
        output_dir / f"{column}_summary.csv"
    )

    save_bar_plot(
        counts=counts,
        title=f"Distribution of {column}",
        output_file=output_dir / f"{column}_distribution.png",
        xlabel=column,
        rotation=45,
    )

def analyze_combined_by_categorical_target(
    df: pd.DataFrame,
    target: str,
    output_dir: Path,
):
    """
    Analyze combined thesis responses conditional on a categorical target.

    Produces:
        - one CSV per thesis
        - one overall CSV containing all theses
        - one heatmap per thesis
    """

    target_dir = (
        output_dir
        / "combined_by_targets"
        / target
    )

    target_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    for column in COMBINED_COLUMNS:

        temp = df[
            [target, column]
        ].dropna()

        # Proportion of each combined response within target category
        proportions = pd.crosstab(
            temp[target],
            temp[column],
            normalize="index",
        )

        proportions = proportions.reindex(
            columns=[
                -1.0,
                -0.5,
                0.0,
                0.5,
                1.0,
            ],
            fill_value=0,
        )

        proportions.columns = [
            "oppose_important_-1",
            "oppose_not_important_-0.5",
            "neutral_0",
            "support_not_important_0.5",
            "support_important_1",
        ]

        proportions.insert(
            0,
            "variable",
            column,
        )

        proportions.to_csv(
            target_dir / f"{column}_by_{target}.csv"
        )

        all_results.append(proportions)

        # ----------------------------------------------------
        # Heatmap
        # ----------------------------------------------------

        heatmap_data = proportions.drop(
            columns="variable"
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        image = ax.imshow(
            heatmap_data.values,
            aspect="auto",
            interpolation="nearest",
        )

        ax.set_title(
            f"{column} by {target}"
        )

        ax.set_xlabel(
            "Combined response category"
        )

        ax.set_ylabel(
            target
        )

        ax.set_xticks(
            range(len(heatmap_data.columns))
        )

        ax.set_xticklabels(
            [
                "-1",
                "-0.5",
                "0",
                "+0.5",
                "+1",
            ]
        )

        ax.set_yticks(
            range(len(heatmap_data.index))
        )

        ax.set_yticklabels(
            heatmap_data.index
        )

        for i in range(heatmap_data.shape[0]):
            for j in range(heatmap_data.shape[1]):

                value = heatmap_data.iloc[i, j]

                ax.text(
                    j,
                    i,
                    f"{value:.1%}",
                    ha="center",
                    va="center",
                )

        fig.colorbar(
            image,
            ax=ax,
            label="Proportion",
        )

        fig.tight_layout()

        fig.savefig(
            target_dir / f"{column}_by_{target}.png",
            dpi=150,
        )

        plt.close(fig)

    # --------------------------------------------------------
    # Combined summary
    # --------------------------------------------------------

    combined = pd.concat(
        all_results,
        axis=0,
    )

    combined.to_csv(
        target_dir
        / f"all_combined_responses_by_{target}.csv"
    )

def analyze_combined_by_age(
    df: pd.DataFrame,
    output_dir: Path,
):
    """
    Analyze age conditional on each combined response category.
    """

    age_dir = (
        output_dir
        / "combined_by_targets"
        / "demographics_age"
    )

    age_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    for column in COMBINED_COLUMNS:

        temp = df[
            [column, "demographics_age"]
        ].dropna()

        summary = (
            temp
            .groupby(column)["demographics_age"]
            .agg(
                count="count",
                mean="mean",
                median="median",
                std="std",
            )
            .reindex(
                [-1.0, -0.5, 0.0, 0.5, 1.0]
            )
        )

        summary.index.name = "combined_score"
        summary.insert(
            0,
            "variable",
            column,
        )

        summary.to_csv(
            age_dir
            / f"{column}_by_age.csv"
        )

        all_results.append(summary)

        # ----------------------------------------------------
        # Age plot
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

        age_groups = [
            temp.loc[
                temp[column] == value,
                "demographics_age",
            ]
            for value in [-1.0, -0.5, 0.0, 0.5, 1.0]
        ]

        ax.boxplot(
            age_groups,
            tick_labels=[
                "-1",
                "-0.5",
                "0",
                "+0.5",
                "+1",
            ],
        )

        ax.set_title(
            f"Age by {column}"
        )

        ax.set_xlabel(
            "Combined response"
        )

        ax.set_ylabel(
            "Age"
        )

        fig.tight_layout()

        fig.savefig(
            age_dir
            / f"{column}_by_age.png",
            dpi=150,
        )

        plt.close(fig)

    combined = pd.concat(
        all_results,
        axis=0,
    )

    combined.to_csv(
        age_dir
        / "all_combined_responses_by_age.csv"
    )

def analyze_age(
    df: pd.DataFrame,
    output_dir: Path,
):
    """Analyze age target."""

    age = df["demographics_age"].dropna()

    summary = age.describe()
    summary.to_csv(
        output_dir / "demographics_age_summary.csv"
    )

    save_histogram(
        series=age,
        title="Distribution of Age",
        output_file=output_dir / "demographics_age_histogram.png",
        xlabel="Age",
    )

    save_boxplot(
        series=age,
        title="Age Boxplot",
        output_file=output_dir / "demographics_age_boxplot.png",
        ylabel="Age",
    )


def analyze_targets(
    df: pd.DataFrame,
    output_dir: Path,
):
    """Run EDA for demographic targets."""

    target_dir = output_dir / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)

    for column in [
        "demographics_gender",
        "demographics_region",
        "demographics_education",
    ]:
        analyze_categorical_target(
            df=df,
            column=column,
            output_dir=target_dir,
        )

    analyze_age(
        df=df,
        output_dir=target_dir,
    )


# ============================================================
# Survey response EDA
# ============================================================

def analyze_position_responses(
    df: pd.DataFrame,
    output_dir: Path,
):
    """Analyze thesis position variables."""

    position_dir = output_dir / "responses_position"
    position_dir.mkdir(parents=True, exist_ok=True)

    all_distributions = []

    for column in POSITION_COLUMNS:

        counts = (
            df[column]
            .value_counts()
            .reindex([-1, 0, 1], fill_value=0)
        )

        proportions = counts / counts.sum()

        summary = pd.DataFrame(
            {
                "count": counts,
                "proportion": proportions,
            }
        )

        summary.to_csv(
            position_dir / f"{column}_summary.csv"
        )

        save_bar_plot(
            counts=counts,
            title=f"Distribution of {column}",
            output_file=position_dir / f"{column}_distribution.png",
            xlabel="Position",
        )

        temp = proportions.rename(column)
        all_distributions.append(temp)

    distribution_table = pd.concat(
        all_distributions,
        axis=1,
    ).T

    distribution_table.columns = [
        "oppose_-1",
        "neutral_0",
        "support_1",
    ]

    distribution_table.to_csv(
        position_dir / "all_position_distributions.csv"
    )


def analyze_importance_responses(
    df: pd.DataFrame,
    output_dir: Path,
):
    """Analyze importance variables."""

    importance_dir = output_dir / "responses_importance"
    importance_dir.mkdir(parents=True, exist_ok=True)

    all_distributions = []

    for column in IMPORTANCE_COLUMNS:

        counts = (
            df[column]
            .value_counts()
            .reindex([0, 1], fill_value=0)
        )

        proportions = counts / counts.sum()

        summary = pd.DataFrame(
            {
                "count": counts,
                "proportion": proportions,
            }
        )

        summary.to_csv(
            importance_dir / f"{column}_summary.csv"
        )

        save_bar_plot(
            counts=counts,
            title=f"Distribution of {column}",
            output_file=importance_dir / f"{column}_distribution.png",
            xlabel="Importance",
        )

        temp = proportions.rename(column)
        all_distributions.append(temp)

    distribution_table = pd.concat(
        all_distributions,
        axis=1,
    ).T

    distribution_table.columns = [
        "not_important_0",
        "important_1",
    ]

    distribution_table.to_csv(
        importance_dir / "all_importance_distributions.csv"
    )


def analyze_combined_responses(
    df: pd.DataFrame,
    output_dir: Path,
):
    """Analyze experimental combined response variables."""

    combined_dir = output_dir / "responses_combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    expected_values = [
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
    ]

    all_distributions = []
    summary_rows = []

    for column in COMBINED_COLUMNS:

        series = df[column].dropna()

        counts = (
            series
            .value_counts()
            .reindex(expected_values, fill_value=0)
        )

        proportions = counts / counts.sum()

        # ----------------------------------------------------
        # Full distribution
        # ----------------------------------------------------

        summary = pd.DataFrame(
            {
                "count": counts,
                "proportion": proportions,
            }
        )

        summary.to_csv(
            combined_dir / f"{column}_summary.csv"
        )

        # ----------------------------------------------------
        # Distribution plot
        # ----------------------------------------------------

        save_bar_plot(
            counts=counts,
            title=f"Distribution of {column}",
            output_file=combined_dir / f"{column}_distribution.png",
            xlabel="Combined score",
        )

        # ----------------------------------------------------
        # Store distribution for heatmap
        # ----------------------------------------------------

        all_distributions.append(
            proportions.rename(column)
        )

        # ----------------------------------------------------
        # Summary statistics
        # ----------------------------------------------------

        summary_rows.append(
            {
                "variable": column,
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std(),
                "proportion_negative": (series < 0).mean(),
                "proportion_neutral": (series == 0).mean(),
                "proportion_positive": (series > 0).mean(),
                "proportion_oppose_important": (
                    series == -1.0
                ).mean(),
                "proportion_support_important": (
                    series == 1.0
                ).mean(),
            }
        )

    # ========================================================
    # Combined distribution table
    # ========================================================

    distribution_table = pd.concat(
        all_distributions,
        axis=1,
    ).T

    distribution_table.columns = [
        "oppose_important_-1",
        "oppose_not_important_-0.5",
        "neutral_0",
        "support_not_important_0.5",
        "support_important_1",
    ]

    distribution_table.to_csv(
        combined_dir / "all_combined_distributions.csv"
    )

    # ========================================================
    # Summary statistics table
    # ========================================================

    summary_table = pd.DataFrame(
        summary_rows
    )

    summary_table.to_csv(
        combined_dir / "combined_summary_statistics.csv",
        index=False,
    )

    # ========================================================
    # Heatmap
    # ========================================================

    heatmap_data = distribution_table.copy()

    heatmap_data.columns = [
        "-1\nOppose + Important",
        "-0.5\nOppose",
        "0\nNeutral",
        "+0.5\nSupport",
        "+1\nSupport + Important",
    ]

    fig, ax = plt.subplots(
        figsize=(11, 9)
    )

    image = ax.imshow(
        heatmap_data.values,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_title(
        "Distribution of Combined Thesis Responses"
    )

    ax.set_xlabel(
        "Combined response category"
    )

    ax.set_ylabel(
        "Thesis response"
    )

    ax.set_xticks(
        range(len(heatmap_data.columns))
    )

    ax.set_xticklabels(
        heatmap_data.columns
    )

    ax.set_yticks(
        range(len(heatmap_data.index))
    )

    ax.set_yticklabels(
        [
            f"T{i}"
            for i in range(1, 21)
        ]
    )

    # Add percentage values to cells
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):

            value = heatmap_data.iloc[i, j]

            ax.text(
                j,
                i,
                f"{value:.1%}",
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Proportion",
    )

    fig.tight_layout()

    fig.savefig(
        combined_dir / "combined_distribution_heatmap.png",
        dpi=150,
    )

    plt.close(fig)

# ============================================================
# Missingness EDA
# ============================================================

def analyze_missingness(
    df: pd.DataFrame,
    output_dir: Path,
):
    """Analyze missing values across all variables."""

    missing_dir = output_dir / "missingness"
    missing_dir.mkdir(parents=True, exist_ok=True)

    missing_count = df.isna().sum()
    missing_proportion = df.isna().mean()

    summary = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_proportion": missing_proportion,
        }
    ).sort_values(
        "missing_proportion",
        ascending=False,
    )

    summary.to_csv(
        missing_dir / "missingness_summary.csv"
    )

    # Only plot variables with at least one missing value.
    plot_data = summary[
        summary["missing_count"] > 0
    ].sort_values(
        "missing_proportion",
        ascending=True,
    )

    if not plot_data.empty:

        fig, ax = plt.subplots(
            figsize=(10, max(6, len(plot_data) * 0.25))
        )

        ax.barh(
            plot_data.index,
            plot_data["missing_proportion"] * 100,
        )

        ax.set_title("Missingness by Variable")
        ax.set_xlabel("Missing values (%)")
        ax.set_ylabel("Variable")

        fig.tight_layout()
        fig.savefig(
            missing_dir / "missingness_by_variable.png",
            dpi=150,
        )

        plt.close(fig)


# ============================================================
# Overall numerical summaries
# ============================================================

def analyze_numerical_variables(
    df: pd.DataFrame,
    output_dir: Path,
):
    """Save numerical descriptive statistics."""

    numerical_dir = output_dir / "numerical_summary"
    numerical_dir.mkdir(parents=True, exist_ok=True)

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns

    summary = df[numerical_columns].describe().T

    summary.to_csv(
        numerical_dir / "numerical_descriptive_statistics.csv"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Exploratory Data Analysis for the preprocessed dataset."
    )

    parser.add_argument(
        "data_file",
        type=Path,
        help="Path to the preprocessed CSV or Parquet file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/eda"),
        help="Directory where EDA outputs will be saved.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    print(f"\nInput file: {args.data_file}")
    print(f"Output directory: {args.output_dir}")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data(args.data_file)

    print(f"\nDataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Run EDA
    # --------------------------------------------------------

    print("\n[1/10] Analyzing demographic targets...")
    analyze_targets(
        df,
        args.output_dir,
    )

    print("[2/10] Analyzing position responses...")
    analyze_position_responses(
        df,
        args.output_dir,
    )

    print("[3/10] Analyzing importance responses...")
    analyze_importance_responses(
        df,
        args.output_dir,
    )

    print("[4/10] Analyzing combined responses...")
    analyze_combined_responses(
        df,
        args.output_dir,
    )

    print("[5/10] Analyzing missingness...")
    analyze_missingness(
        df,
        args.output_dir,
    )

    print("[6/10] Creating numerical summaries...")
    analyze_numerical_variables(
        df,
        args.output_dir,
    )

    print("[7/10] Analyzing combined responses by gender...")
    analyze_combined_by_categorical_target(
        df,
        "demographics_gender",
        args.output_dir,
    )

    print("[8/10] Analyzing combined responses by education...")
    analyze_combined_by_categorical_target(
        df,
        "demographics_education",
        args.output_dir,
    )

    print("[9/10] Analyzing combined responses by region...")
    analyze_combined_by_categorical_target(
        df,
        "demographics_region",
        args.output_dir,
    )

    print("[10/10] Analyzing combined responses by age...")
    analyze_combined_by_age(
        df,
        args.output_dir,
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)

    print(f"\nResults saved to:")
    print(args.output_dir.resolve())

    print("\nGenerated structure:")
    print(
        """
    results/
    └── eda/
        ├── targets/
        ├── responses_position/
        ├── responses_importance/
        ├── responses_combined/
        ├── missingness/
        └── numerical_summary/
            """
    )


if __name__ == "__main__":
    main()