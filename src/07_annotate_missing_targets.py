#!/usr/bin/env python3

"""
Annotate a dataset with Bayesian imputations for missing demographic targets.

The script:
    1. Loads a CSV or Parquet dataset.
    2. Loads a previously fitted Bayesian posterior.
    3. Generates predictions for the full dataset.
    4. Preserves the original target values.
    5. Fills only missing target values with Bayesian imputations.
    6. Adds an imputation flag.
    7. Adds uncertainty/confidence information.
    8. Saves an updated CSV or Parquet file.

Important:
    - Observed demographic values are NEVER replaced.
    - The input dataset must already contain the predictor variables in
      the representation expected by the fitted model.
    - This script does not retrain the model.
    - This script does not use target values to update the posterior.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Feature definitions
# ----------------------------------------------------------------------

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

FEATURE_SETS = {
    "position_importance": (
        POSITION_COLUMNS + IMPORTANCE_COLUMNS
    ),
    "combined": COMBINED_COLUMNS,
    "all": (
        POSITION_COLUMNS
        + IMPORTANCE_COLUMNS
        + COMBINED_COLUMNS
    ),
}


# ----------------------------------------------------------------------
# Target class definitions
# ----------------------------------------------------------------------

GENDER_CLASSES = [
    "male",
    "female",
    "diverse",
]


REGION_CLASSES = [
    "Lima Metropolitana",
    "Arequipa",
    "Callao",
    "La Libertad",
    "Extranjero",
    "Cusco",
    "Piura",
    "Lambayeque",
    "Junín",
    "Ica",
    "Áncash",
    "Cajamarca",
    "Lima Provincias",
    "Tacna",
    "Puno",
    "Huánuco",
    "San Martín",
    "Amazonas",
    "Ayacucho",
    "Apurímac",
    "Loreto",
    "Moquegua",
    "Ucayali",
    "Pasco",
    "Madre de Dios",
    "Tumbes",
    "Huancavelica",
]


EDUCATION_CLASSES = [
    "primary",
    "secondary",
    "undergraduate",
    "graduate",
]


# ----------------------------------------------------------------------
# Input/output
# ----------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    """Load CSV or Parquet input."""

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(
        "Unsupported input format. "
        "Use .csv or .parquet."
    )


def save_data(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """Save CSV or Parquet output."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(
            path,
            index=False,
        )
        return

    if suffix in {".parquet", ".pq"}:
        df.to_parquet(
            path,
            index=False,
        )
        return

    raise ValueError(
        "Unsupported output format. "
        "Use .csv or .parquet."
    )


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

def validate_columns(
    df: pd.DataFrame,
    target: str,
    feature_set: str,
) -> list[str]:

    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set: {feature_set}"
        )

    feature_columns = FEATURE_SETS[feature_set]

    required_columns = [
        target_column_name(target),
        *feature_columns,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The input dataset is missing required "
            "model predictor/target columns:\n"
            + "\n".join(missing_columns)
        )

    return feature_columns


def target_column_name(target: str) -> str:

    if target in {
        "gender",
        "region",
        "education",
        "age",
    }:
        return f"demographics_{target}"

    raise ValueError(
        f"Unsupported target: {target}"
    )


# ----------------------------------------------------------------------
# Posterior helpers
# ----------------------------------------------------------------------

def posterior_array(
    idata: az.InferenceData,
    variable: str,
) -> np.ndarray:
    """
    Extract posterior samples and flatten chain/draw dimensions.

    Returns:
        samples × remaining dimensions
    """

    if not hasattr(idata, "posterior"):
        raise ValueError(
            "The model file does not contain a posterior group."
        )

    if variable not in idata.posterior:
        raise ValueError(
            f"Posterior variable '{variable}' "
            "was not found in the model file."
        )

    values = (
        idata.posterior[variable]
        .values
    )

    if values.ndim < 2:
        raise ValueError(
            f"Unexpected posterior shape for "
            f"'{variable}': {values.shape}"
        )

    n_chains = values.shape[0]
    n_draws = values.shape[1]

    return values.reshape(
        n_chains * n_draws,
        *values.shape[2:],
    )


# ----------------------------------------------------------------------
# Classification predictions
# ----------------------------------------------------------------------

def softmax(
    logits: np.ndarray,
) -> np.ndarray:

    logits = (
        logits
        - np.max(
            logits,
            axis=-1,
            keepdims=True,
        )
    )

    exp_logits = np.exp(logits)

    return (
        exp_logits
        / exp_logits.sum(
            axis=-1,
            keepdims=True,
        )
    )


def classification_predictions(
    idata: az.InferenceData,
    X: np.ndarray,
    target: str,
) -> tuple[np.ndarray, np.ndarray]:

    if target == "gender":
        classes = GENDER_CLASSES

    elif target == "region":
        classes = REGION_CLASSES

    else:
        raise ValueError(
            "classification_predictions only supports "
            "gender and region."
        )

    alpha = posterior_array(
        idata,
        "alpha",
    )

    beta = posterior_array(
        idata,
        "beta",
    )

    # beta:
    # posterior_samples × classes_without_reference × predictors

    # alpha:
    # posterior_samples × classes_without_reference

    eta = (
        alpha[:, None, :]
        + np.einsum(
            "spk,nk->spn",
            beta,
            X,
        )
    )

    n_samples = eta.shape[0]
    n_observations = X.shape[0]
    n_classes = len(classes)

    logits = np.zeros(
        (
            n_samples,
            n_observations,
            n_classes,
        )
    )

    logits[:, :, 1:] = (
        np.transpose(
            eta,
            (0, 2, 1),
        )
    )

    probabilities = softmax(
        logits
    )

    mean_probabilities = (
        probabilities.mean(axis=0)
    )

    predictions = np.argmax(
        mean_probabilities,
        axis=1,
    )

    confidence = np.max(
        mean_probabilities,
        axis=1,
    )

    return (
        predictions,
        confidence,
    )


# ----------------------------------------------------------------------
# Education predictions
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (
        1.0 + np.exp(-x)
    )


def education_predictions(
    idata: az.InferenceData,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    beta = posterior_array(
        idata,
        "beta",
    )

    cutpoints = posterior_array(
        idata,
        "cutpoints",
    )

    eta = np.einsum(
        "sp,n p->sn",
        beta,
        X,
    )

    # The expression above contains a space in the
    # einsum notation for readability in source.
    # Recalculate using the standard notation.
    eta = np.einsum(
        "sp,np->sn",
        beta,
        X,
    )

    c1 = cutpoints[:, 0][:, None]
    c2 = cutpoints[:, 1][:, None]
    c3 = cutpoints[:, 2][:, None]

    p0 = sigmoid(c1 - eta)
    p1 = sigmoid(c2 - eta) - p0
    p2 = sigmoid(c3 - eta) - sigmoid(c2 - eta)
    p3 = 1.0 - sigmoid(c3 - eta)

    probabilities = np.stack(
        [p0, p1, p2, p3],
        axis=-1,
    )

    mean_probabilities = (
        probabilities.mean(axis=0)
    )

    predictions = np.argmax(
        mean_probabilities,
        axis=1,
    )

    confidence = np.max(
        mean_probabilities,
        axis=1,
    )

    return (
        predictions,
        confidence,
    )


# ----------------------------------------------------------------------
# Age predictions
# ----------------------------------------------------------------------

def age_predictions(
    idata: az.InferenceData,
    X: np.ndarray,
    nonlinear: bool,
) -> tuple[np.ndarray, np.ndarray]:

    alpha = posterior_array(
        idata,
        "alpha",
    )

    beta = posterior_array(
        idata,
        "beta",
    )

    mu = (
        alpha[:, None]
        + np.einsum(
            "sp,np->sn",
            beta,
            X,
        )
    )

    if nonlinear:

        gamma = posterior_array(
            idata,
            "gamma",
        )

        X_squared = X ** 2

        mu += np.einsum(
            "sp,np->sn",
            gamma,
            X_squared,
        )

    # Posterior mean of latent regression mean.
    predicted_age = (
        mu.mean(axis=0)
    )

    posterior_sd = (
        mu.std(axis=0)
    )

    # Age is an integer demographic variable.
    predicted_age = np.rint(
        predicted_age
    )

    # Keep predictions within the valid
    # age range used by the project.
    predicted_age = np.clip(
        predicted_age,
        18,
        99,
    )

    return (
        predicted_age,
        posterior_sd,
    )


# ----------------------------------------------------------------------
# Annotation
# ----------------------------------------------------------------------

def annotate_classification(
    df: pd.DataFrame,
    target: str,
    predictions: np.ndarray,
    confidence: np.ndarray,
) -> None:

    column = target_column_name(
        target
    )

    original_column = (
        f"{column}_original"
    )

    imputed_column = (
        f"{column}_imputed"
    )

    flag_column = (
        f"{column}_was_imputed"
    )

    confidence_column = (
        f"{column}_confidence"
    )

    # Preserve original values exactly.
    df[original_column] = df[column]

    missing = df[column].isna()

    df[imputed_column] = df[column]

    if target == "gender":
        classes = GENDER_CLASSES

    elif target == "region":
        classes = REGION_CLASSES

    elif target == "education":
        classes = EDUCATION_CLASSES

    else:
        raise ValueError(
            f"Unsupported classification target: {target}"
        )

    predicted_labels = np.asarray(
        [
            classes[index]
            for index in predictions
        ],
        dtype=object,
    )

    df.loc[
        missing,
        imputed_column,
    ] = predicted_labels[missing]

    df[flag_column] = (
        missing.astype(int)
    )

    df[confidence_column] = np.nan

    df.loc[
        missing,
        confidence_column,
    ] = confidence[missing]


def annotate_age(
    df: pd.DataFrame,
    predictions: np.ndarray,
    uncertainty: np.ndarray,
) -> None:

    column = "demographics_age"

    original_column = (
        "demographics_age_original"
    )

    imputed_column = (
        "demographics_age_imputed"
    )

    flag_column = (
        "demographics_age_was_imputed"
    )

    uncertainty_column = (
        "demographics_age_posterior_sd"
    )

    # Preserve original age.
    df[original_column] = df[column]

    missing = (
        pd.to_numeric(
            df[column],
            errors="coerce",
        ).isna()
    )

    df[imputed_column] = df[column]

    df.loc[
        missing,
        imputed_column,
    ] = predictions[missing]

    df[flag_column] = (
        missing.astype(int)
    )

    df[uncertainty_column] = np.nan

    df.loc[
        missing,
        uncertainty_column,
    ] = uncertainty[missing]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Annotate a dataset with Bayesian "
            "imputations for missing demographic targets."
        )
    )

    parser.add_argument(
        "data_file",
        type=Path,
        help="Input CSV or Parquet file.",
    )

    parser.add_argument(
        "--model-file",
        required=True,
        type=Path,
        help="Saved Bayesian posterior (.nc).",
    )

    parser.add_argument(
        "--target",
        required=True,
        choices=[
            "gender",
            "region",
            "education",
            "age",
        ],
        help="Target to annotate.",
    )

    parser.add_argument(
        "--features",
        required=True,
        choices=[
            "position_importance",
            "combined",
            "all",
        ],
        help="Feature representation used by the model.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV or Parquet file.",
    )

    args = parser.parse_args()

    print(
        "======================================================================"
    )
    print(
        "Bayesian target annotation"
    )
    print(
        "======================================================================"
    )

    print(
        f"Input:       {args.data_file}"
    )
    print(
        f"Model:       {args.model_file}"
    )
    print(
        f"Target:      {args.target}"
    )
    print(
        f"Features:    {args.features}"
    )
    print(
        f"Output:      {args.output}"
    )

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    df = load_data(
        args.data_file
    )

    print(
        f"\nObservations: {len(df):,}"
    )

    feature_columns = validate_columns(
        df,
        args.target,
        args.features,
    )

    # ------------------------------------------------------------------
    # Prepare predictors
    # ------------------------------------------------------------------

    X = (
        df[feature_columns]
        .astype(float)
        .to_numpy()
    )

    print(
        f"Predictors:   {X.shape[1]}"
    )

    # ------------------------------------------------------------------
    # Load posterior
    # ------------------------------------------------------------------

    print(
        "\nLoading posterior..."
    )

    idata = az.from_netcdf(
        args.model_file
    )

    if not hasattr(idata, "posterior"):
        raise ValueError(
            "Model file does not contain "
            "a posterior group."
        )

    print(
        "Posterior loaded."
    )

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    if args.target in {
        "gender",
        "region",
    }:

        predictions, confidence = (
            classification_predictions(
                idata,
                X,
                args.target,
            )
        )

        annotate_classification(
            df,
            args.target,
            predictions,
            confidence,
        )

    elif args.target == "education":

        predictions, confidence = (
            education_predictions(
                idata,
                X,
            )
        )

        annotate_classification(
            df,
            args.target,
            predictions,
            confidence,
        )

    elif args.target == "age":

        nonlinear = (
            "gamma"
            in idata.posterior
        )

        predictions, uncertainty = (
            age_predictions(
                idata,
                X,
                nonlinear,
            )
        )

        annotate_age(
            df,
            predictions,
            uncertainty,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    target_column = target_column_name(
        args.target
    )

    flag_column = (
        f"{target_column}_was_imputed"
    )

    n_imputed = int(
        df[flag_column].sum()
    )

    n_observed = (
        len(df) - n_imputed
    )

    print(
        "\nAnnotation summary:"
    )

    print(
        f"  Observed target values: {n_observed:,}"
    )

    print(
        f"  Model-imputed values:   {n_imputed:,}"
    )

    if n_imputed > 0:

        percentage = (
            100.0
            * n_imputed
            / len(df)
        )

        print(
            f"  Imputed percentage:      {percentage:.2f}%"
        )

    else:

        print(
            "  No missing target values "
            "required imputation."
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    save_data(
        df,
        args.output,
    )

    print(
        "\nSaved annotated dataset:"
    )

    print(
        f"  {args.output}"
    )

    print(
        "======================================================================"
    )


if __name__ == "__main__":
    main()