from pathlib import Path
import argparse
import importlib.util

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)


# ---------------------------------------------------------------------
# Model module
# ---------------------------------------------------------------------

def load_model_module():
    """
    Load the model definitions from 04_bayesian_models.py.

    The file name starts with a number, so it is loaded dynamically
    rather than imported with a normal Python import statement.
    """

    model_file = (
        Path(__file__).resolve().parent
        / "04_bayesian_models.py"
    )

    spec = importlib.util.spec_from_file_location(
        "bayesian_models",
        model_file,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load model module from {model_file}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_data(data_file: Path) -> pd.DataFrame:
    """Load a CSV or Parquet dataset."""

    suffix = data_file.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(data_file)

    if suffix == ".parquet":
        return pd.read_parquet(data_file)

    raise ValueError(
        f"Unsupported file type: {suffix}. "
        "Use .csv or .parquet."
    )


# ---------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------

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
        POSITION_COLUMNS
        + IMPORTANCE_COLUMNS
    ),
    "combined": COMBINED_COLUMNS,
    "all": (
        POSITION_COLUMNS
        + IMPORTANCE_COLUMNS
        + COMBINED_COLUMNS
    ),
}


# ---------------------------------------------------------------------
# Target definitions
# ---------------------------------------------------------------------

TARGET_COLUMNS = {
    "gender": "demographics_gender",
    "region": "demographics_region",
    "education": "demographics_education",
    "age": "demographics_age",
}


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


# ---------------------------------------------------------------------
# Likelihood names
# ---------------------------------------------------------------------

def get_likelihood_name(model_name: str) -> str:
    """
    Return the name of the observed likelihood variable in the
    Bayesian model.

    The age models are named age_linear and age_nonlinear, but their
    likelihood variable is simply named 'age'.
    """

    if model_name in {
        "gender",
        "region",
        "education",
    }:
        return model_name

    if model_name in {
        "age_linear",
        "age_nonlinear",
    }:
        return "age"

    raise ValueError(
        f"Unknown model name: {model_name}"
    )


# ---------------------------------------------------------------------
# Prediction preparation
# ---------------------------------------------------------------------

def prepare_test_data(
    df: pd.DataFrame,
    target: str,
    feature_set: str,
):
    """
    Prepare test predictors and observed target.

    The test files have already had invalid/missing target
    observations removed by 05_validation_split.py.
    """

    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set: {feature_set}"
        )

    if target not in TARGET_COLUMNS:
        raise ValueError(
            f"Unknown target: {target}"
        )

    target_column = TARGET_COLUMNS[target]
    feature_columns = FEATURE_SETS[feature_set]

    required_columns = [
        target_column,
        *feature_columns,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    model_df = df[
        required_columns
    ].copy()

    X = (
        model_df[feature_columns]
        .astype(float)
        .to_numpy()
    )

    if target == "age":

        y = pd.to_numeric(
            model_df[target_column],
            errors="coerce",
        ).to_numpy(dtype=float)

    else:

        y = model_df[
            target_column
        ].astype("category").cat.codes.to_numpy()

    return X, y


# ---------------------------------------------------------------------
# Target encoding
# ---------------------------------------------------------------------

def encode_target(
    values: pd.Series,
    classes: list[str],
) -> np.ndarray:
    """
    Encode categorical target values using the exact class ordering
    used by the Bayesian model.
    """

    mapping = {
        value: index
        for index, value in enumerate(classes)
    }

    encoded = values.map(mapping)

    if encoded.isna().any():

        unknown = values[
            encoded.isna()
        ].unique()

        raise ValueError(
            "Test data contains target categories that are "
            f"not present in the model classes: {unknown}"
        )

    return encoded.to_numpy(dtype=int)


def prepare_classification_test_data(
    df: pd.DataFrame,
    target: str,
    feature_set: str,
):
    """Prepare classification test data using model class ordering."""

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

    target_column = TARGET_COLUMNS[target]
    feature_columns = FEATURE_SETS[feature_set]

    required_columns = [
        target_column,
        *feature_columns,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    X = (
        df[feature_columns]
        .astype(float)
        .to_numpy()
    )

    y = encode_target(
        df[target_column].astype(str),
        classes,
    )

    return X, y, classes


# ---------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------

def build_model(
    model_module,
    target: str,
    X: np.ndarray,
    y: np.ndarray,
):
    """
    Build the same model structure used during training.

    The test predictors are used for the model's X data. The saved
    posterior is then used directly for posterior predictive
    simulation.

    The y argument is required because the model-building functions
    in 04_bayesian_models.py define the likelihood with an observed
    variable.
    """

    if target == "gender":

        return model_module.build_gender_model(
            X,
            y,
        )

    if target == "region":

        return model_module.build_region_model(
            X,
            y,
        )

    if target == "education":

        return model_module.build_education_model(
            X,
            y,
        )

    if target == "age_linear":

        return model_module.build_age_linear_model(
            X,
            y,
        )

    if target == "age_nonlinear":

        return model_module.build_age_nonlinear_model(
            X,
            y,
        )

    raise ValueError(
        f"Unknown target/model: {target}"
    )


# ---------------------------------------------------------------------
# Posterior predictive sampling
# ---------------------------------------------------------------------

def generate_test_predictions(
    idata: az.InferenceData,
    model: pm.Model,
    likelihood_name: str,
    random_seed: int,
):
    """
    Generate posterior predictive samples for the held-out test set.

    return_inferencedata=False is used deliberately here. It gives us
    the raw posterior predictive dictionary directly and avoids
    dependence on the posterior_predictive InferenceData group,
    which is causing problems with the installed ArviZ/PyMC versions.
    """

    print(
        f"\nGenerating posterior predictive samples "
        f"for '{likelihood_name}'..."
    )

    with model:
        predictions = pm.sample_posterior_predictive(
            idata.posterior,
            var_names=[likelihood_name],
            random_seed=random_seed,
            return_inferencedata=False,
        )

    # ---------------------------------------------------------------
    # Debugging information
    # ---------------------------------------------------------------

    print("\nPosterior predictive object:")
    print(
        f"  Type: {type(predictions).__name__}"
    )

    if not isinstance(predictions, dict):

        raise TypeError(
            "Expected posterior predictive samples as a "
            f"dictionary, but received: {type(predictions)}"
        )

    print(
        f"  Variables: {list(predictions.keys())}"
    )

    if likelihood_name not in predictions:

        raise ValueError(
            f"Posterior predictive samples do not contain "
            f"'{likelihood_name}'. "
            f"Available variables: {list(predictions.keys())}"
        )

    prediction_array = np.asarray(
        predictions[likelihood_name]
    )

    print(
        f"  '{likelihood_name}' raw shape: "
        f"{prediction_array.shape}"
    )

    print(
        f"  '{likelihood_name}' dtype: "
        f"{prediction_array.dtype}"
    )

    return prediction_array


# ---------------------------------------------------------------------
# Prediction array formatting
# ---------------------------------------------------------------------

def flatten_posterior_samples(
    predictions: np.ndarray,
    n_observations: int,
) -> np.ndarray:
    """
    Convert posterior predictive samples into:

        observations × posterior samples

    PyMC commonly returns:

        chains × draws × observations

    This function also handles a two-dimensional
    draws × observations result.
    """

    predictions = np.asarray(
        predictions
    )

    print(
        f"\nFormatting prediction array:"
        f"\n  Input shape: {predictions.shape}"
    )

    if predictions.ndim == 3:

        # Expected:
        # chains × draws × observations

        if predictions.shape[2] != n_observations:

            raise ValueError(
                "Unexpected posterior predictive shape. "
                f"Expected the final dimension to contain "
                f"{n_observations} observations, but received "
                f"{predictions.shape}."
            )

        n_chains = predictions.shape[0]
        n_draws = predictions.shape[1]

        predictions = predictions.reshape(
            n_chains * n_draws,
            n_observations,
        )

        predictions = predictions.T

    elif predictions.ndim == 2:

        # Could already be:
        # draws × observations

        if predictions.shape[1] == n_observations:

            predictions = predictions.T

        # Or:
        # observations × draws

        elif predictions.shape[0] == n_observations:

            pass

        else:

            raise ValueError(
                "Could not determine observation dimension "
                f"from prediction shape {predictions.shape}. "
                f"Expected one dimension to equal "
                f"{n_observations}."
            )

    else:

        raise ValueError(
            "Unexpected number of dimensions in posterior "
            f"predictive samples: {predictions.ndim}. "
            f"Shape: {predictions.shape}"
        )

    print(
        f"  Final shape: {predictions.shape}"
    )

    print(
        f"  Observations: {predictions.shape[0]}"
    )

    print(
        f"  Posterior samples: {predictions.shape[1]}"
    )

    if predictions.shape[0] != n_observations:

        raise ValueError(
            "Prediction observation count does not match "
            f"test data. Predictions contain "
            f"{predictions.shape[0]} observations, while "
            f"test data contains {n_observations}."
        )

    return predictions


# ---------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------

def evaluate_classification(
    predictions: np.ndarray,
    target: str,
    y_true: np.ndarray,
) -> dict:
    """
    Evaluate Bayesian classification predictions.

    predictions must have shape:

        observations × posterior samples

    The posterior mode is used as the final class prediction.
    """

    predictions = flatten_posterior_samples(
        predictions,
        len(y_true),
    )

    # ---------------------------------------------------------------
    # Posterior mode
    # ---------------------------------------------------------------

    y_pred = np.apply_along_axis(
        lambda values: np.bincount(
            values.astype(int)
        ).argmax(),
        axis=1,
        arr=predictions,
    )

    print(
        "\nPrediction class counts:"
    )

    predicted_counts = np.bincount(
        y_pred,
        minlength=int(
            max(
                np.max(y_true),
                np.max(y_pred),
            )
        ) + 1,
    )

    observed_counts = np.bincount(
        y_true,
        minlength=len(predicted_counts),
    )

    for class_index in range(
        len(predicted_counts)
    ):

        print(
            f"  Class {class_index}: "
            f"observed={observed_counts[class_index]:,}, "
            f"predicted={predicted_counts[class_index]:,}"
        )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    return {
        "target": target,
        "metric": "macro_f1",
        "value": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------
# Age evaluation
# ---------------------------------------------------------------------

def evaluate_age(
    predictions: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """
    Evaluate Bayesian age predictions.

    predictions must have shape:

        observations × posterior samples

    The posterior predictive mean is used as the point prediction.
    """

    predictions = flatten_posterior_samples(
        predictions,
        len(y_true),
    )

    y_pred = np.mean(
        predictions,
        axis=1,
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    return {
        "metric": "MAE",
        "value": mae,
        "RMSE": rmse,
    }


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Bayesian models on held-out test data."
        )
    )

    parser.add_argument(
        "test_file",
        type=Path,
        help="Held-out test CSV or Parquet dataset.",
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "gender",
            "region",
            "education",
            "age_linear",
            "age_nonlinear",
        ],
        help="Model/target to evaluate.",
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
        "--model-file",
        type=Path,
        required=True,
        help="Saved Bayesian InferenceData (.nc) file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/validation"
        ),
        help="Directory for evaluation results.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for posterior predictive sampling.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Bayesian model validation")
    print("=" * 70)

    print(
        f"\nTest data: {args.test_file}"
    )

    print(
        f"Model: {args.model}"
    )

    print(
        f"Features: {args.features}"
    )

    print(
        f"Posterior: {args.model_file}"
    )

    # ---------------------------------------------------------------
    # Load test data
    # ---------------------------------------------------------------

    df = load_data(
        args.test_file
    )

    print(
        f"\nTest observations: "
        f"{len(df):,}"
    )

    # ---------------------------------------------------------------
    # Prepare test data
    # ---------------------------------------------------------------

    if args.model in {
        "gender",
        "region",
        "education",
    }:

        X, y_true, classes = (
            prepare_classification_test_data(
                df,
                args.model,
                args.features,
            )
        )

        print(
            f"Predictors: {X.shape[1]}"
        )

        print(
            f"Classes: {len(classes)}"
        )

        print(
            f"Class order: {classes}"
        )

        print(
            "\nObserved test class counts:"
        )

        observed_counts = np.bincount(
            y_true,
            minlength=len(classes),
        )

        for class_index, class_name in enumerate(
            classes
        ):

            print(
                f"  {class_index}: "
                f"{class_name} = "
                f"{observed_counts[class_index]:,}"
            )

    else:

        X, y_true = prepare_test_data(
            df,
            "age",
            args.features,
        )

        print(
            f"Predictors: {X.shape[1]}"
        )

    # ---------------------------------------------------------------
    # Load posterior
    # ---------------------------------------------------------------

    print(
        "\nLoading posterior..."
    )

    idata = az.from_netcdf(
        args.model_file
    )

    print(
        f"Posterior groups: "
        f"{idata.groups}"
    )

    # ArviZ versions using xarray DataTree may return group names
    # with a leading slash, e.g. '/posterior'.
    posterior_group = getattr(
        idata,
        "posterior",
        None,
    )

    if posterior_group is None:

        raise ValueError(
            "The supplied NetCDF file does not contain "
            "a posterior group."
        )

    print(
        f"Posterior variables: "
        f"{list(posterior_group.data_vars)}"
    )

    # ---------------------------------------------------------------
    # Rebuild model with test predictors
    # ---------------------------------------------------------------

    print(
        "\nBuilding prediction model..."
    )

    model_module = load_model_module()

    model = build_model(
        model_module,
        args.model,
        X,
        y_true,
    )

    likelihood_name = get_likelihood_name(
        args.model
    )

    print(
        f"Likelihood variable: "
        f"{likelihood_name}"
    )

    print(
        f"Model observed variables: "
        f"{[rv.name for rv in model.observed_RVs]}"
    )

    print(
        f"Model free variables: "
        f"{[rv.name for rv in model.free_RVs]}"
    )

    if likelihood_name not in [
        rv.name
        for rv in model.observed_RVs
    ]:

        raise ValueError(
            f"Expected likelihood '{likelihood_name}' "
            "was not found among model observed variables."
        )

    # ---------------------------------------------------------------
    # Generate posterior predictive predictions
    # ---------------------------------------------------------------

    prediction_array = generate_test_predictions(
        idata=idata,
        model=model,
        likelihood_name=likelihood_name,
        random_seed=args.random_seed,
    )

    # ---------------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------------

    if args.model in {
        "gender",
        "region",
        "education",
    }:

        metrics = evaluate_classification(
            prediction_array,
            args.model,
            y_true,
        )

        print(
            "\nClassification results:"
        )

        print(
            f"  Macro F1:    "
            f"{metrics['value']:.4f}"
        )

        print(
            f"  Weighted F1: "
            f"{metrics['weighted_f1']:.4f}"
        )

        print(
            f"  Accuracy:    "
            f"{metrics['accuracy']:.4f}"
        )

    else:

        metrics = evaluate_age(
            prediction_array,
            y_true,
        )

        print(
            "\nAge prediction results:"
        )

        print(
            f"  MAE:  "
            f"{metrics['value']:.4f}"
        )

        print(
            f"  RMSE: "
            f"{metrics['RMSE']:.4f}"
        )

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "model": args.model,
        "feature_set": args.features,
        "test_observations": len(y_true),
        "metric": metrics["metric"],
        "value": metrics["value"],
    }

    if "weighted_f1" in metrics:

        result["weighted_f1"] = (
            metrics["weighted_f1"]
        )

    if "accuracy" in metrics:

        result["accuracy"] = (
            metrics["accuracy"]
        )

    if "RMSE" in metrics:

        result["RMSE"] = metrics["RMSE"]

    result_df = pd.DataFrame(
        [result]
    )

    output_file = (
        args.output_dir
        / (
            f"{args.model}"
            f"_{args.features}"
            "_metrics.csv"
        )
    )

    result_df.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nSaved metrics to:\n"
        f"  {output_file}"
    )

    print(
        "\nValidation completed successfully."
    )


if __name__ == "__main__":
    main()