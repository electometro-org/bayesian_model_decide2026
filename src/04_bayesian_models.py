from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm


# ============================================================
# Configuration
# ============================================================

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
    "combined": (
        COMBINED_COLUMNS
    ),
    "all": (
        POSITION_COLUMNS
        + IMPORTANCE_COLUMNS
        + COMBINED_COLUMNS
    ),
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

# ============================================================
# Data loading
# ============================================================

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


# ============================================================
# Feature preparation
# ============================================================

def prepare_gender_data(
    df: pd.DataFrame,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:

    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set: {feature_set}"
        )

    feature_columns = FEATURE_SETS[feature_set]

    required_columns = [
        "demographics_gender",
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

    model_df = df[required_columns].copy()

    # Target is not imputed.
    model_df = model_df.dropna(
        subset=["demographics_gender"]
    )

    unexpected = set(
        model_df["demographics_gender"].dropna().unique()
    ) - set(GENDER_CLASSES)

    if unexpected:
        raise ValueError(
            "Unexpected gender categories: "
            f"{sorted(unexpected)}"
        )

    y = pd.Categorical(
        model_df["demographics_gender"],
        categories=GENDER_CLASSES,
    ).codes

    X = (
        model_df[feature_columns]
        .astype(float)
        .to_numpy()
    )

    return (
        X,
        y.astype("int64"),
        model_df.index,
    )

def prepare_region_data(
    df: pd.DataFrame,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """
    Prepare the region target and selected predictors.

    Rows with missing region are excluded because region is
    the target for this model.

    Lima is the reference category.
    """

    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set: {feature_set}"
        )

    feature_columns = FEATURE_SETS[feature_set]

    required_columns = [
        "demographics_region",
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

    model_df = df[required_columns].copy()

    # Target is not imputed.
    model_df = model_df.dropna(
        subset=["demographics_region"]
    )

    # --------------------------------------------------------
    # Check region categories
    # --------------------------------------------------------

    unexpected = set(
        model_df["demographics_region"].dropna().unique()
    ) - set(REGION_CLASSES)

    if unexpected:
        raise ValueError(
            "Unexpected region categories: "
            f"{sorted(unexpected)}"
        )

    # --------------------------------------------------------
    # Encode region
    # --------------------------------------------------------

    y = pd.Categorical(
        model_df["demographics_region"],
        categories=REGION_CLASSES,
    ).codes

    # --------------------------------------------------------
    # Predictors
    # --------------------------------------------------------

    X = (
        model_df[feature_columns]
        .astype(float)
        .to_numpy()
    )

    return (
        X,
        y.astype("int64"),
        model_df.index,
    )

def prepare_education_data(
    df: pd.DataFrame,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:

    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set: {feature_set}"
        )

    feature_columns = FEATURE_SETS[feature_set]

    required_columns = [
        "demographics_education",
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

    model_df = df[required_columns].copy()

    # Target is not imputed.
    model_df = model_df.dropna(
        subset=["demographics_education"]
    )

    unexpected = set(
        model_df["demographics_education"].dropna().unique()
    ) - set(EDUCATION_CLASSES)

    if unexpected:
        raise ValueError(
            "Unexpected education categories: "
            f"{sorted(unexpected)}"
        )

    y = pd.Categorical(
        model_df["demographics_education"],
        categories=EDUCATION_CLASSES,
        ordered=True,
    ).codes

    X = (
        model_df[feature_columns]
        .astype(float)
        .to_numpy()
    )

    return (
        X,
        y.astype("int64"),
        model_df.index,
    )

def prepare_age_data(
    df: pd.DataFrame,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:

    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set: {feature_set}"
        )

    feature_columns = FEATURE_SETS[feature_set]

    required_columns = [
        "demographics_age",
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

    model_df = df[required_columns].copy()

    # --------------------------------------------------------
    # Target is not imputed
    # --------------------------------------------------------

    model_df = model_df.dropna(
        subset=["demographics_age"]
    )

    # --------------------------------------------------------
    # Check age values
    # --------------------------------------------------------

    age = pd.to_numeric(
        model_df["demographics_age"],
        errors="coerce",
    )

    # Invalid ages below 18 are excluded from the
    # age training target.
    age = age.where(
        age >= 18
    )

    valid_age = age.notna()

    model_df = model_df.loc[
        valid_age
    ].copy()

    age = age.loc[
        valid_age
    ]

    # --------------------------------------------------------
    # Predictors
    # --------------------------------------------------------

    X = (
        model_df[feature_columns]
        .astype(float)
        .to_numpy()
    )

    y = age.astype(float).to_numpy()

    return (
        X,
        y,
        model_df.index,
    )

# ============================================================
# Gender model
# ============================================================

def build_gender_model(
    X: np.ndarray,
    y: np.ndarray,
) -> pm.Model:
    """
    Bayesian multinomial logistic regression for gender.

    Male is the reference category.

    Classes:
        0 = male
        1 = female
        2 = diverse
    """

    n_observations, n_predictors = X.shape

    with pm.Model() as model:

        # ----------------------------------------------------
        # Data containers
        # ----------------------------------------------------

        X_data = pm.Data(
            "X",
            X,
        )

        # ----------------------------------------------------
        # Priors
        # ----------------------------------------------------

        # Male is the reference category, so only two sets
        # of coefficients are estimated.
        alpha = pm.Normal(
            "alpha",
            mu=0.0,
            sigma=1.5,
            shape=2,
        )

        beta = pm.Normal(
            "beta",
            mu=0.0,
            sigma=0.5,
            shape=(2, n_predictors),
        )

        # ----------------------------------------------------
        # Linear predictors
        # ----------------------------------------------------

        eta = alpha + pm.math.dot(
            X_data,
            beta.T,
        )

        # Add the reference category's zero linear predictor.
        zero = pm.math.zeros(
            (n_observations, 1)
        )

        logits = pm.math.concatenate(
            [
                zero,
                eta,
            ],
            axis=1,
        )

        probabilities = pm.Deterministic(
            "probabilities",
            pm.math.softmax(logits, axis=1),
        )

        # ----------------------------------------------------
        # Likelihood
        # ----------------------------------------------------

        pm.Categorical(
            "gender",
            p=probabilities,
            observed=y,
        )

    return model

# ============================================================
# Region model
# ============================================================

def build_region_model(X, y):
    n_observations, n_predictors = X.shape
    n_regions = len(REGION_CLASSES)
    n_non_reference = n_regions - 1

    with pm.Model() as model:

        X_data = pm.Data("X", X)

        # Hierarchical prior for region-specific coefficients
        beta_sigma = pm.HalfNormal(
            "beta_sigma",
            sigma=0.5,
        )

        beta = pm.Normal(
            "beta",
            mu=0.0,
            sigma=beta_sigma,
            shape=(n_non_reference, n_predictors),
        )

        # Region-specific intercepts
        alpha = pm.Normal(
            "alpha",
            mu=0.0,
            sigma=1.5,
            shape=n_non_reference,
        )

        # Linear predictor
        eta = alpha + pm.math.dot(X_data, beta.T)

        # Lima is the reference category
        zero = pm.math.zeros(
            (n_observations, 1)
        )

        logits = pm.math.concatenate(
            [zero, eta],
            axis=1,
        )

        probabilities = pm.math.softmax(logits, axis=1)

        # Multinomial likelihood
        pm.Categorical(
            "region",
            p=probabilities,
            observed=y,
        )

    return model

def run_region_prior_predictive(model, output_dir):
    with model:
        prior_predictive = pm.sample_prior_predictive(
            draws=500,
            random_seed=42,
        )

    prior_predictive.to_netcdf(
        output_dir / "region_prior_predictive.nc"
    )

    return prior_predictive

def run_region_model(
    data_file: Path,
    output_dir: Path,
    feature_set: str,
    draws: int,
    tune: int,
    chains: int,
    random_seed: int,
) -> None:

    print("=" * 70)
    print("Bayesian model: Region | Hierarchical multinomial logistic")
    print("=" * 70)

    print("\nLoading data...")
    df = load_data(data_file)

    X, y, row_index = prepare_region_data(
        df,
        feature_set,
    )

    print(f"\nFeature representation: {feature_set}")
    print(f"Predictors: {X.shape[1]}")
    print(f"Observations used: {len(y):,}")

    print("\nRegion distribution:")

    region_counts = pd.Series(y).value_counts().sort_index()

    for code, count in region_counts.items():
        print(
            f"  {REGION_CLASSES[code]:<20} "
            f"{count:,} "
            f"({count / len(y):.2%})"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nBuilding region model...")

    model = build_region_model(
        X,
        y,
    )

    print("\nRunning prior predictive sampling...")

    with model:
        prior_predictive = pm.sample_prior_predictive(
            draws=500,
            random_seed=random_seed,
        )

    print("\nRunning posterior sampling...")

    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.90,
            random_seed=random_seed,
            return_inferencedata=True,
        )

    print("\nRunning posterior predictive sampling...")

    with model:
        idata = pm.sample_posterior_predictive(
            idata,
            random_seed=random_seed,
            extend_inferencedata=True,
        )

    model_file = (
        output_dir
        / f"region_{feature_set}.nc"
    )

    idata.to_netcdf(model_file)

    print(
        f"\nSaved InferenceData to:\n"
        f"  {model_file}"
    )

    print("\nSaving diagnostics...")

    save_diagnostics(
        idata,
        output_dir,
        parameter_names=["alpha", "beta", "beta_sigma"],
        plot_variables=["beta_sigma"],
    )

    print("\nRegion model completed successfully.")

# ============================================================
# Education model
# ============================================================

def build_education_model(X, y):
    n_observations, n_predictors = X.shape

    with pm.Model() as model:
        X_data = pm.Data("X", X)

        beta = pm.Normal(
            "beta",
            mu=0.0,
            sigma=0.5,
            shape=n_predictors,
        )

        cutpoints = pm.Normal(
            "cutpoints",
            mu=0.0,
            sigma=1.5,
            shape=3,
            transform=pm.distributions.transforms.ordered,
        )

        eta = pm.math.dot(X_data, beta)

        pm.OrderedLogistic(
            "education",
            eta=eta,
            cutpoints=cutpoints,
            observed=y,
        )

    return model

def run_education_prior_predictive(
    model: pm.Model,
    output_dir: Path,
    draws: int,
    random_seed: int,
):
    """Run and save education prior predictive samples."""

    print("\nRunning education prior predictive sampling...")

    with model:
        prior_predictive = pm.sample_prior_predictive(
            draws=draws,
            random_seed=random_seed,
        )

    output_path = (
        output_dir
        / "education_prior_predictive.nc"
    )

    prior_predictive.to_netcdf(
        output_path
    )

    print(
        f"Saved prior predictive to:\n"
        f"  {output_path}"
    )

    return prior_predictive


def run_education_model(
    data_file: Path,
    output_dir: Path,
    feature_set: str,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    random_seed: int,
    target_accept: float
) -> None:

    print("=" * 70)
    print(
        "Bayesian model: Education | Ordinal logistic regression"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading data...")

    df = load_data(
        data_file
    )

    print(
        f"Dataset dimensions: "
        f"{df.shape[0]:,} rows × {df.shape[1]} columns"
    )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    X, y, row_index = prepare_education_data(
        df,
        feature_set,
    )

    print(
        f"\nFeature representation: "
        f"{feature_set}"
    )

    print(
        f"Predictors: {X.shape[1]}"
    )

    print(
        f"Observations used: {len(y):,}"
    )

    # --------------------------------------------------------
    # Education distribution
    # --------------------------------------------------------

    print("\nEducation distribution:")

    education_counts = (
        pd.Series(y)
        .value_counts()
        .sort_index()
    )

    for code, count in education_counts.items():
        print(
            f"  {EDUCATION_CLASSES[code]:<15} "
            f"{count:,} "
            f"({count / len(y):.2%})"
        )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\nBuilding education model...")

    model = build_education_model(
        X,
        y,
    )

    # --------------------------------------------------------
    # Prior predictive
    # --------------------------------------------------------

    prior_predictive = run_education_prior_predictive(
        model=model,
        output_dir=output_dir,
        draws=500,
        random_seed=random_seed,
    )

    # --------------------------------------------------------
    # Posterior
    # --------------------------------------------------------

    print("\nRunning posterior sampling...")

    # Valid starting values for the ordered cutpoints.
    # They must be strictly increasing.
    initvals = {
        "cutpoints": np.array([-1.0, 0.0, 1.0]),
    }

    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
            initvals=initvals,
        )

    # --------------------------------------------------------
    # Posterior predictive
    # --------------------------------------------------------

    print(
        "\nRunning posterior predictive sampling..."
    )

    with model:
        idata = pm.sample_posterior_predictive(
            idata,
            random_seed=random_seed,
            extend_inferencedata=True,
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_file = (
        output_dir
        / f"education_{feature_set}.nc"
    )

    idata.to_netcdf(
        model_file
    )

    print(
        f"\nSaved InferenceData to:\n"
        f"  {model_file}"
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print("\nSaving diagnostics...")

    save_diagnostics(
        idata,
        output_dir,
        parameter_names=["beta", "cutpoints"],
        plot_variables=["cutpoints"],
    )

    # --------------------------------------------------------
    # Posterior predictive checks
    # --------------------------------------------------------

    save_education_posterior_predictive_checks(
        idata,
        output_dir,
    )

    print(
        "\nEducation model completed successfully."
    )

# ============================================================
# Age model
# ============================================================
def build_age_linear_model(
    X: np.ndarray,
    y: np.ndarray,
) -> pm.Model:
    """
    Bayesian Student-t regression for age.

    Linear predictor:
        mu = alpha + X * beta
    """

    n_observations, n_predictors = X.shape

    with pm.Model() as model:

        # ----------------------------------------------------
        # Data
        # ----------------------------------------------------

        X_data = pm.Data(
            "X",
            X,
        )

        # ----------------------------------------------------
        # Priors
        # ----------------------------------------------------

        alpha = pm.Normal(
            "alpha",
            mu=30.0,
            sigma=15.0,
        )

        beta = pm.Normal(
            "beta",
            mu=0.0,
            sigma=3.0,
            shape=n_predictors,
        )

        sigma = pm.HalfNormal(
            "sigma",
            sigma=15.0,
        )

        nu = pm.Exponential(
            "nu_minus_two",
            lam=0.1,
        )

        nu = pm.Deterministic(
            "nu",
            2.0 + nu,
        )

        # ----------------------------------------------------
        # Linear predictor
        # ----------------------------------------------------

        mu = pm.Deterministic(
            "mu",
            alpha + pm.math.dot(
                X_data,
                beta,
            ),
        )

        # ----------------------------------------------------
        # Likelihood
        # ----------------------------------------------------

        pm.StudentT(
            "age",
            nu=nu,
            mu=mu,
            sigma=sigma,
            observed=y,
        )

    return model

def run_age_linear_model(
    data_file: Path,
    output_dir: Path,
    feature_set: str,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    random_seed: int,
    target_accept: float,
) -> None:

    print("=" * 70)
    print("Bayesian model: Age | Linear Student-t regression")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading data...")

    df = load_data(data_file)

    print(
        f"Dataset dimensions: "
        f"{df.shape[0]:,} rows × {df.shape[1]} columns"
    )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    X, y, row_index = prepare_age_data(
        df,
        feature_set,
    )

    print(
        f"\nFeature representation: {feature_set}"
    )

    print(
        f"Predictors: {X.shape[1]}"
    )

    print(
        f"Observations used: {len(y):,}"
    )

    # --------------------------------------------------------
    # Age distribution
    # --------------------------------------------------------

    print("\nAge distribution:")

    print(
        f"  Mean:   {np.mean(y):.2f}"
    )

    print(
        f"  Median: {np.median(y):.2f}"
    )

    print(
        f"  SD:     {np.std(y, ddof=1):.2f}"
    )

    print(
        f"  Min:    {np.min(y):.0f}"
    )

    print(
        f"  Max:    {np.max(y):.0f}"
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\nBuilding age linear model...")

    model = build_age_linear_model(
        X,
        y,
    )

    # --------------------------------------------------------
    # Prior predictive
    # --------------------------------------------------------

    print("\nRunning prior predictive sampling...")

    with model:
        prior_predictive = pm.sample_prior_predictive(
            draws=500,
            random_seed=random_seed,
        )

    prior_predictive.to_netcdf(
        output_dir / "age_linear_prior_predictive.nc"
    )

    # --------------------------------------------------------
    # Posterior
    # --------------------------------------------------------

    print("\nRunning posterior sampling...")

    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
        )

    # --------------------------------------------------------
    # Posterior predictive
    # --------------------------------------------------------

    print(
        "\nRunning posterior predictive sampling..."
    )

    with model:
        idata = pm.sample_posterior_predictive(
            idata,
            random_seed=random_seed,
            extend_inferencedata=True,
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_file = (
        output_dir
        / f"age_linear_{feature_set}.nc"
    )

    idata.to_netcdf(
        model_file
    )

    print(
        f"\nSaved InferenceData to:\n"
        f"  {model_file}"
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print("\nSaving diagnostics...")

    save_diagnostics(
        idata,
        output_dir,
        parameter_names=[
            "alpha",
            "beta",
            "sigma",
            "nu",
        ],
        plot_variables=[
            "alpha",
            "sigma",
            "nu",
        ],
    )

    print(
        "\nAge linear model completed successfully."
    )

def build_age_nonlinear_model(
    X: np.ndarray,
    y: np.ndarray,
) -> pm.Model:
    """
    Bayesian Student-t regression for age with quadratic predictor effects.

    Linear component:
        alpha + X * beta

    Nonlinear component:
        X^2 * gamma
    """

    n_observations, n_predictors = X.shape

    X_squared = X ** 2

    with pm.Model() as model:
        X_data = pm.Data(
            "X",
            X,
        )

        X_squared_data = pm.Data(
            "X_squared",
            X_squared,
        )

        alpha = pm.Normal(
            "alpha",
            mu=30.0,
            sigma=15.0,
        )

        beta = pm.Normal(
            "beta",
            mu=0.0,
            sigma=3.0,
            shape=n_predictors,
        )

        gamma = pm.Normal(
            "gamma",
            mu=0.0,
            sigma=1.0,
            shape=n_predictors,
        )

        sigma = pm.HalfNormal(
            "sigma",
            sigma=15.0,
        )

        nu_minus_two = pm.Exponential(
            "nu_minus_two",
            lam=0.1,
        )

        nu = pm.Deterministic(
            "nu",
            2.0 + nu_minus_two,
        )

        mu = pm.Deterministic(
            "mu",
            alpha
            + pm.math.dot(X_data, beta)
            + pm.math.dot(X_squared_data, gamma),
        )

        pm.StudentT(
            "age",
            nu=nu,
            mu=mu,
            sigma=sigma,
            observed=y,
        )

    return model

def run_age_nonlinear_model(
    data_file: Path,
    output_dir: Path,
    feature_set: str,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    random_seed: int,
    target_accept: float,
) -> None:

    print("=" * 70)
    print("Bayesian model: Age | Nonlinear Student-t regression")
    print("=" * 70)

    print("\nLoading data...")
    df = load_data(data_file)

    print(
        f"Dataset dimensions: "
        f"{df.shape[0]:,} rows × {df.shape[1]} columns"
    )

    X, y, row_index = prepare_age_data(
        df,
        feature_set,
    )

    print(f"\nFeature representation: {feature_set}")
    print(f"Predictors: {X.shape[1]}")
    print(f"Observations used: {len(y):,}")

    print("\nAge distribution:")
    print(f"  Mean:   {np.mean(y):.2f}")
    print(f"  Median: {np.median(y):.2f}")
    print(f"  SD:     {np.std(y, ddof=1):.2f}")
    print(f"  Min:    {np.min(y):.0f}")
    print(f"  Max:    {np.max(y):.0f}")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nBuilding age nonlinear model...")
    model = build_age_nonlinear_model(
        X,
        y,
    )

    print("\nRunning prior predictive sampling...")
    with model:
        prior_predictive = pm.sample_prior_predictive(
            draws=500,
            random_seed=random_seed,
        )

    prior_predictive.to_netcdf(
        output_dir / "age_nonlinear_prior_predictive.nc"
    )

    print("\nRunning posterior sampling...")
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
        )

    print("\nRunning posterior predictive sampling...")
    with model:
        idata = pm.sample_posterior_predictive(
            idata,
            random_seed=random_seed,
            extend_inferencedata=True,
        )

    model_file = (
        output_dir
        / f"age_nonlinear_{feature_set}.nc"
    )

    idata.to_netcdf(
        model_file
    )

    print(
        f"\nSaved InferenceData to:\n"
        f"  {model_file}"
    )

    print("\nSaving diagnostics...")

    save_diagnostics(
        idata,
        output_dir,
        parameter_names=[
            "alpha",
            "beta",
            "gamma",
            "sigma",
            "nu",
        ],
        plot_variables=[
            "alpha",
            "sigma",
            "nu",
        ],
    )

    print("\nAge nonlinear model completed successfully.")


# ============================================================
# Diagnostics
# ============================================================

def save_diagnostics(
    idata: az.InferenceData,
    output_dir: Path,
    parameter_names: list[str] | None = None,
    plot_variables: list[str] | None = None,
) -> None:
    """Save standard Bayesian diagnostics."""

    if parameter_names is None:
        parameter_names = ["alpha", "beta"]

    if plot_variables is None:
        plot_variables = parameter_names

    diagnostics_dir = (
        output_dir
        / "diagnostics"
    )

    diagnostics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Posterior summary
    # --------------------------------------------------------

    summary = az.summary(
        idata,
        var_names=parameter_names,
        round_to=4,
    )

    summary.to_csv(
        diagnostics_dir
        / "posterior_summary.csv"
    )

    # --------------------------------------------------------
    # Sampling diagnostics
    # --------------------------------------------------------

    sample_stats = idata.sample_stats

    divergences = int(
        sample_stats["diverging"]
        .sum()
        .values
    )

    diagnostics = pd.DataFrame(
        {
            "divergences": [
                divergences
            ],
        }
    )

    diagnostics.to_csv(
        diagnostics_dir
        / "sampling_diagnostics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Trace plot
    # --------------------------------------------------------

    az.plot_trace(
        idata,
        var_names=plot_variables,
    )

    plt.tight_layout()

    plt.savefig(
        diagnostics_dir
        / "trace_parameters.png",
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # Rank plot
    # --------------------------------------------------------

    az.plot_rank(
        idata,
        var_names=plot_variables,
    )

    plt.tight_layout()

    plt.savefig(
        diagnostics_dir
        / "rank_parameters.png",
        dpi=150,
    )

    plt.close()

def save_education_posterior_predictive_checks(
    idata: az.InferenceData,
    output_dir: Path,
) -> None:
    """Save posterior predictive checks for education."""

    ppc_dir = (
        output_dir
        / "posterior_predictive"
    )

    ppc_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if "posterior_predictive" not in idata.groups:
        print(
            "No posterior predictive group found."
        )
        return

    predicted = (
        idata.posterior_predictive["education"]
        .values
    )

    # chains × draws × observations
    predicted = predicted.reshape(
        -1,
        predicted.shape[-1],
    )

    observed = (
        idata.observed_data["education"]
        .values
    )

    observed_counts = np.bincount(
        observed.astype(int),
        minlength=len(EDUCATION_CLASSES),
    )

    predictive_counts = np.apply_along_axis(
        lambda x: np.bincount(
            x.astype(int),
            minlength=len(EDUCATION_CLASSES),
        ),
        axis=1,
        arr=predicted,
    )

    predictive_summary = pd.DataFrame(
        {
            "category": EDUCATION_CLASSES,
            "observed_count": observed_counts,
            "posterior_predictive_mean": (
                predictive_counts.mean(axis=0)
            ),
            "posterior_predictive_sd": (
                predictive_counts.std(axis=0)
            ),
        }
    )

    predictive_summary.to_csv(
        ppc_dir
        / "education_category_counts.csv",
        index=False,
    )

    print(
        f"Saved posterior predictive checks to:\n"
        f"  {ppc_dir}"
    )

# ============================================================
# Posterior predictive checks
# ============================================================

def save_posterior_predictive_checks(
    idata: az.InferenceData,
    output_dir: Path,
) -> None:
    """Create basic posterior predictive diagnostics."""

    ppc_dir = output_dir / "posterior_predictive"
    ppc_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Predictive probabilities
    # --------------------------------------------------------

    probabilities = (
        idata.posterior["probabilities"]
        .mean(
            dim=("chain", "draw")
        )
        .values
    )

    probability_summary = pd.DataFrame(
        probabilities,
        columns=[
            "predicted_male",
            "predicted_female",
            "predicted_diverse",
        ],
    )

    probability_summary.to_csv(
        ppc_dir / "mean_predicted_probabilities.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Observed vs posterior predictive category counts
    # --------------------------------------------------------

    if "posterior_predictive" not in idata.groups:
        return

    predicted = (
        idata.posterior_predictive["gender"]
        .values
    )

    # Shape:
    # chains x draws x observations
    predicted = predicted.reshape(
        -1,
        predicted.shape[-1],
    )

    observed_counts = np.bincount(
        idata.observed_data["gender"].values,
        minlength=3,
    )

    predictive_counts = np.apply_along_axis(
        lambda x: np.bincount(
            x.astype(int),
            minlength=3,
        ),
        axis=1,
        arr=predicted,
    )

    predictive_summary = pd.DataFrame(
        {
            "category": GENDER_CLASSES,
            "observed_count": observed_counts,
            "posterior_predictive_mean": (
                predictive_counts.mean(axis=0)
            ),
            "posterior_predictive_sd": (
                predictive_counts.std(axis=0)
            ),
        }
    )

    predictive_summary.to_csv(
        ppc_dir / "category_counts.csv",
        index=False,
    )


# ============================================================
# Main experiment
# ============================================================

def run_gender_model(
    data_file: Path,
    output_dir: Path,
    feature_set: str,
    draws: int,
    tune: int,
    chains: int,
    random_seed: int,
) -> None:

    print("=" * 70)
    print("Bayesian model: Gender | Combined predictors")
    print("=" * 70)

    print("\nLoading data...")
    df = load_data(data_file)

    print(
        f"Dataset dimensions: "
        f"{df.shape[0]:,} rows × {df.shape[1]} columns"
    )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    feature_columns = FEATURE_SETS[feature_set]
    
    X, y, row_index = prepare_gender_data(
        df,
        feature_set,
    )

    print(
    f"\nFeature representation: {feature_set}"
    )

    print(
        f"Predictors: {X.shape[1]}"
    )

    print(
        f"Observations used: {len(y):,}"
    )

    print(
        "\nGender distribution:"
    )

    gender_counts = pd.Series(
        y
    ).value_counts().sort_index()

    for code, count in gender_counts.items():
        print(
            f"  {GENDER_CLASSES[code]:<10} "
            f"{count:,} "
            f"({count / len(y):.2%})"
        )

    print(
        f"\nPredictors: {X.shape[1]}"
    )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\nBuilding PyMC model...")

    model = build_gender_model(
        X,
        y,
    )

    # --------------------------------------------------------
    # Prior predictive
    # --------------------------------------------------------

    print("\nRunning prior predictive sampling...")

    with model:
        prior_predictive = pm.sample_prior_predictive(
        draws=500,
        random_seed=random_seed,
    )

    # --------------------------------------------------------
    # Posterior sampling
    # --------------------------------------------------------

    print("\nRunning posterior sampling...")

    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.90,
            random_seed=random_seed,
            return_inferencedata=True,
        )

    # --------------------------------------------------------
    # Posterior predictive
    # --------------------------------------------------------

    print("\nRunning posterior predictive sampling...")

    with model:
        idata = pm.sample_posterior_predictive(
            idata,
            random_seed=random_seed,
            extend_inferencedata=True,
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    model_file = (
        output_dir
        / f"gender_{feature_set}.nc"
    )

    idata.to_netcdf(
        model_file
    )

    print(
        f"\nSaved InferenceData to:\n"
        f"  {model_file}"
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print("\nSaving diagnostics...")

    save_diagnostics(
        idata,
        output_dir,
        parameter_names=["alpha", "beta"],
        plot_variables=["alpha"],
    )

    save_posterior_predictive_checks(
        idata,
        output_dir,
    )

    print("\nModel completed successfully.")


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Bayesian predictive modeling for demographic "
            "targets."
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
        "--target",
        choices=[
            "gender",
            "region",
            "education",
            "age_linear",
            "age_nonlinear",
        ],
        default="gender",
        help="Target variable.",
    )

    parser.add_argument(
        "--features",
        choices=[
            "position_importance",
            "combined",
            "all",
        ],
        default="combined",
        help="Feature representation for this prototype.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/bayesian_models"
        ),
        help="Output directory.",
    )

    parser.add_argument(
        "--draws",
        type=int,
        default=1000,
        help="Posterior draws per chain.",
    )

    parser.add_argument(
        "--tune",
        type=int,
        default=1000,
        help="Tuning iterations per chain.",
    )

    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of MCMC chains.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="?",
    )

    parser.add_argument(
        "--target_accept",
        type=float,
        default=0.90,
        help="?",
    )

    args = parser.parse_args()

    if args.target == "education":
        run_education_model(
            data_file=args.data_file,
            output_dir=args.output_dir,
            feature_set=args.features,
            draws=args.draws,
            tune=args.tune,
            cores=args.cores,
            chains=args.chains,
            target_accept=args.target_accept,
            random_seed=args.random_seed,
        )
    elif args.target == "age_linear":
        run_age_linear_model(
            data_file=args.data_file,
            output_dir=args.output_dir,
            feature_set=args.features,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            random_seed=args.random_seed,
        )
    elif args.target == "age_nonlinear":
        run_age_nonlinear_model(
            data_file=args.data_file,
            output_dir=args.output_dir,
            feature_set=args.features,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            random_seed=args.random_seed,
        )
    elif args.target == "region":
        run_region_model(
            data_file=args.data_file,
            feature_set=args.features,
            output_dir=args.output_dir,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            random_seed=args.random_seed,
        )
    elif args.target == "gender":
        run_gender_model(
            data_file=args.data_file,
            feature_set=args.features,
            output_dir=args.output_dir,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            random_seed=args.random_seed,
        )
    else:
        return


if __name__ == "__main__":
    main()