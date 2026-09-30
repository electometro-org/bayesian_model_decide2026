# Bayesian Demographic Prediction and Imputation

## 1. Project overview

This project uses Bayesian models to predict missing demographic information from survey responses.

The dataset contains:

* **Gender**
* **Region**
* **Education**
* **Age**

The demographic variables are treated as **prediction targets**, while the survey responses are used as predictors.

---

## 2. Data preparation

The original survey responses contain two components for each of the 20 survey questions:

* `responses_t*_1`: respondent's position
* `responses_t*_2`: importance of the issue

### Position coding

The original position values were recoded as:

| Original value | Meaning | Model value |
| -------------: | ------- | ----------: |
|            0.0 | Oppose  |          -1 |
|            0.5 | Neutral |           0 |
|            1.0 | Support |          +1 |

### Importance coding

The importance values were recoded as:

| Original value | Meaning       | Model value |
| -------------: | ------------- | ----------: |
|            1.0 | Not important |           0 |
|            2.0 | Important     |           1 |

### Combined representation

An additional experimental feature was created by combining position and importance:

| Position | Importance    | Combined value |
| -------- | ------------- | -------------: |
| Oppose   | Important     |           -1.0 |
| Oppose   | Not important |           -0.5 |
| Neutral  | Either        |            0.0 |
| Support  | Not important |           +0.5 |
| Support  | Important     |           +1.0 |

The original position and importance variables were retained. The combined variables were added as an additional representation.

---

## 3. Missing values

Predictor variables had low levels of missingness and were handled using **mode imputation** during preprocessing.

Demographic target variables were **not imputed during preprocessing**.

For age, values below 18 were treated as invalid and therefore unavailable as an age target.

---

## 4. Feature representations

Three predictor representations were evaluated:

### `position_importance`

Uses the original 20 position variables and 20 importance variables.

**40 predictors total.**

### `combined`

Uses the 20 combined variables.

**20 predictors total.**

### `all`

Uses:

* 20 position variables
* 20 importance variables
* 20 combined variables

**60 predictors total.**

These representations are compared using held-out test data.

---

## 5. Bayesian models

Four demographic targets were modeled.

### Gender

A Bayesian multinomial logistic regression was used.

Classes:

* `male`
* `female`
* `diverse`

The model estimates the posterior probability of each gender category given the survey responses.

---

### Region

A Bayesian hierarchical multinomial logistic regression was used.

The model predicts one of the observed region categories while allowing coefficient variation across region categories through hierarchical regularization.

This is particularly relevant because the region distribution contains both a very large Lima category and several relatively small categories.

---

### Education

A Bayesian ordinal logistic regression was used.

Education is treated as ordered:

```text
primary
    ↓
secondary
    ↓
undergraduate
    ↓
graduate
```

The model therefore uses the ordering of the education categories rather than treating them as unrelated nominal classes.

No education sensitivity analysis was performed.

---

### Age

Age is modeled using Bayesian Student-t regression.

Two specifications were evaluated:

#### Linear

```text
age = intercept + survey effects + error
```

#### Nonlinear

The nonlinear model adds quadratic effects:

```text
age = intercept
      + linear survey effects
      + quadratic survey effects
      + error
```

The nonlinear specification was included because previous exploratory modeling suggested that some age relationships may not be purely linear.

The data determine whether the additional nonlinear terms improve predictive performance.

---

## 6. Bayesian priors

Weakly informative priors were used to regularize the models.

For example, the classification coefficients use Normal priors centered at zero, while the age model uses priors centered around a plausible adult age.

The age model also uses a Student-t likelihood, which allows heavier-tailed residuals than a normal regression.

The priors are therefore intended to provide reasonable regularization rather than impose strong demographic assumptions.

---

## 7. Train/test validation

The models were evaluated using held-out observations.

An **80/20 split** was used.

For classification targets:

* Gender → stratified split
* Region → stratified split
* Education → stratified split

For age:

* Random 80/20 split

The random seed was fixed at **42**.

The test observations were not used to fit the corresponding Bayesian posterior.

Each target has its own train/test split because the amount of missingness differs between demographic targets.

---

## 8. Evaluation metrics

### Classification

Gender, region, and education use:

* **Macro F1**
* Weighted F1
* Accuracy

Macro F1 is particularly useful when classes have different frequencies because it gives each class equal weight.

The validation output also reports:

* observed class counts
* predicted class counts
* precision
* recall
* F1 for each class

These class-specific values are important when interpreting predictions for relatively small categories.

### Age

Age uses:

* **MAE** — Mean Absolute Error
* **RMSE** — Root Mean Squared Error

MAE represents the average absolute difference between predicted and observed age.

RMSE gives more weight to larger prediction errors.

Both are measured in **years**.

---

## 9. Interpreting the validation results

The validation results should be interpreted as **predictive performance on previously unseen observations**.

For classification:

* Higher Macro F1 means better average class-level F1 across categories.
* Weighted F1 gives more influence to common classes.
* Accuracy is the proportion of correctly classified observations.
* Class-specific recall indicates how well the model identifies a particular class.
* Class-specific precision indicates how often predictions for a particular class are correct.

For age:

* Lower MAE means smaller average absolute age error.
* Lower RMSE means fewer/lower large errors.

The three feature representations can therefore be compared separately for each target.

For age, the linear and nonlinear specifications can also be compared using the same held-out test observations.

---

# 10. Final annotation / imputation

After model validation, a selected trained posterior can be used to annotate the complete dataset.

The annotation script:

1. Loads the original dataset.
2. Loads a previously fitted Bayesian posterior.
3. Generates predictions for all observations.
4. Preserves the original demographic value.
5. Fills only missing demographic values.
6. Records which values were imputed.
7. Records prediction confidence or uncertainty.
8. Saves a new dataset.

The original data are therefore preserved.

---

## 11. Interpreting the annotated columns

Suppose the original dataset contains:

```text
demographics_gender
```

The annotation process creates:

```text
demographics_gender_original
demographics_gender_imputed
demographics_gender_was_imputed
demographics_gender_confidence
```

### `*_original`

The original value from the input dataset.

This column should be treated as the historical/raw demographic value.

It is never modified.

---

### `*_imputed`

The final completed value.

If the original demographic value exists:

```text
*_imputed = original value
```

If the original value is missing:

```text
*_imputed = Bayesian prediction
```

Therefore, this is the column to use when a complete demographic variable is required.

---

### `*_was_imputed`

A binary indicator:

```text
0 = value was originally observed
1 = value was supplied by the Bayesian model
```

This column is important because it allows downstream analyses to distinguish observed demographic information from model-generated information.

---

### Classification confidence

For gender, region, and education:

```text
*_confidence
```

represents the posterior mean probability of the selected class.

For example:

```text
0.92
```

means that the selected class has an estimated posterior probability of approximately 92%.

A value such as:

```text
0.38
```

indicates substantially more uncertainty about the selected class.

**Confidence is not the same as correctness.**

---

## 12. Age uncertainty

For age, the annotation includes:

```text
demographics_age_posterior_sd
```

This represents the posterior standard deviation of the estimated latent age.

For example:

```text
demographics_age_imputed = 29
demographics_age_posterior_sd = 4.8
```

can be interpreted as a model estimate centered around age 29 with posterior uncertainty represented by approximately 4.8 years.

This should **not** be interpreted as saying that the person's true age is necessarily within exactly ±4.8 years.

---

## 13. Example

Suppose the original data contain:

| Original gender | Imputed gender | Was imputed | Confidence |
| --------------- | -------------- | ----------: | ---------: |
| male            | male           |           0 |          — |
| female          | female         |           0 |          — |
| missing         | male           |           1 |       0.81 |
| missing         | diverse        |           1 |       0.46 |

The first two values were observed in the original data.

The last two were generated by the Bayesian model.

The `was_imputed` column makes this distinction explicit.

---

## 14. Important interpretation rule

The Bayesian imputed values are **model-based estimates, not observed facts**.

They should therefore be treated differently from the original demographic responses.

In particular:

* `*_original` = observed/source information
* `*_imputed` = completed variable combining observed and model-estimated values
* `*_was_imputed` = identifies model-generated observations
* `*_confidence` / `*_posterior_sd` = uncertainty information

For analyses where the distinction between observed and estimated demographics matters, retain and use the imputation indicator.

---

## 15. Reproducibility

The main scripts are:

```text
01_data_audit.py
02_preprocessing.py
03_eda.py
04_bayesian_models.py
05_validation_split.py
06_sensitivity_validation_predictions.py
07_run_all_models_and_validation.sbatch
08_annotate_missing_targets.py
```

The workflow is:

```text
Raw data
   ↓
Data audit
   ↓
Preprocessing
   ↓
EDA
   ↓
Target-specific train/test splits
   ↓
Bayesian model fitting
   ↓
Held-out validation
   ↓
Select model/feature representation
   ↓
Bayesian annotation
   ↓
Completed dataset + imputation indicators
```

The final annotated dataset should always retain the original demographic values and an explicit indicator identifying model-imputed observations.
