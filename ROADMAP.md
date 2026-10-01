# Bayesian Model Improvement Roadmap

The current Bayesian models are the reference models for the next stage of the project. The results of the first imputation round published on Mendeley (XGBoost / Random Forest) are an **external benchmark**. 

---

## 1. Where we stand

Held-out results of the current reference models (80/20 split, seed 42, per-target split) against the Mendeley benchmark.

| Target | Best current model | Result | Mendeley benchmark | Reading |
|---|---|---|---|---|
| Gender | multinomial logit, `position_importance` | macro F1 0.441, weighted F1 0.659, accuracy 0.667 (n=3775) | macro F1 0.471 (XGBoost) | Small gap. Macro F1 far below accuracy suggests "diverse" is rarely or never predicted (to be confirmed by the per-class report). |
| Region | hierarchical multinomial logit (all feature sets identical) | macro F1 0.030, weighted F1 0.546, accuracy 0.677 (n=3759) | none (planned for dataset v2) | Identical to an "always Lima" predictor with about 27 classes. Features currently change nothing. |
| Education | ordinal logit, `position_importance` | macro F1 0.191, weighted F1 0.424, accuracy 0.571 (n=3668) | F1 0.397 (Random Forest; macro or weighted not stated) | Close to a majority-class baseline. The gap is large if the benchmark is macro F1, absent if it is weighted. |
| Age | Student-t, nonlinear, `all` (`position_importance` nearly equal) | MAE 8.48, RMSE 11.96, R² about 0.12 (estimate) (n=3622) | MAE 7.38, MSE 103.5 (RMSE 10.17), R² 0.366 (XGBoost) | Nonlinear beats linear by about 0.25 years MAE, but the gap to the benchmark is about 1.1 years. |

The R² for age is derived from the benchmark's MSE and R² (implied variance about 163) and is approximate.

**Feature sets.** `combined` is never the best and `all` does not beat `position_importance` for gender, education or the nonlinear age model. Differences are within noise. For all extension experiments, use **`position_importance` (40 predictors)**.

---

## 2. Evaluation protocol

### 2.1 Splits

- During single-target model comparison, keep the existing per-target stratified 80/20 split (age: random). Splits are stratified per target.
- For anything that chains targets (section 4.3), switch to **one shared split** for all targets, stratified on a combined key (for example gender × region with small cells merged, or region alone). Otherwise a respondent can be in the test set for one target and in the training set for another, which leaks information through the chain.
- Compare models for the same target on the **same held-out observations**.

### 2.2 Metric phases

- **Phase 1 (model development).** Classification: macro F1, weighted F1, accuracy, per-class precision/recall/F1. Age: MAE, RMSE. Education additionally uses an ordinal-aware measure.
- **Phase 2 (before any imputation is run).** Add log loss, Brier score, calibration, and prediction-interval coverage for age, for the leading models.
- The final model per target is chosen **after Phase 2**. Phase 1 rankings are provisional.
- Save per-observation posterior class probabilities and posterior predictive summaries for the test set during Phase 1, so Phase 2 needs no refitting.

### 2.3 Comparing models

- During development, compare candidates with **PSIS-LOO / ELPD** on the training data. Fall back to k-fold CV where Pareto-k diagnostics are unreliable (for example very small regions).
- Keep the test set for the final check, and report **bootstrap intervals** on test metrics.
- Report convergence diagnostics (R-hat, ESS, divergences) for every extension.

---

## 3. Experiments per target

Experiments are run in the order listed. Each one is compared against the best model so far; if it does not help, the simpler model is kept.

### 3.1 Gender

1. Per-class report; report "diverse" separately from macro F1.
2. Interactions (4.1) and hierarchical pooling across questions (4.2).
3. SMOTE-NC for "diverse" (4.4). Prior or threshold adjustment is not planned.
4. Demographics as predictors: age, education, region (4.3).

### 3.2 Region

1. **Diagnose first.** Inspect posterior coefficients and class probabilities for a sample of test rows. If they vary by respondent, the argmax is simply always Lima. If they are nearly identical, the model is collapsing to the prior (over-shrinkage or a bug).
2. Different hierarchical models (4.2), including a macro-region hierarchy (for example Lima, coast, sierra, selva) so small regions borrow strength.
3. Interactions (4.1).
4. Thresholds or SMOTE-NC on the macro-regions.
5. Demographics as predictors (4.3).

### 3.3 Education

1. Interactions (4.1).
2. Threshold adjustment, then SMOTE-NC (4.4).
3. Partial proportional-odds model (relaxes the assumption that each predictor shifts all category boundaries equally).
4. Hierarchical variants (4.2); BART if gaps remain (4.5).
5. Demographics as predictors, especially age (4.3), last.

### 3.4 Age

1. Interactions (4.1).
2. Demographics as predictors: education and gender (4.3).
3. Heteroscedastic Student-t (residual scale depends on predictors); also check a log transform of age.
4. Hierarchical variants (4.2).
5. BART if gaps remain (4.5).

---

## 4. Method notes

### 4.1 Normalized interactions

- Build interactions from standardized predictors. With 20 questions there are 190 pairwise terms (more if importance is included), so use a sparsity prior (regularized horseshoe or R2D2) that shrinks unneeded terms toward zero.
- Avoid unrestricted higher-order interactions at first.

### 4.2 Hierarchical models

Compare these variants separately, one change at a time, using the same split and PSIS-LOO:

- Partial pooling across the 20 question effects (`question effect ~ Normal(shared mean, shared scale)`).
- Macro-region hierarchy for region.
- Group-varying slopes by demographic.

### 4.3 Demographics as predictors

**Step 4a: observed demographics with mode fill.**

- Use the other demographics as predictors. Never use the target itself.
- Fill missing predictor values with the mode **and add a "was missing" indicator** for each demographic, so a filled value does not look like a real answer.
- **Masked evaluation is required.** Test rows have the target observed, so these respondents mostly answered the other demographic questions as well. The rows to be imputed in practice mostly did not. Report test performance both with the real predictors and with the other demographics masked at the missingness rate seen among the rows to be imputed. Only the masked version is a realistic estimate for imputation.

**Step 4b: chained prediction.** After 4a:

- Fit demographic models on the training data and pass **posterior draws or probabilities**, not hard labels.
- Train the second stage on out-of-fold predictions (cross-fitting), not on true values, to avoid a train/serve mismatch.
- The test data must never be used to fit the demographic models. Use the shared split from 2.1.

Predictions from the survey alone add no new information beyond what the survey already contains. Gains are expected only through model structure or through observed demographics.

### 4.4 SMOTE

- Split first; generate synthetic rows from training data only, and inside each CV fold if CV is used.
- Predictors are discrete (−1/0/+1 and 0/1), so use SMOTE-NC (or another categorical-aware variant); standard SMOTE interpolates into values that do not exist.
- SMOTE changes the class balance the model sees, so posterior probabilities no longer reflect real frequencies. Correct them back to the real class priors afterwards if `*_confidence` values or calibration checks are used.
- Compare against threshold adjustment where both are planned.

### 4.5 Flexible nonlinear models

BART or a similar Bayesian tree ensemble is a conditional step: try it only if interactions and hierarchical effects leave a clear gap to the benchmark. It gives tree-like flexibility with posterior uncertainty.

---

## 5. Decision rule

```
Current best model
        ↓
Does the extension improve validation (by the agreed threshold)?
   ┌────┴────┐
  YES        NO
   │          │
   ▼          ▼
new reference  keep the simpler model
   │          │
   └────┬─────┘
        ↓
next planned experiment
```

There is no fixed stopping point. The work continues until either predictive performance approaches or exceeds the benchmark, or further complexity no longer produces meaningful gains. Both outcomes are informative.

---

## 6. Not current priorities

Alternative priors beyond the sparsity priors above, latent-factor models, joint demographic models, and detailed missingness models. They can be reconsidered later if the main path does not produce sufficient performance.

**Later:** theory-informed question categories and sociopolitical profiles, as a new feature set (evaluated against position_importance on the same split and metrics) and as a separate profile component. To keep this cheap, keep the splits, seeds and saved test predictions of the current runs.
