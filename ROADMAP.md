# Bayesian Model Improvement Roadmap

The current Bayesian models serve as the reference models for the next stage of the project.

The reported results from the Mendeley experiment are treated as an external benchmark. The objective is not simply to reproduce or exceed those values, but to investigate whether increasingly appropriate Bayesian model structures can improve predictive performance while retaining uncertainty estimates and a principled probabilistic framework.

The improvement process will be incremental:

> **Model → validate → compare → improve → validate again**

The same held-out test observations should be retained whenever models for the same target are compared.

---

## 1. Current Bayesian models — reference

The existing models are the starting point:

* Gender: multinomial logistic regression
* Region: hierarchical multinomial logistic regression
* Education: ordinal logistic regression
* Age: Student-t regression

  * linear specification
  * quadratic/nonlinear specification

The current held-out validation results provide the reference performance for all subsequent experiments.

The Mendeley results provide an external benchmark for comparison.

---

# 2. Normalized interactions

The first model extension will introduce interactions between survey variables.

The motivation is that the effect of one survey response may depend on another response.

Instead of assuming:

```text
effect of question A
+
effect of question B
```

the model can represent:

```text
effect of question A
+
effect of question B
+
interaction between A and B
```

### Bayesian specification

Interactions should be constructed using normalized/standardized predictors rather than directly multiplying the original variables.

The interaction coefficients should receive appropriate regularizing Bayesian priors so that unnecessary interactions are shrunk toward zero.

We should initially avoid unrestricted higher-order interactions.

### Evaluation

For each target:

1. Fit the interaction model on the training data.
2. Generate predictions for the same held-out test set.
3. Calculate the same validation metrics.
4. Compare against the current reference model.
5. Examine whether the additional complexity produces a meaningful improvement.

If interactions improve performance, continue investigating them.

If they do not, retain the simpler model and move to the next extension.

---

# 3. Hierarchical effects

The next major extension will introduce more hierarchical structure across the 20 survey questions.

The questions are related and currently have separate coefficients. A hierarchical model can allow these effects to partially pool:

```text
                 shared distribution
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
      question 1     question 2     question 20
```

This allows information to be shared across questions while retaining question-specific effects.

### Possible structure

Question-specific coefficients can be modeled as:

```text
question effect ~ Normal(shared mean, shared scale)
```

with the shared parameters receiving Bayesian priors.

### Evaluation

Again:

1. Train on the same training observations.
2. Predict the same test observations.
3. Calculate the same metrics.
4. Compare against the best model established so far.

The purpose is to determine whether partial pooling improves generalization relative to treating all question effects independently.

---

# 4. Demographics as predictors

After evaluating the survey-only model improvements, investigate whether demographic information can improve prediction.

The general structure becomes:

```text
survey responses
        +
completed demographics
        ↓
target prediction
```

The demographic variables considered are:

* gender
* region
* education
* age

### Important validation requirement

Demographic completion must respect the train/test boundary.

For example:

```text
TRAIN
  ↓
fit demographic model
  ↓
predict missing demographics
  ↓
construct enriched training predictors
  ↓
fit final model


TEST
  ↓
use only training-fitted demographic model
  ↓
predict missing demographics
  ↓
construct enriched test predictors
  ↓
evaluate
```

The test data must never be used to fit the demographic-imputation models.

### Hard versus probabilistic demographic predictors

We should initially consider whether demographic information should be represented as:

* hard predicted categories, or
* posterior class probabilities.

For Bayesian modeling, posterior probabilities may be particularly useful because they retain uncertainty rather than treating an uncertain prediction as a known fact.

For age, posterior uncertainty can similarly be propagated rather than using only a single predicted age.

---

# 5. Continue improving if performance improves

There is no predetermined stopping point after the first extension.

The Mendeley results provide a benchmark, but they are not a hard ceiling.

The decision process should be:

```text
Current model
      ↓
Does extension improve validation?
      │
   ┌──┴──┐
  YES    NO
   │      │
   ▼      ▼
keep    retain simpler
   │      │
   └──┬───┘
      ↓
try next justified extension
```

If an extension produces a meaningful improvement, it becomes the new reference model and further improvements can be investigated.

If an extension does not improve predictive performance, the simpler model should be retained rather than adding complexity without evidence of benefit.

---

# 6. Model comparison criteria

Every extension should be evaluated using the existing held-out test framework.

For classification:

* Accuracy
* Macro F1
* Weighted F1
* Class-specific precision
* Class-specific recall
* Class-specific F1

For age:

* MAE
* RMSE

In addition, Bayesian-specific considerations should be examined where useful:

* predictive uncertainty
* posterior behavior
* convergence diagnostics
* whether additional complexity produces unstable estimates

A small numerical improvement should not automatically justify a substantially more complicated model.

---

# 7. Current priority

The improvement branch is intentionally limited to three main directions:

```text
1. Normalized interactions
          ↓
2. Hierarchical effects
          ↓
3. Demographics as predictors
```

Alternative priors, latent-factor models, joint demographic models, sophisticated nonlinear functions, and detailed missingness models are **not current priorities**.

They can be reconsidered later if the main improvement path does not produce sufficient predictive performance.

---

## Overall principle

The project should remain empirical and incremental.

We do not assume that a more complex Bayesian model will perform better.

Each extension must demonstrate its value on held-out data.

The Mendeley results provide an external benchmark, while the Bayesian models are progressively improved until either:

1. predictive performance approaches or exceeds the benchmark, or
2. further complexity no longer produces meaningful improvements.

In either case, the experimental results are informative.
