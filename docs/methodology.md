# Methodology

## 1. Experimental Design

The study evaluates a decision-native model using controlled binary decision tasks.

Each experiment consists of a set of cases containing:

- a model state;
- a decision question;
- criteria defining the true and false outcomes;
- a ground-truth label;
- the probability returned by the decision model.

The basic representation is:

```text
State + Question + Criteria
            ↓
       Decision Model
            ↓
     Probability p ∈ [0,1]
            ↓
      Binary Decision
```

A probability close to 1 represents support for the true outcome, while a probability close to 0 represents support for the false outcome.

---

## 2. Ground Truth

Ground truth is defined independently of the model output.

For each evaluation case:

```text
label = 1 → true
label = 0 → false
```

The model is never used to generate the ground-truth label.

This separation is important because calibration and reliability measurements require an independently defined reference outcome.

---

## 3. Basic Classification

The continuous probability output is converted into a binary prediction using a threshold.

For the primary binary analysis:

```text
p ≥ 0.5 → predicted true
p < 0.5 → predicted false
```

The resulting predictions are compared against the ground-truth labels.

The following classification metrics are calculated:

### Accuracy

The proportion of all cases classified correctly.

### Precision

The proportion of predicted-positive cases that are actually positive.

### Recall

The proportion of actual-positive cases that are correctly identified.

### Specificity

The proportion of actual-negative cases that are correctly identified.

### F1 Score

The harmonic mean of precision and recall.

---

## 4. Probability-Based Evaluation

Classification accuracy does not describe whether a probability estimate is meaningful.

For this reason, the study also evaluates the probability directly.

### Brier Score

For binary outcomes:

```text
Brier = mean((p - y)²)
```

where:

- `p` is the predicted probability;
- `y` is the ground-truth label.

Lower values indicate predictions that are closer to the observed outcomes.

### Log Loss

Binary log loss is calculated as:

```text
-log(p)       when y = 1
-log(1 - p)   when y = 0
```

The metric penalizes incorrect predictions made with high confidence.

### Expected Calibration Error

Predictions are grouped into probability bins.

For each bin, predicted confidence is compared with observed accuracy.

The weighted difference across bins provides the expected calibration error.

---

## 5. Confidence

For binary decisions, confidence is defined as the probability assigned to the selected outcome.

Therefore:

```text
p ≥ 0.5:
    confidence = p

p < 0.5:
    confidence = 1 - p
```

This makes confidence independent of which class was selected.

A prediction of:

```text
p = 0.98
```

has confidence `0.98`, while:

```text
p = 0.03
```

has confidence `0.97`.

---

## 6. Pilot Dataset

The first pilot contains 50 hand-designed cases.

The class distribution is balanced:

```text
True  = 25
False = 25
```

The cases cover:

- mathematics;
- science;
- general knowledge;
- reasoning.

The pilot is intended as a controlled baseline rather than a representative benchmark.

Its primary purpose is to verify the complete experimental pipeline before introducing harder sources of variation.

---

## 7. Experimental Logging

Each API evaluation records the information required to reproduce the analysis.

The experiment pipeline separates:

```text
Input Dataset
      ↓
API Runner
      ↓
Raw Result
      ↓
Structured Log
      ↓
Statistical Analysis
      ↓
Figures / Tables
```

The runner records API usage information so that experiment cost can also be monitored.

Private API responses are excluded from the public repository.

---

## 8. Counterfactual Evaluation

The main stability experiments will use pairs or groups of cases representing the same underlying decision.

An original case is paired with one or more semantically equivalent variants.

For example:

```text
Original
"Was the package delivered yesterday?"

Variant
"According to the information provided, did the package arrive yesterday?"
```

The wording changes, but the underlying proposition remains unchanged.

The ground-truth label therefore remains constant.

The primary stability quantity is the change in probability:

```text
Δp = |p_original - p_variant|
```

Larger values indicate greater sensitivity to the perturbation.

Binary decision flips will also be recorded.

---

## 9. Distribution Shift

Distribution-shift experiments will modify characteristics of the evaluation data while preserving the underlying evaluation objective.

The planned shifts include:

### Language Shift

Evaluate equivalent tasks in different languages.

### Domain Shift

Move from the domains represented in the initial pilot to domains not represented, or less represented, in the baseline data.

### Task Shift

Alter the structure of the decision problem while retaining a comparable binary decision objective.

Performance and calibration will be measured separately for each distribution.

---

## 10. Robustness Experiments

Additional experiments will investigate whether the decision signal depends on factors that should not materially change the underlying answer.

These include:

- state perturbations;
- decision representation;
- question decomposition;
- batching;
- repeated evaluation.

The purpose is not to assume that every variation should produce identical numerical probabilities.

Instead, the analysis asks whether the magnitude of variation is compatible with the intended use of the probability as a control signal.

---

## 11. Control Experiment

The final stage connects the decision signal to downstream generative inference.

A simplified controller is:

```text
                 Input
                   │
                   ▼
              JEV decision
                   │
             ┌─────┴─────┐
             │           │
        sufficient    insufficient
        confidence    confidence
             │           │
             ▼           ▼
          proceed      escalate
                         │
                         ▼
                  stronger model
```

The controller may also include an abstention path where appropriate.

The evaluation compares:

- task reliability;
- number of expensive model calls;
- escalation rate;
- abstention rate;
- computation or token usage.

The objective is to determine whether the decision signal provides practical value when used as part of an inference policy.

---

## 12. Statistical Reporting

Results will be reported separately for each experimental condition.

Where appropriate, the analysis will include:

- sample size;
- mean and median;
- dispersion;
- confidence intervals;
- calibration measurements;
- error counts;
- decision-flip rates;
- probability-change distributions.

The analysis will avoid treating a single aggregate score as sufficient evidence for reliability.

---

## 13. Reproducibility

Every experiment should have:

1. a fixed or versioned input dataset;
2. a documented experiment configuration;
3. deterministic analysis code where possible;
4. recorded model/version information;
5. recorded API usage;
6. generated tables and figures;
7. a documented execution procedure.

Raw private API outputs may remain local when they contain information that should not be committed to the public repository.

---

## 14. Interpretation

The central interpretation rule is:

> High accuracy is necessary for a useful decision signal, but it is not sufficient.

A decision model intended to control generative inference must also be evaluated for calibration, stability, robustness, and practical utility.

Consequently, the study will treat the following outcomes separately:

```text
Capability
Calibration
Stability
Robustness
Control Utility
```

A strong result in one category will not be treated as evidence that the other properties automatically hold.