# Experiment Matrix

This document tracks the planned empirical evaluation of the decision-native control framework.

The matrix is intentionally staged. Early experiments establish measurement validity before later experiments introduce distribution shift, adversarial conditions, and downstream control.

---

## Experiment Overview

| ID | Experiment | Primary Question | Main Measurements | Status |
|---|---|---|---|---|
| E1 | Basic Capability | Can JEV solve the decision tasks? | Accuracy, precision, recall, F1 | Complete |
| E2 | Calibration | Do probabilities correspond to empirical correctness? | Brier, log loss, ECE, reliability diagram | Baseline complete |
| E3 | Selective Prediction | Can confidence be used for abstention? | Coverage, risk, accuracy | Planned |
| E4 | Counterfactual Stability | Does semantic-preserving wording change the decision signal? | Δ probability, flip rate | Next |
| E5 | Adversarial Robustness | How sensitive is the signal to targeted perturbations? | Δ probability, flip rate | Planned |
| E6 | Language Shift | Does reliability transfer across languages? | Accuracy, Brier, ECE | Planned |
| E7 | Domain Shift | Does reliability transfer across domains? | Accuracy, Brier, ECE | Planned |
| E8 | JEV Controller | Can JEV control downstream inference? | Reliability, escalation rate, cost | Planned |
| E9 | LLM Router Baseline | How does a JEV controller compare with a generative routing approach? | Reliability, cost, routing decisions | Planned |
| E10 | Uncertainty Budget | How does computation change under a fixed reliability target? | Cost, coverage, risk | Planned |
| E11 | Ablation | Which components contribute to observed behaviour? | Change in metrics | Planned |
| E12 | Reproducibility | Are results stable across repeated runs? | Variance, agreement | Planned |
| E13 | Decision Representation | Does the representation of the decision affect the output? | Probability change, flips | Planned |
| E14 | State Manipulation | Does irrelevant state affect decisions? | Probability change, flips | Planned |
| E15 | Question Decomposition | Does decomposition or batching affect decisions? | Probability change, agreement | Planned |
| E16 | Temporal Stability | Are repeated decisions stable enough for control? | Drift, variance, hysteresis | Planned |

---

## E1 — Basic Capability

### Objective

Establish whether the model can correctly solve the controlled binary decision cases.

### Dataset

The initial pilot contains 50 balanced cases.

### Measurements

- accuracy;
- precision;
- recall;
- specificity;
- F1.

### Current result

The initial pilot produced 50 correct decisions out of 50 cases.

This result is treated as a baseline and not as evidence of broad generalization.

---

## E2 — Calibration

### Objective

Determine whether the probability output contains useful information about correctness.

### Measurements

- Brier score;
- log loss;
- ECE;
- reliability diagrams;
- confidence distributions.

### Current result

The initial 50-case pilot produced:

```text
Brier score:       0.002046
Log loss:          0.023419
ECE (5 bins):      0.022200
ECE (10 bins):     0.022200
Mean confidence:   0.977800
```

The small sample size and concentration of probabilities near the extremes limit the conclusions that can be drawn from this baseline.

---

## E3 — Selective Prediction

### Objective

Determine whether JEV confidence can identify cases where the system should abstain or defer.

### Planned analysis

Sort cases by confidence and measure risk at different coverage levels.

The analysis will examine the trade-off between:

```text
Coverage ↔ Error
```

The objective is to determine whether reducing the number of accepted decisions produces a predictable reduction in error.

---

## E4 — Counterfactual Stability

### Objective

Measure whether semantically equivalent inputs produce stable decision signals.

### Design

Each base case will have one or more semantic-preserving variants.

Possible perturbations include:

- paraphrasing;
- sentence restructuring;
- changes in irrelevant context;
- formatting changes;
- lexical substitution.

The ground-truth label remains unchanged.

### Measurements

For an original probability `p₀` and variant probability `p₁`:

```text
Δp = |p₀ - p₁|
```

Also record whether the binary decision changes.

---

## E5 — Adversarial Robustness

### Objective

Determine how easily the decision signal can be changed by targeted input modifications.

### Design

Construct perturbations that attempt to alter the probability without changing the intended ground-truth outcome.

The evaluation will distinguish between:

- semantic-preserving perturbations;
- misleading but still truth-preserving context;
- explicitly adversarial constructions.

---

## E6 — Language Shift

### Objective

Evaluate whether the observed reliability transfers across languages.

### Design

Create equivalent decision cases across selected languages while preserving the underlying proposition.

The exact language set will be finalized before data collection.

### Measurements

- accuracy;
- Brier score;
- log loss;
- ECE;
- confidence distribution;
- decision-flip rate for equivalent cases.

---

## E7 — Domain Shift

### Objective

Measure the effect of moving away from the domains represented in the initial pilot.

### Design

Evaluate equivalent decision structures across additional domains.

The analysis will compare both classification performance and probability calibration.

---

## E8 — JEV Controller

### Objective

Determine whether JEV can be used as a control mechanism for downstream generative inference.

### Basic policy

A configurable controller will map JEV confidence to an action:

```text
High confidence
      ↓
Accept / proceed

Intermediate confidence
      ↓
Escalate

Low confidence
      ↓
Abstain or escalate
```

The exact thresholds will be determined from the calibration experiments rather than selected after observing final test results.

### Measurements

- task reliability;
- coverage;
- escalation rate;
- abstention rate;
- expensive-model calls;
- token usage;
- estimated cost.

---

## E9 — LLM Router Baseline

### Objective

Provide a comparison against a routing strategy based on a generative language model.

The baseline will use a documented routing prompt and fixed decision policy.

The comparison will focus on:

- reliability;
- routing behaviour;
- computation;
- cost.

No assumption is made beforehand about which approach will perform better.

---

## E10 — Uncertainty Budget

### Objective

Study the relationship between a fixed computation budget and achieved reliability.

The controller will be evaluated under different allowed escalation budgets.

Example:

```text
Escalation budget
      ↓
0% ─── 10% ─── 25% ─── 50%
      ↓
Reliability / Coverage / Cost
```

This experiment is intended to characterize the computation-reliability trade-off.

---

## E11 — Ablation

### Objective

Determine which design choices materially affect the results.

Potential ablations include:

- probability thresholds;
- decision representations;
- state context;
- question decomposition;
- batching;
- controller hysteresis;
- escalation policy.

Each ablation will be compared against a fixed reference configuration.

---

## E12 — Reproducibility

### Objective

Determine whether repeated evaluations produce stable results.

Where the API and model configuration permit repeated evaluation, the same cases will be evaluated multiple times.

Measurements will include:

- probability variance;
- decision agreement;
- calibration variation;
- metric variation.

---

## E13 — Decision Representation Sensitivity

### Objective

Determine whether different ways of expressing the same binary decision produce different probabilities.

Examples include:

```text
Question:
"Was the package delivered?"

Equivalent criterion:
"Was the package delivered according to the tracking record?"
```

The underlying decision remains unchanged.

The resulting probability distributions will be compared.

---

## E14 — State Manipulation

### Objective

Measure sensitivity to changes in the model state that should not alter the intended decision.

The experiment will distinguish between:

- relevant state changes;
- irrelevant state changes;
- conflicting contextual information;
- adversarial contextual additions.

The purpose is to identify whether the decision signal depends on state features beyond those required to solve the task.

---

## E15 — Question Decomposition

### Objective

Study whether the way a decision problem is decomposed affects the output.

For related questions, compare:

```text
Single combined decision
```

against:

```text
Question A
Question B
Question C
```

and compare those with batched representations where appropriate.

The underlying information and evaluation objective should remain comparable.

---

## E16 — Temporal Stability

### Objective

Determine whether repeated decisions remain stable enough for use in a control loop.

The analysis will examine:

- probability drift;
- decision agreement;
- repeated-call variance;
- threshold crossings;
- hysteresis behaviour.

A controller that repeatedly switches between actions because of small probability changes may require a stability mechanism.

---

## Experimental Dependencies

The experiments are staged because later measurements depend on earlier validation.

```text
E1/E2
  │
  ├──→ E3
  │
  └──→ E4
         │
         ├──→ E5
         ├──→ E6
         ├──→ E7
         └──→ E13–E16
                    │
                    ▼
                  E8
                    │
              ┌─────┴─────┐
              ▼           ▼
             E9          E10
              │
              ▼
             E11/E12
```

The current next experiment is **E4 — Counterfactual Stability**.

---

## Status Convention

Experiments use the following status labels:

- **Complete** — implemented and executed;
- **In progress** — implementation or data collection underway;
- **Planned** — methodology defined but not yet executed;
- **Blocked** — requires an unresolved dependency or decision.