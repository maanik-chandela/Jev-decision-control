# Research Plan

## Working Title

**Decision-Native Control of Generative AI Inference: Counterfactual Stability, Calibration, and Risk-Aware Computation Allocation with JEV**

---

## Motivation

Generative AI systems are increasingly being used in workflows where answering a question is not necessarily the correct first action.

For some inputs, a system may be sufficiently certain to answer directly. For others, it may be preferable to use a stronger model, request additional information, or abstain.

This creates a decision problem around the generative model.

The purpose of this research is to investigate whether a decision-native model can provide a useful and reliable signal for making those decisions.

JEV is used as the first publicly accessible System One model in the study.

---

## Research Question

> Can a decision-native model such as JEV provide a reliable control signal for deciding when an AI system should stop, escalate to a stronger model, or abstain—and does that signal remain reliable under semantic-preserving perturbations and distribution shift?

---

## Objectives

The study has four primary objectives.

### 1. Measure decision reliability

Determine whether JEV's probability output corresponds to the empirical correctness of its decisions.

### 2. Measure decision stability

Determine whether semantically equivalent changes to an input produce substantially different decisions or confidence values.

### 3. Evaluate robustness

Measure how reliability changes under language shift, domain shift, state manipulation, and alternative decision representations.

### 4. Evaluate practical control

Determine whether the decision signal can be used to control access to a more expensive generative model.

---

## Hypotheses

### H1 — Calibration

JEV's probability output should contain information about the likelihood that its decision is correct, although calibration may vary across tasks and domains.

### H2 — Counterfactual Stability

Semantically preserving perturbations should generally preserve the underlying decision. Large changes in probability under such perturbations would indicate sensitivity that could matter for downstream control.

### H3 — Distribution Shift

Calibration and reliability may degrade when the evaluation distribution differs from the distribution used to construct the initial pilot.

### H4 — Computation Allocation

A confidence-aware controller should be able to reduce unnecessary generative-model calls while maintaining a predefined reliability target.

### H5 — Decision Representation Sensitivity

Changing the representation of a decision may affect the resulting probability even when the underlying task remains equivalent.

### H6 — State Manipulation Robustness

Changes to irrelevant or adversarial portions of the model state may influence the decision signal.

### H7 — Question Decomposition Sensitivity

Different ways of decomposing or batching related questions may produce different decision behaviour.

### H8 — Temporal Stability

Repeated decisions under equivalent conditions should remain sufficiently stable for the signal to be useful in a control policy.

---

## Study Phases

### Phase 1 — Baseline Pilot

Purpose:

- verify API integration;
- establish the data schema;
- establish automated logging;
- measure basic classification performance;
- obtain initial calibration measurements.

The initial pilot contains 50 balanced binary decision cases.

### Phase 2 — Calibration

Expand the evaluation set to include more varied difficulty and probability ranges.

Measure:

- Brier score;
- log loss;
- expected calibration error;
- reliability diagrams;
- confidence versus correctness;
- selective prediction behaviour.

### Phase 3 — Counterfactual Stability

Construct semantic-preserving variants of the same underlying cases.

Examples include changes to:

- wording;
- sentence structure;
- irrelevant contextual phrasing;
- ordering;
- formatting.

The underlying truth value remains unchanged.

The primary measurement is the change in JEV probability and decision between the original and perturbed versions.

### Phase 4 — Distribution Shift

Evaluate the decision signal under changes in:

- language;
- domain;
- task structure.

The objective is to determine whether reliability measured on the initial distribution transfers to other distributions.

### Phase 5 — Robustness and Representation

Study:

- state manipulation;
- decision representation;
- question decomposition;
- batching;
- repeated evaluation.

### Phase 6 — Control

Connect JEV to a downstream generative model.

A controller will determine whether to:

1. answer using the available system;
2. invoke a stronger model;
3. abstain or request additional information.

The controller will be evaluated using both reliability and computation-related measurements.

---

## Evaluation Principle

The study does not treat high classification accuracy as sufficient evidence of a useful control signal.

A decision signal used for control should satisfy several properties simultaneously:

```text
Correct
   +
Calibrated
   +
Stable
   +
Robust
   +
Useful for control
```

The experiments are therefore designed to evaluate these properties separately before combining them in the final controller.

---

## Current Scope

The current implementation focuses on JEV as the first experimental System One model.

The framework is intended to remain model-agnostic where possible so that later experiments can compare different decision mechanisms without changing the evaluation methodology.

---

## Expected Outcome

The primary research outcome is an empirical characterization of when decision-native confidence can and cannot be treated as a reliable control signal for generative inference.

The study does not assume beforehand that JEV will satisfy all reliability requirements.

Negative or mixed results are considered useful outcomes because they identify conditions under which decision-native control should not be trusted.