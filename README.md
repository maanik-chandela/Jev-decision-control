# Decision-Native Control of Generative AI Inference

**Research project investigating whether structured decision signals can be used to control when generative AI systems should answer, defer, or abstain.**

---

## Overview

Large language models are increasingly used in settings where producing an answer is only one part of the problem. A system may also need to decide whether it has enough confidence to answer, whether a stronger model should be invoked, or whether a request should be deferred altogether.

This project studies that decision layer separately from generation.

The current work uses **JEV**, a System One decision model developed by TypeSafe AI, as an experimental case study. Rather than treating JEV simply as another classifier, the project evaluates whether its structured probability output can serve as a reliable control signal for downstream generative inference.

The broader evaluation framework developed in this project is called **DART — Decision-native AI Reliability Testing**.

The initial experiments focus on calibration and reliability. Later experiments will examine counterfactual stability, distribution shift, state manipulation, decision representation, and computation-aware routing.

---

## Research Question

> Can a decision-native model such as JEV provide a reliable control signal for deciding when an AI system should stop, escalate to a stronger model, or abstain—and does that signal remain reliable under semantic-preserving perturbations and distribution shift?

The project is therefore concerned with two related questions:

1. **How reliable is the decision signal itself?**
2. **Can that signal be used to make downstream inference more selective without unnecessarily increasing computation?**

---

## Working Hypotheses

The experiments are designed around the following hypotheses:

- **H1 — Calibration:** JEV's probability output is informative about the correctness of its decisions, but calibration may vary across tasks and domains.
- **H2 — Counterfactual stability:** semantically equivalent changes in wording may alter the decision probability even when the underlying answer remains unchanged.
- **H3 — Distribution shift:** calibration and reliability may degrade when the language, domain, or task distribution changes.
- **H4 — Computation allocation:** a confidence-aware controller can reduce unnecessary calls to an expensive generative model while maintaining a target reliability level.
- **H5 — Decision representation:** the representation used to express a decision can affect the resulting probability.
- **H6 — State robustness:** changes to irrelevant or adversarial parts of the model state may influence the decision signal.
- **H7 — Decomposition sensitivity:** splitting or batching questions differently may affect decision behaviour.
- **H8 — Temporal stability:** repeated decisions under otherwise equivalent conditions should remain sufficiently stable for use in a control loop.

These hypotheses will be tested independently rather than assuming that a strong result on one property implies reliability on the others.

---

## Evaluation Framework

DART evaluates a decision-native model along several dimensions:

```text
                ┌─────────────────────┐
                │  Decision-Native    │
                │       Model         │
                └──────────┬──────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
     Calibration      Stability       Robustness
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
                 Control / Routing
                           │
                           ▼
              Generative AI Inference
```

The evaluation is not limited to accuracy. A model can classify a fixed benchmark correctly while still producing a poorly calibrated or unstable confidence signal.

The main measurements therefore include:

- accuracy
- precision
- recall
- specificity
- F1
- Brier score
- log loss
- expected calibration error (ECE)
- confidence distributions
- probability separation
- selective prediction behaviour
- counterfactual decision stability
- distribution-shift robustness
- state-manipulation robustness
- computation allocation

---

## Current Status

### Phase 1 — Baseline Pilot Evaluation

The first pilot consists of **50 hand-designed binary decision cases**.

The dataset contains:

- 25 positive cases
- 25 negative cases
- mathematics
- science
- general knowledge
- reasoning

The pilot was intentionally kept small and controlled. Its purpose is to verify the API pipeline, data representation, logging, analysis, and initial reliability measurements before moving to more demanding experiments.

### Pilot results

| Metric | Result |
|---|---:|
| Cases | 50 |
| Correct | 50 |
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| Specificity | 1.0000 |
| F1 | 1.0000 |
| Brier score | 0.002046 |
| Log loss | 0.023419 |
| ECE (5 bins) | 0.022200 |
| ECE (10 bins) | 0.022200 |
| Mean confidence | 0.977800 |
| Probability separation | 0.620000 |
| Confidently wrong cases | 0 |
| Uncertain cases | 0 |

The results show that JEV separated the cases in this small pilot very strongly. They should **not** be interpreted as evidence of general reliability. The pilot contains relatively simple, hand-designed cases and does not yet include the perturbations or distribution shifts that motivate the main research question.

The next experiments therefore move away from simply increasing the number of similar examples.

---

## Experiments

The planned experiment matrix currently includes:

| ID | Experiment | Purpose |
|---|---|---|
| E1 | Basic capability | Establish baseline decision performance |
| E2 | Calibration | Measure whether probabilities correspond to empirical correctness |
| E3 | Selective prediction | Examine confidence-based abstention |
| E4 | Counterfactual stability | Test semantic-preserving perturbations |
| E5 | Adversarial robustness | Test sensitivity to targeted changes |
| E6 | Language shift | Evaluate behaviour across languages |
| E7 | Domain shift | Evaluate behaviour outside the original task distribution |
| E8 | JEV controller | Use JEV to control downstream inference |
| E9 | LLM router baseline | Compare against a generative-model routing approach |
| E10 | Uncertainty budget | Study computation under reliability constraints |
| E11 | Ablation | Identify which components affect performance |
| E12 | Reproducibility | Measure repeated-run consistency |
| E13 | Decision-type sensitivity | Test alternative decision representations |
| E14 | State manipulation | Test sensitivity to changes in model state |
| E15 | Question decomposition | Study effects of decomposition and batching |
| E16 | Temporal stability | Examine repeated decisions and hysteresis |

The experiments will be added incrementally as their methodology and data collection procedures are finalized.

---

## Repository Structure

```text
jev-decision-control/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── budget.py
│   └── test_connection.py
│
├── experiments/
│   ├── day1_first_decision.py
│   ├── pilot_runner.py
│   ├── analyze_pilot.py
│   ├── plot_pilot.py
│   │
│   ├── data/
│   │   ├── pilot_questions.csv
│   │   └── day1_logger.py
│   │
│   └── results/
│       ├── pilot_confidence_distribution.png
│       ├── pilot_probability_by_difficulty.png
│       ├── pilot_probability_by_domain.png
│       ├── pilot_probability_distribution.png
│       ├── pilot_reliability_diagram.png
│       └── pilot_reliability_table.csv
│
├── docs/
├── configs/
└── tests/
```

Raw API outputs and credentials are intentionally excluded from version control.

---

## Reproducibility

The project is being developed with reproducibility in mind.

Experiments are separated into:

1. input datasets
2. API execution
3. result logging
4. statistical analysis
5. visualization

The Python environment is specified in `requirements.txt`.

API credentials are stored locally in `.env` and are excluded through `.gitignore`.

The repository does not contain private API responses or secret credentials.

---

## Cost Control

The JEV API is metered by input tokens. Since the project is being developed under a limited research budget, experiment runners include an internal cost guard.

The pilot was intentionally executed in small batches rather than sending a large dataset to the API at once.

The reported pilot execution cost was approximately **$0.00067**.

The cost-control mechanism is treated as a research reproducibility feature as well as a practical safeguard against accidental large-scale API usage.

---

## Research Direction

The central idea of this project is not to determine whether JEV is simply "better" than a conventional language model.

Instead, the question is whether a **decision-native model can provide a sufficiently reliable intermediate signal to control a more expensive generative system**.

This changes the evaluation target from:

```text
Question → Answer
```

to:

```text
Question
   ↓
Decision
   ↓
Confidence / Risk
   ↓
Control Policy
   ↓
Answer / Escalation / Abstention
```

The distinction is important because a control signal has requirements beyond ordinary classification accuracy. It must be calibrated, stable under irrelevant changes, robust under distribution shift, and useful when connected to an actual decision policy.

---

## Current Limitations

The current results have several important limitations:

- the initial dataset contains only 50 cases;
- the cases are hand-designed;
- most cases are relatively simple;
- the pilot does not establish performance under distribution shift;
- no claim about real-world routing performance is made yet;
- the controller experiments have not yet been conducted.

These limitations are intentional at this stage. The pilot is primarily an infrastructure and baseline evaluation rather than the final empirical study.

---

## Roadmap

### Completed

- [x] JEV API integration
- [x] Budget guard
- [x] Reproducible Python environment
- [x] Initial balanced pilot dataset
- [x] Automated pilot runner
- [x] Classification metrics
- [x] Calibration metrics
- [x] Reliability analysis
- [x] Pilot visualizations

### Next

- [ ] Counterfactual / semantic-preserving perturbation dataset
- [ ] Stability metrics
- [ ] Expanded calibration study
- [ ] Language-shift evaluation
- [ ] Domain-shift evaluation
- [ ] State-manipulation experiments
- [ ] Decision representation ablations
- [ ] JEV-based inference controller
- [ ] Comparison with an LLM routing baseline
- [ ] Full statistical analysis
- [ ] Reproducibility study
- [ ] Paper preparation

---

## Status

**Current milestone:** `v0.1.0 — Pilot Evaluation`

This repository is an ongoing research project. Results and methodology may change as experiments are added and weaknesses in the initial evaluation are identified.

---

## Citation

A formal citation entry will be added once the research paper and project metadata are finalized.

---

## License

See `LICENSE` for the terms under which this repository is distributed.