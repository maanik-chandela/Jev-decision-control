# Reproducibility

This document describes the procedures used to reproduce the experiments in this repository.

The project separates API execution from analysis so that statistical results and figures can be regenerated from recorded experiment outputs.

---

## Environment

The project is developed and tested on macOS.

Python dependencies are specified in:

```text
requirements.txt
```

A local virtual environment is recommended.

Example setup:

```bash
cd ~/JEV-Research/jev-decision-control

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## API Configuration

The JEV API key is stored locally in:

```text
.env
```

The environment variable used by the project is:

```text
TYPESAFE_API_KEY
```

The `.env` file is excluded from version control.

API credentials must never be committed to the repository.

---

## Experiment Execution

Experiments are executed from the repository root.

Before running an experiment:

```bash
cd ~/JEV-Research/jev-decision-control
source .venv/bin/activate
```

The API connectivity test can be run with:

```bash
python src/test_connection.py
```

The pilot runner supports selecting a range of cases.

For example:

```bash
python experiments/pilot_runner.py --start 1 --end 5
```

The exact command and configuration for each experiment should be recorded when the experiment is finalized.

---

## Data Separation

The repository distinguishes between public research artifacts and private execution outputs.

### Public

The following may be committed:

- experiment definitions;
- public input datasets;
- analysis scripts;
- plotting scripts;
- aggregate results;
- figures;
- documentation.

### Private

The following should remain local:

- API keys;
- raw API responses;
- private logs;
- temporary files;
- local virtual environments.

These files are excluded through `.gitignore`.

---

## Result Generation

The current analysis pipeline follows:

```text
pilot_questions.csv
        │
        ▼
   pilot_runner.py
        │
        ▼
pilot_results.csv
        │
        ├──────────────┐
        ▼              ▼
analyze_pilot.py   plot_pilot.py
        │              │
        ▼              ▼
pilot_analysis.csv   figures
```

The raw result files are not committed to the public repository.

Aggregate analysis outputs and figures may be committed when they do not expose private information or API credentials.

---

## Cost Control

API usage is monitored by the experiment infrastructure.

The project uses an internal budget guard to prevent accidental large-scale execution.

The cost calculation is based on the API input token count and the current documented input-token price.

Output tokens are treated according to the pricing configuration used by the project.

The internal budget is a software safeguard. It should not be considered a replacement for provider-side billing controls.

---

## Reproducible Analysis

Analysis scripts should not modify the original input dataset.

Instead, they read experiment outputs and generate separate analysis artifacts.

For example:

```bash
python experiments/analyze_pilot.py
```

and:

```bash
python experiments/plot_pilot.py
```

The generated tables and figures are written to the experiment results directory.

---

## Versioning

Each major experimental milestone should be associated with a Git commit.

The project uses descriptive commit messages such as:

```text
feat: establish baseline JEV pilot evaluation
feat: add calibration analysis
feat: add counterfactual stability evaluation
docs: add experimental methodology
test: validate pilot dataset schema
fix: correct experiment output paths
```

Major research milestones may also receive version tags.

---

## Reproducibility Record

For each finalized experiment, the following information should be recorded:

```text
Experiment ID:
Date:
Code version:
Model:
Model version:
Dataset version:
Number of cases:
Random seed:
API configuration:
Input token count:
Estimated cost:
Analysis script:
Result files:
```

If a parameter does not apply to an experiment, it should be explicitly marked as not applicable rather than omitted.

---

## Repeated Runs

When an experiment depends on stochastic or externally served model behaviour, repeated runs should be treated as separate observations.

The analysis should report:

- number of runs;
- agreement between runs;
- probability variation;
- metric variation.

A single successful execution should not automatically be described as deterministic.

---

## Dataset Versioning

Once a dataset is used for a published result, its contents should not be silently changed.

If corrections are required:

1. document the change;
2. update the dataset version;
3. rerun dependent analyses;
4. record the change in Git history.

This prevents changes to evaluation data from being confused with changes in model performance.

---

## Reproducibility Goal

The long-term goal is that an independent researcher should be able to:

1. inspect the methodology;
2. recreate the experiment inputs;
3. configure their own API credentials;
4. execute the experiment;
5. regenerate the aggregate analysis;
6. regenerate the figures;
7. compare their results with the reported results.

Where external API behaviour prevents exact reproduction, the limitation will be documented explicitly.