# SBI Infection Assignment

Simulation-Based Inference coursework for **ST3247**, focused on parameter recovery in a stochastic **SIR** epidemic model evolving on an adaptive contact network.

## Overview

This repository studies inference for the unknown parameter triple

- `beta`: infection probability,
- `gamma`: recovery probability,
- `rho`: rewiring probability,

using partially observed epidemic data from an adaptive-network SIR model.

The observed data consist of

- infected fraction over time,
- rewiring counts over time, and
- final degree histograms.

The repository is organised to follow the structure of the report itself, so that each main directory corresponds closely to a major section of the write-up and can be run and inspected on its own.

## Repository structure

```text
SBI_infection_assignment/
├── 2_Basic_Rejection_ABC/
│   └── 2_basic_ABC.py
├── 3_Summary_Statistics/
│   └── 3_summary_statistics.py
├── 4_Advanced_Methods/
│   ├── 4_ABC_MCMC.py
│   ├── 4a.py
│   └── 4d.py
├── archive/
│   ├── fast_simulator_test/
│   ├── original_simulator/
│   └── old_4_Advanced_methods.py
├── data/
│   ├── final_degree_histograms.csv
│   ├── infected_timeseries.csv
│   └── rewiring_timeseries.csv
├── figures/
│   ├── q2/
│   ├── q3/
│   └── q4/
├── results/
│   └── robustness/
│       ├── posterior_movement.csv
│       ├── robustness_summary.csv
│       └── run_config.json
├── tools/
│   ├── robustness_study.py
│   ├── synthetic_validation.py
│   └── visual_demo_org_simulator.py
├── README.md
└── requirements.txt
```

## What each part does

### `2_Basic_Rejection_ABC/`
Contains the baseline rejection ABC implementation used in the first main inference section of the report.

- `2_basic_ABC.py` runs the basic rejection ABC pipeline on the observed data and produces the baseline posterior results.

### `3_Summary_Statistics/`
Contains the summary-statistic design and comparison code.

- `3_summary_statistics.py` compares alternative summary statistics and studies how the choice of summaries affects posterior recovery and parameter identifiability.

### `4_Advanced_Methods/`
Contains the more advanced inference methods developed after the basic ABC baseline.

- `4a.py` implements the regression-adjustment part of the advanced methods section.
- `4_ABC_MCMC.py` implements the ABC-MCMC approach.
- `4d.py` implements the synthetic-likelihood / advanced comparison component used in the later part of the analysis.

### `archive/`
Contains older, exploratory, or superseded material that was kept for transparency but separated from the main workflow.

- `fast_simulator_test/` contains development or checking material related to simulator speed-ups.
- `original_simulator/` stores the older reference simulator material.
- `old_4_Advanced_methods.py` is an earlier combined advanced-methods script retained for record-keeping.

This folder is not part of the main submission workflow, but is kept as supporting development history.

### `data/`
Contains the observed datasets used by the inference scripts.

- `infected_timeseries.csv`: infected fraction trajectories across replicates.
- `rewiring_timeseries.csv`: rewiring-count trajectories across replicates.
- `final_degree_histograms.csv`: final degree distributions across replicates.

### `figures/`
Stores generated plots grouped by report section.

- `q2/`: figures from the basic rejection ABC section.
- `q3/`: figures from the summary-statistics section.
- `q4/`: figures from the advanced-methods section.

### `results/robustness/`
Stores outputs from the robustness study.

- `posterior_movement.csv`: movement of posterior summaries under robustness changes.
- `robustness_summary.csv`: compact robustness summary table.
- `run_config.json`: configuration used for the robustness run.

### `tools/`
Contains supporting scripts used for validation, robustness checks, and simulator demonstrations.

- `robustness_study.py` runs the robustness analysis and writes outputs to `results/robustness/`.
- `synthetic_validation.py` runs synthetic-truth validation experiments.
- `visual_demo_org_simulator.py` provides a simple demonstration/visualisation for the original simulator.

## Why some code is repeated across files

Some functionality appears more than once across the main scripts. In a larger software project, this could certainly be refactored into a more modular shared package.

For this coursework repository, however, the main scripts were intentionally kept **largely self-contained by section**.

This design was chosen for several practical reasons:

1. **Direct alignment with the report**  
   Each major script corresponds closely to one section of the write-up, so the reader can inspect the code for that section without jumping across many files.

2. **Ease of marking and readability**  
   A marker can open a single script and see the full workflow for that section in one place: simulator assumptions, summaries, priors, inference routine, and plotting logic.

3. **Section-by-section reproducibility**  
   Each section can be run independently. This made it easier during development to modify one method without unintentionally breaking the others.

4. **Transparency over abstraction**  
   For an assignment submission, having the full logic visible in each main file can be clearer than hiding important steps behind a deeper import structure.

So while the repository is not maximally refactored, the repeated code reflects a deliberate trade-off in favour of **clarity, self-containment, and report-aligned reproducibility**.

## How to run

Run the main scripts from the repository root.

### Basic rejection ABC

```bash
python 2_Basic_Rejection_ABC/2_basic_ABC.py
```

### Summary-statistics comparison

```bash
python 3_Summary_Statistics/3_summary_statistics.py
```

### Advanced methods

```bash
python 4_Advanced_Methods/4a.py
python 4_Advanced_Methods/4_ABC_MCMC.py
python 4_Advanced_Methods/4d.py
```

### Robustness study

```bash
python tools/robustness_study.py
```

### Synthetic validation

```bash
python tools/synthetic_validation.py
```

## Outputs

- Figures are stored in `figures/q2`, `figures/q3`, and `figures/q4`.
- Robustness outputs are stored in `results/robustness/`.
- Additional outputs depend on the individual script being run.

## Environment setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Recommended Python version

Python **3.10+** is recommended.

## Notes

- The repository structure is intentionally close to the structure of the report.
- The main scripts prioritise methodological clarity and section-level reproducibility.
- Supporting and older material has been separated into `archive/` and `tools/` so that the primary workflow remains easy to follow.

---

## Suggested `requirements.txt`

You can copy this into `requirements.txt`:

```txt
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
numba>=0.58
```
