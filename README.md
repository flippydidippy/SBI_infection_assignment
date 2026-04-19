# SBI Infection Assignment

Simulation-Based Inference (SBI) coursework for **ST3247**, focused on recovering epidemic parameters in a stochastic **SIR** model evolving on an adaptive contact network.

## Overview

This repository studies inference for the unknown parameter triple

- `beta`: infection probability,
- `gamma`: recovery probability,
- `rho`: rewiring probability,

using partially observed epidemic data generated from an adaptive-network SIR model.

The observed data consist of

- infected fraction over time,
- rewiring counts over time,
- final degree histograms.

The repository is organised around the workflow of the report itself:

- **Q2**: baseline rejection ABC,
- **Q3**: summary-statistic design,
- **Q4**: advanced methods,
- **additional tools**: robustness checks, synthetic validation, and visualisation.

A useful design feature of this repository is that the main scripts remain straightforward to inspect and run, while shared components are reused where this improves consistency. In particular:

- `run_q3_Summary_Statistics.py` builds directly on the simulator and data-loading pipeline from `run_q2_Basic_Rejection_ABC.py`,
- `run_q4_Advanced_Methods.py` reuses the Q2 simulator and the Q3 `S4` summary statistic,
- and additional experiments are collected under `additional_tools/` rather than being mixed into the main analysis scripts.

Overall, this gives the repository a good balance between **clarity by analysis stage** and **reuse of common functionality**.

## Current repository layout

```text
SBI_infection_assignment/
├── additional_tools/
│   ├── robustness_study.py
│   ├── synthetic_validation.py
│   └── visual_demo_org_simulator.py
├── archive/
│   ├── fast_simulator_test/
│   ├── original_simulator/
│   └── old_4_ABC_MCMC.py
├── data/
│   ├── final_degree_histograms.csv
│   ├── infected_timeseries.csv
│   └── rewiring_timeseries.csv
├── results/
│   ├── figures/
│   │   ├── q2/
│   │   ├── q3/
│   │   ├── q4/
│   │   └── synthetic_validation.png
│   └── robustness/
│       ├── posterior_movement.csv
│       ├── robustness_summary.csv
│       └── run_config.json
├── README.md
├── requirements.txt
├── run_q2_Basic_Rejection_ABC.py
├── run_q3_Summary_Statistics.py
├── run_q4_Advanced_Methods.py
├── run_q4a_LLRA.py
└── run_q4d_ABC_MCMC.py
```

## What each file or folder does

### `run_q2_Basic_Rejection_ABC.py`
Baseline rejection-ABC workflow.

This script contains the fast Numba-based simulator, the observed-data loading and preprocessing pipeline, the `S1` summary statistic, prior sampling, scale estimation, rejection ABC, and posterior plots. It is the main script for the Q2 baseline analysis.

### `run_q3_Summary_Statistics.py`
Summary-statistic design and comparison.

This script reuses the Q2 simulator and data-loading code, then defines and compares the candidate summary sets `S1`–`S4`. It generates sensitivity plots and posterior comparisons to show which summaries are most informative for `beta`, `gamma`, and `rho`.

### `run_q4_Advanced_Methods.py`
Combined Q4 advanced-methods workflow.

This script is the most integrated version of the advanced analysis. It explicitly reuses:

- the **Q2 simulator** and observed-data pipeline, and
- the **Q3 `S4` summary statistic**,

before adding the advanced inference machinery, including ABC-MCMC, a same-budget rejection-ABC comparison, regression adjustment, diagnostics, and plotting.

### `run_q4a_LLRA.py`
Standalone local linear regression adjustment (LLRA) workflow.

This file focuses on the Beaumont-style regression-adjustment analysis. It is useful when running or inspecting the LLRA component separately from the larger Q4 script.

### `run_q4d_ABC_MCMC.py`
Standalone advanced MCMC-based workflow.

This file contains the more specialised Q4 MCMC-based analysis and is kept separately from the combined script for ease of inspection and reproducibility.

## Supporting folders

### `data/`
Observed datasets used throughout the analysis.

- `infected_timeseries.csv`: infected-fraction trajectories across replicates,
- `rewiring_timeseries.csv`: rewiring-count trajectories across replicates,
- `final_degree_histograms.csv`: final degree distributions across replicates.

### `results/`
Saved outputs from the main workflows and supporting studies.

#### `results/figures/`
Plots grouped by analysis stage.

- `q2/`: figures from 2. Basic Rejection ABC,
- `q3/`: figures from 3. Summary Statistics,
- `q4/`: figures from 4. Advanced Methods,
- `synthetic_validation.png`: saved synthetic-truth validation figure from 4. Advanced Methods.

#### `results/robustness/`
Outputs from the robustness study.

- `posterior_movement.csv`: movement of posterior summaries under alternative settings,
- `robustness_summary.csv`: compact robustness summary table,
- `run_config.json`: stored configuration for the robustness run.

### `additional_tools/`
Supporting scripts that are useful but not part of the main section-by-section workflow.

- `robustness_study.py`: runs the robustness analysis across tolerances, seeds, proposal scales, and distance metrics,
- `synthetic_validation.py`: runs synthetic-truth validation experiments,
- `visual_demo_org_simulator.py`: provides a visual demonstration of the original-style simulator.

### `archive/`
Older or reference material retained for transparency.

This folder is not part of the main workflow, but it preserves earlier simulator checks and superseded scripts.

- `fast_simulator_test/`: development and checking material related to the simulator,
- `original_simulator/`: older reference simulator code,
- `old_4_ABC_MCMC.py`: an earlier advanced-methods implementation retained for record-keeping.

## Design philosophy

This branch is intentionally organised around the report workflow rather than around a deep package hierarchy.

At the same time, it is **not** just a collection of fully isolated scripts. The code now has a clearer progression of reuse:

- Q2 defines the core fast simulator and the basic data/ABC pipeline,
- Q3 reuses those foundations and adds competing summary-statistic sets,
- Q4 builds on both earlier stages, especially the Q2 simulator and the Q3 `S4` summary.

This gives the repository three advantages:

1. **Direct alignment with the write-up**  
   Each main script corresponds closely to a report section.

2. **Reasonable code reuse without over-engineering**  
   Shared pieces are reused where this helps clarity, but the scripts remain readable on their own.

3. **Section-level reproducibility**  
   Each major stage of the analysis can still be run independently.

In short, the current structure is a deliberate compromise between **clarity**, **report alignment**, and **lightweight modularity**.

## How to run the analysis

Run the scripts from the repository root.

### Q2: Baseline rejection ABC

```bash
python run_q2_Basic_Rejection_ABC.py
```

### Q3: Summary-statistics comparison

```bash
python run_q3_Summary_Statistics.py
```

### Q4: Combined advanced-methods workflow

```bash
python run_q4_Advanced_Methods.py
```

### Q4a: Local linear regression adjustment only

```bash
python run_q4a_LLRA.py
```

### Q4d: Standalone advanced MCMC workflow

```bash
python run_q4d_ABC_MCMC.py
```

### Synthetic-truth validation

```bash
python additional_tools/synthetic_validation.py
```

### Robustness study

```bash
python additional_tools/robustness_study.py
```

### Original-simulator visual demo

```bash
python additional_tools/visual_demo_org_simulator.py
```

## Environment setup

Create and activate a virtual environment, then install the required packages.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Requirements

The current `requirements.txt` lists:

- `numpy>=1.24`
- `pandas>=2.0`
- `matplotlib>=3.7`
- `numba>=0.58`

## Notes

- Python 3.10+ is recommended.
- The repository is meant to be read alongside the report.
- The main analysis scripts are top-level on purpose, so the primary workflows are immediately visible.
- Supporting experiments and older development material are separated into `additional_tools/` and `archive/` to keep the main workflow cleaner.
