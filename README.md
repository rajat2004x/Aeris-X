# AERIS-X
### Aerospace Health Intelligence Platform

**An AI-driven aerospace health monitoring and prognostics platform for aircraft engine telemetry, predictive maintenance, anomaly detection, sensor fault isolation, and operational intelligence.**

<p align="center">
  <strong>Research • Machine Learning • Deep Learning • Aerospace Intelligence • MLOps</strong>
</p>

---

## Overview

AERIS-X is a research-oriented aerospace health intelligence platform designed to investigate how machine learning and deep learning can support aircraft engine health monitoring and predictive maintenance.

The platform uses NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) datasets to investigate engine degradation, estimate remaining useful life (RUL), evaluate health-state trajectories, detect anomalous behavior, and study sensor fault isolation.

Beyond predictive modeling, AERIS-X is being developed as a modular engineering platform with data pipelines, experiment tracking, model interfaces, persistence, monitoring, APIs, and intelligence contracts.

The project emphasizes **reproducibility, rigorous evaluation, transparent limitations, and evidence-based engineering** rather than reporting artificially improved model performance.

## Key Capabilities

| Capability | Description |
|---|---|
| RUL Prognostics | Predict remaining useful life from sequential engine telemetry |
| Health-State Estimation | Estimate engine health and degradation trajectories |
| Temporal Deep Learning | Transformer-based temporal modeling |
| Classical ML Benchmarking | XGBoost, Random Forest, and Linear Regression |
| Sensor Fault Isolation | Investigate bias, drift, stuck-at, noise, and intermittent dropout |
| Anomaly Detection | Evaluate statistical and reconstruction-based anomaly detectors |
| Cross-Dataset Evaluation | Investigate generalization across FD001–FD004 |
| Digital Twin & Fault Injection | Develop synthetic fault scenarios for controlled evaluation |
| Spacecraft Telemetry Research | Explore SMAP and MSL telemetry datasets |
| Experiment Reproducibility | Store manifests, configurations, metrics, predictions, and reports |

## System Architecture

AERIS-X follows a modular architecture that separates data processing, modeling, evaluation, intelligence, persistence, and API functionality.

```text
             AEROSPACE TELEMETRY
                      |
                      v
             DATA INGESTION
                      |
                      v
          PREPROCESSING PIPELINE
                      |
                      v
        FEATURE ENGINEERING & SEQUENCES
                      |
                      v
              MODEL LAYER
          /       |        \
         v        v         v
      RUL      HEALTH     ANOMALY
   PROGNOSIS   ESTIMATION DETECTION
          \       |        /
           \      |       /
                    v
          SENSOR FAULT ISOLATION
                    |
                    v
          INTELLIGENCE & DIAGNOSIS
                    |
                    v
          EVALUATION & MONITORING
                    |
                    v
          PERSISTENCE & REST APIs
                    |
                    v
          OPERATIONAL INTERFACE
```

The architecture is modular and under active development. Not every component is currently a production-ready end-to-end workflow.

## Technology Stack

**Programming and ML**
- Python 3.11
- NumPy and Pandas
- Scikit-learn
- XGBoost
- PyTorch

**Deep Learning**
- Transformer architectures
- LSTM
- GRU
- Autoencoders

**Backend and Platform**
- FastAPI
- Pydantic
- SQLAlchemy
- PostgreSQL integration foundation

**Experimentation and Engineering**
- MLflow / DVC integration planning
- Docker
- GitHub Actions
- Pytest
- Git LFS

## Research Datasets

### NASA C-MAPSS

The primary research dataset is NASA C-MAPSS, containing simulated aircraft engine degradation trajectories.

| Dataset | Operating Conditions | Fault Conditions |
|---|---|---|
| FD001 | 1 | 1 |
| FD002 | 6 | 1 |
| FD003 | 1 | 2 |
| FD004 | 6 | 2 |

Research includes:
- Telemetry preprocessing
- Engine-wise sequence generation
- RUL label generation
- Train, development, and final evaluation splits
- Operating-condition analysis
- Cross-dataset generalization

### Additional Telemetry Research

SMAP and MSL spacecraft telemetry datasets have also been explored for anomaly detection and spacecraft health intelligence.

N-CMAPSS acquisition remains subject to data-access availability.

## Research Results

The following results are recorded from the project's experiments. They represent specific evaluation settings, not universal performance guarantees.

### Classical ML Benchmark — Step 7

| Model | MAE |
|---|---:|
| XGBoost | 18.6983 |
| Random Forest | 20.8728 |
| Linear Regression | 23.1157 |

Mean RUL baseline: 48.9.

### Temporal Deep Learning — Step 8

The Transformer temporal model achieved:

| Metric | Result |
|---|---:|
| MAE | 17.2109 |
| RMSE | 24.5902 |
| PHM Score | 61942.23 |

### Health-State Estimation — Step 12

FD001 health-state estimation achieved an MAE of **0.05505**.

Cross-domain transfer experiments across FD001–FD004 demonstrated substantial generalization limitations. Additional improvement experiments did not establish a superior candidate.

### Sensor Fault Detection — Step 10.1

The held-out statistical detector evaluation reported:

| Metric | Result |
|---|---:|
| Precision | 0.08 |
| Recall | 0.5854 |
| F1 Score | 0.1408 |
| Accuracy | 0.7838 |
| False Positive Rate | 0.21 |
| ROC-AUC | 0.7095 |

The experiments demonstrate that normal engine degradation can be confused with sensor faults.

### Anomaly Detection — Step 11

The anomaly detection experiments identified weak detection performance and limitations in recall. These findings are retained rather than presented as successful production detection.

### LSTM / GRU — Step 13

LSTM and GRU experiments were implemented and evaluated against the Transformer baseline.

Neither model established a better overall candidate. GRU remains a compact alternative worth documenting.

**Research integrity:** Negative findings, cross-domain degradation, false positives, and detector limitations are preserved in the experiment reports.

## Repository Structure

```text
AERIS-X/
│
├── .github/                  # CI workflows
├── .aeris/                   # Project state and roadmap
│
├── data/                     # Dataset-related resources
│
├── models/                   # Model artifacts and interfaces
│
├── reports/
│   └── experiments/
│       ├── step7_classical_ml/
│       ├── step8_temporal_dl/
│       ├── step9_comparison/
│       ├── step10_sensor_health_fdi/
│       ├── step11_anomaly_detection/
│       ├── step12_health_state/
│       ├── step12_health_state_improvement/
│       ├── step12_health_state_multidataset/
│       └── step13_lstm_gru/
│
├── scripts/                  # Experiment execution scripts
│
├── src/
│   └── aeris/
│       ├── anomaly/
│       ├── api/
│       ├── config/
│       ├── core/
│       ├── data/
│       ├── db/
│       ├── digital_twin/
│       ├── domain/
│       ├── evaluation/
│       ├── experiment/
│       ├── fdi/
│       ├── generalization/
│       ├── health/
│       ├── intelligence/
│       ├── models/
│       ├── monitoring/
│       ├── preprocessing/
│       ├── repositories/
│       ├── services/
│       └── spacecraft/
│
├── tests/                    # Automated tests
├── .gitattributes            # Git LFS configuration
├── pyproject.toml            # Project configuration
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.11
- Git
- Git LFS
- A compatible environment for PyTorch

### Clone the repository

```bash
git clone https://github.com/rajat2004x/Aeris-X.git
cd Aeris-X
```

### Configure Git LFS

```bash
git lfs install
git lfs pull
```

### Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install dependencies

Install dependencies using the repository's dependency configuration. For example, if the project is configured for editable installation:

```bash
pip install -e .
```

If optional development dependencies are defined separately, install those according to the project configuration.

### Run tests

```bash
pytest
```

### Validate project roadmap

```bash
python scripts/validate_roadmap.py
```

### Run experiments

Experiment scripts are available under `scripts/`. For example:

```bash
python scripts/run_step12_health_state_multidataset.py
python scripts/run_step13_lstm_gru.py
```

Check each script's configuration and data requirements before execution.

## Experiment Reproducibility

AERIS-X maintains experiment artifacts to support reproducibility and evaluation.

These include:
- Experiment configuration files
- Dataset characterization
- Preprocessing and split manifests
- Leakage audits
- Model checkpoints
- Training histories
- Prediction arrays
- Evaluation metrics
- CSV and JSON result files
- Visualization outputs
- Research reports

Large processed NumPy arrays are stored using **Git LFS**.

## Current Development Status

The project roadmap tracks completed milestones, the current research milestone, and future platform development.

| Milestone | Area | Status |
|---|---|---|
| 1–6 | Data foundation and preprocessing | Completed |
| 7–9 | Classical ML, temporal DL, comparison | Completed |
| 10–10.2 | Sensor health and FDI | Completed |
| 11 | Anomaly detection | Completed |
| 12 | Health-state estimation and generalization | Completed |
| 13 | LSTM / GRU evaluation | Research implemented; formal roadmap gate pending |
| 14–16 | Platform and intelligence development | Implemented work exists; roadmap gates pending |
| 17–30 | Future development | Blocked pending earlier gates |

The roadmap validator is the source of truth for milestone status.

## Known Limitations

AERIS-X is an active research and engineering project, not a certified aerospace safety system.

Current limitations include:

- Cross-dataset generalization remains challenging.
- Sensor fault detection produces false positives.
- Anomaly detection performance requires further improvement.
- LSTM and GRU have not demonstrated superiority over the Transformer baseline.
- Some intelligence components remain experimental.
- Production deployment, database validation, and operational integration require further verification.
- N-CMAPSS data acquisition remains unresolved.

Predictions and experimental outputs must not be interpreted as flight-certified diagnostics or operational maintenance decisions.

## Future Direction

Planned development includes:

- Completing formal research milestone gates.
- Improving fault isolation and unknown-fault detection.
- Advancing digital-twin-based fault injection.
- Strengthening temporal health estimation and prognostics.
- Completing backend persistence and operational APIs.
- Improving model lifecycle management and monitoring.
- Expanding reproducible evaluation across aerospace telemetry datasets.
- Advancing deployment, observability, security, and CI/CD.

## Project Philosophy

AERIS-X is built around four principles:

1. **Evidence over assumptions** — report measured results, including negative findings.
2. **Reproducibility over isolated metrics** — preserve configurations, artifacts, and evaluation procedures.
3. **Modularity over tightly coupled experiments** — separate data, models, services, and intelligence.
4. **Engineering rigor over inflated claims** — distinguish research prototypes from production-ready capabilities.

## Author

**Rajat Kumar Singh**

GitHub: [@rajat2004x](https://github.com/rajat2004x)

Repository: [AERIS-X](https://github.com/rajat2004x/Aeris-X)

---

*AERIS-X — Advancing aerospace health intelligence through reproducible machine learning research and modular engineering.*