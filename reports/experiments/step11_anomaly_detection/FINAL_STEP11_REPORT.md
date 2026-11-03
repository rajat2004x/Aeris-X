# Step 11 — System Anomaly Detection

## Scope and protocol

This package evaluates a deterministic robust statistical detector on NASA C-MAPSS
FD001. The detector is fit on the existing Step 6 `X_train` windows only. The first
half of `X_validation` is development data for threshold selection; the second half
is a frozen holdout. Controlled two-sensor BIAS/DRIFT scenarios are injected only
for final evaluation. This is system-level abnormal-behavior detection, not Step 10
sensor FDI or fault-type classification.

The detector aggregates the mean of the three largest per-sensor median/MAD
deviations and applies a 3-window persistence rule after a
99.0th-percentile development threshold.

## Results

| Metric | Value |
|---|---:|
| Threshold | 26.191149 |
| Precision | 0.600000 |
| Recall | 0.060000 |
| F1 | 0.109091 |
| ROC-AUC | 0.524433 |
| Normal holdout false-positive rate | 0.013333 |
| Event recall | 0.010000 |
| False alarms per engine | 0.133333 |

The low point recall and near-random ROC-AUC are retained as measured results; the
controlled injected scenarios do not establish production effectiveness. Event
recall is based on exact temporal overlap after persistence and should not be read
as evidence of broad anomaly-detection quality.

## Limitations

Only the processed FD001 contract is available, so cross-condition and cross-dataset
generalization to FD002–FD004 is not claimed. Normal degradation is measured on the
clean frozen holdout as a false-alarm analysis, not relabeled as abnormal. A
temporal autoencoder was not added because the available historical autoencoder
evidence documents poor practical detection and retraining would create a new
experiment outside this verification scope.

## Reproducibility and leakage

See `experiment_manifest.json`, `config.json`, and `leakage_audit.json`. No final
labels, injected anomalies, or future cycles are used for fitting or thresholding.
