# Step 10: Sensor FDI and Anomaly Detection

## Objective

Investigate whether telemetry anomaly scores can detect controlled synthetic
sensor faults and isolate their affected sensor while distinguishing normal
C-MAPSS degradation from injected faults.

## Data and integrity

NASA C-MAPSS FD001 is the real benchmark dataset. The fault labels in this
experiment are simulated: controlled synthetic faults were injected into
existing normalized validation windows. The autoencoder was trained only on
normal training windows; injected faults were not used for training or
threshold fitting. Previous Step 6-9 data and artifacts were not modified.

## Fault matrix

135 scenarios: five fault types, three severity levels, three
onset positions, and three selected variable sensors. Fault types: BIAS, DRIFT, STUCK_AT, NOISE, INTERMITTENT_DROPOUT.
Severity scales are relative to each sensor's training standard deviation.

## Detector and threshold

An LSTM sequence autoencoder with hidden size 32 was trained for eight epochs
with Adam and MSE loss on normal training windows. The anomaly score is mean
reconstruction MSE. The threshold is the 95th percentile of normal validation
scores: `0.021401`. Normal-validation false-positive rate:
`5.02%`.

## Results

- Precision: `0.0000`
- Recall: `0.0000`
- F1: `0.0000`
- Accuracy: `0.9033`
- False-positive rate: `0.0502`
- False-negative rate: `1.0000`
- ROC-AUC: `0.4714`
- PR-AUC: `0.0419`
- Top-1 isolation accuracy: `0.0741`
- Top-3 isolation accuracy: `0.1926`
- Mean detection latency: `nan cycles`
- Median detection latency: `nan cycles`

## Limitations

Injected faults are controlled simulations, not real aerospace failure labels.
C-MAPSS FD001 is a benchmark engine-degradation dataset, not operational
telemetry. Anomaly detection is not automatically diagnosis; real FDI requires
validated failure data, sensor redundancy, physics, and operational context.
These results do not establish aerospace safety or flight certification.
