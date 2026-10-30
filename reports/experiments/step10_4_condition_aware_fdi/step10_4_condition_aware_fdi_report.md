# Step 10.4 Condition-Aware FDI

This isolated milestone quantifies operating-condition shift and compares global versus condition-aware robust baselines using normal training/calibration data only.

## Results

```
dataset                                 method  threshold  normal_windows  fault_windows  precision  recall       f1      fpr  roc_auc   pr_auc
  FD001       global_baseline_global_threshold  76.366280            3524             20   0.018701    0.95 0.036680 0.282917 0.860287 0.165489
  FD002       global_baseline_global_threshold  27.953403            8950             20   0.000000    0.00 0.000000 0.053631 0.695307 0.006784
  FD003       global_baseline_global_threshold  30.926357            4298             20   0.003891    0.10 0.007491 0.119125 0.623174 0.106122
  FD004       global_baseline_global_threshold  27.962921           10278             20   0.012472    0.55 0.024390 0.084744 0.919391 0.016704
  FD001 condition_baseline_condition_threshold  99.801409            3524             20   0.016129    0.15 0.029126 0.051930 0.751149 0.117154
  FD002 condition_baseline_condition_threshold  28.241813            8950             20   0.000000    0.00 0.000000 0.050950 0.718324 0.007477
  FD003 condition_baseline_condition_threshold  38.425870            4298             20   0.055556    0.10 0.071429 0.007911 0.536191 0.105714
  FD004 condition_baseline_condition_threshold  28.353991           10278             20   0.000000    0.00 0.000000 0.059739 0.576605 0.003397
```

## Not run

Sensor isolation, known-fault classification, open-set rejection, pooled training, self-supervised representation learning, uncertainty integration, and multi-sensor attribution were not run because their development protocols were not established before this controlled baseline comparison.

## Limitations

AERIS-X is a research/portfolio implementation using NASA C-MAPSS simulated turbofan telemetry and controlled synthetic sensor-fault injection. It is not evidence of flight certification, operational aerospace deployment, replacement for certified FDIR, or production aerospace safety certification.
