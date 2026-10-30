# Step 10.4.1 Calibrated Multi-Dataset FDI

Selected fusion model: **random_forest_isotonic**. Threshold: **0.540000**, selected on development data subject to FPR <= 10%.

## Final holdout fusion results

```
dataset                            method  threshold  precision   recall       f1      fpr  roc_auc   pr_auc  normal_windows  fault_windows
  FD001 calibrated_random_forest_isotonic       0.54   0.006783 1.000000 0.013476 0.997162 0.771661 0.017199            3524             24
  FD002 calibrated_random_forest_isotonic       0.54   0.004224 0.916667 0.008410 0.579441 0.916378 0.044317            8950             24
  FD003 calibrated_random_forest_isotonic       0.54   0.005623 1.000000 0.011184 0.987436 0.495269 0.005440            4298             24
  FD004 calibrated_random_forest_isotonic       0.54   0.004355 0.958333 0.008671 0.511578 0.960889 0.115796           10278             24
```

## Historical comparison

Historical Step 10.2.2 FD001 F1 82.24% / FPR 2.89% remains frozen. The new fusion model is a different multi-dataset protocol and does not replace that control.

## Findings and limitations

The supervised fusion and calibration procedures were trained only on development scenarios, but the selected isotonic model did not transfer its development threshold to the final cross-dataset holdout: final FPR remained high on FD001, FD003, and FD004. This is a measured calibration/domain-shift failure, not evidence of improvement. The fault classifier was trained on development scenarios. The pooled rows document repeated development-model fits over the shared feature protocol; they are not evidence of successful dataset-specific pooled generalization. This is not flight-certified FDIR.
