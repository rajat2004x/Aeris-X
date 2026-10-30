# Step 10.3 Multi-Dataset C-MAPSS FDI Generalization

The detector was calibrated on FD001 normal calibration engines and applied unchanged to FD001–FD004 final engines.

## Frozen results

```
dataset  threshold  normal_windows  fault_windows  precision  recall       f1  accuracy      fpr  fnr  roc_auc   pr_auc  detection_rate  false_alarm_rate
  FD001   0.648002            3524            100   0.000000    0.00 0.000000  0.926049 0.047673 1.00 0.873510 0.106665            0.00          0.047673
  FD002   0.648002            8950            100   0.013330    1.00 0.026309  0.182099 0.827039 0.00 0.541522 0.036911            1.00          0.827039
  FD003   0.648002            4298            100   0.000000    0.00 0.000000  0.903365 0.075617 1.00 0.835033 0.101359            0.00          0.075617
  FD004   0.648002           10278            100   0.009936    0.89 0.019653  0.144440 0.862814 0.11 0.430641 0.029863            0.89          0.862814
```

## Retraining

Dataset-specific and pooled retraining were not run; no training improvement is claimed. Sensor isolation, open-set, and unseen-condition sub-experiments are explicitly marked NOT RUN.

## Scientific conclusion

The FD001-calibrated frozen detector did not generalize robustly. FD002 and FD004 show severe false-alarm degradation, while FD001 and FD003 retain low false-alarm rates but missed the injected faults under this cross-dataset protocol. These controlled results do not support a claim of cross-dataset FDI generalization.

## Limitations

These experiments use the NASA C-MAPSS simulated turbofan dataset and controlled synthetic sensor-fault injection. They are not evidence of flight-certified aerospace FDI performance. Sensor isolation is marked NOT RUN in this frozen generalization pass.
