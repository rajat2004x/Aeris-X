# Step 10.4 Final Condition-Aware FDI

The final isolated experiment uses condition residuals, soft temporal evidence, peer-consistency residuals, adaptive normal-only thresholds, sensor-local ranking, fault-type/open-set summaries, and multi-sensor scenarios.

## Final results

```
dataset                                    method  threshold  precision   recall       f1      fpr  roc_auc   pr_auc  normal_windows  fault_windows
  FD001 condition_residual_temporal_peer_ensemble   0.513017   0.051020 0.416667 0.090909 0.052781 0.857862 0.239612            3524             24
  FD002 condition_residual_temporal_peer_ensemble   1.044068   0.029451 0.916667 0.057069 0.081006 0.930903 0.906873            8950             24
  FD003 condition_residual_temporal_peer_ensemble   0.956733   0.028249 0.208333 0.049751 0.040019 0.795501 0.219900            4298             24
  FD004 condition_residual_temporal_peer_ensemble   1.048903   0.030471 0.916667 0.058981 0.068107 0.960441 0.868396           10278             24
```

## Sensor isolation

```
             top1      top3      top5     top10       mrr
dataset                                                  
FD001    0.583333  0.666667  0.708333  0.833333  0.648394
FD002    0.083333  0.166667  0.416667  0.541667  0.217914
FD003    0.500000  0.541667  0.666667  0.750000  0.578271
FD004    0.083333  0.166667  0.416667  0.500000  0.222750
```

## Limitations

Results use NASA C-MAPSS simulated turbofan telemetry and controlled synthetic sensor faults. They are not evidence of flight certification, operational aerospace deployment, replacement for certified FDIR, or production aerospace safety certification.
