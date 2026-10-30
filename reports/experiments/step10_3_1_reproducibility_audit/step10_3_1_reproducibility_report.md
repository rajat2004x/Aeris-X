# Step 10.3.1 Reproducibility Audit

Historical Step 10.2.2 reproduction status: **REPRODUCED**.

The exact historical protocol reproduces the reported FD001 result. Step 10.3 did not use the exact same protocol: it rebuilt raw windows, used 21 sensors, refit per-dataset scalers, changed the split, threshold procedure, and scenario matrix.

Therefore the Step 10.3 FD001 discrepancy is a **DIFFERENT PROTOCOL**, not evidence that the historical detector itself is irreproducible. The corrected benchmark uses the historical 15-sensor contract, saved FD001 scaler, fixed historical threshold, and historical scenario generator.

## Corrected frozen benchmark

```
dataset                               protocol     status  threshold  precision  recall       f1      fpr  fnr  roc_auc   pr_auc  detection_rate  normal_windows  fault_windows  scenario_count  train_engine_count
  FD001           exact historical Step 10.2.2 REPRODUCED   0.611383   0.698413     1.0 0.822430 0.028941  0.0 0.984318 0.715521             1.0            1313             88             NaN                 NaN
  FD002 corrected historical detector contract   VERIFIED   0.611383   0.009737     1.0 0.019286 1.000000  0.0 0.962838 0.117953             1.0            8950             88            88.0               156.0
  FD003 corrected historical detector contract   VERIFIED   0.611383   0.048114     1.0 0.091810 0.405072  0.0 0.988012 0.589796             1.0            4298             88            88.0                60.0
  FD004 corrected historical detector contract   VERIFIED   0.611383   0.008489     1.0 0.016836 1.000000  0.0 0.375191 0.007766             1.0           10278             88            88.0               149.0
```

See `root_cause_matrix.csv`, `data_reproduction_audit.json`, and `protocol_diff.json` for the controlled comparison. The original Step 10.3 artifacts remain preserved.
