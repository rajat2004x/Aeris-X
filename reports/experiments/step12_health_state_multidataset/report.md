# Step 12 Multi-Dataset Health-State Evaluation

## Objective
Evaluate the accepted Step 12 Transformer health-state methodology across FD001–FD004 without changing the accepted FD001 baseline.

## Existing Step 12 control
The immutable FD001 control is MAE 0.0550547, RMSE 0.0795935, R² 0.8176014, Pearson 0.9189300, Spearman 0.9303994, smoothness 0.0140622, 479 monotonicity violations, direction consistency 0.5683060, early-life MAE 0.1045079, and late-life MAE 0.0107813.

## Dataset characterization
See `dataset_characterization.csv`. Operating conditions are counted after deterministic rounding of settings; fault-mode labels are unavailable in the telemetry contract and are not fabricated.

## Protocols
In-domain models use deterministic engine-level train/development/final splits and train-only scalers. Frozen transfer uses the accepted FD001 Transformer, FD001 source preprocessing compatibility map, and FD001 training RUL reference; target datasets are evaluation-only.

## In-domain results

dataset      mae     rmse       r2  correlation  spearman_correlation  monotonicity_violations  degradation_direction_consistency
  FD001 0.055055 0.079593 0.817601     0.918930              0.930399                      479                           0.568306
  FD002 0.103937 0.130812 0.492315     0.741519              0.779001                     3917                           0.567996
  FD003 0.085200 0.120298 0.597906     0.799220              0.873098                     2069                           0.522923
  FD004 0.076649 0.102350 0.606420     0.779809              0.813704                     4692                           0.548214

## Frozen transfer results

source target      mae     rmse        r2  correlation  spearman_correlation
 FD001  FD001 0.055055 0.079593  0.817601     0.918930              0.930399
 FD001  FD002 0.197176 0.249786 -1.030454    -0.079593             -0.163464
 FD001  FD003 0.123879 0.183339  0.501745     0.738173              0.797823
 FD001  FD004 0.255811 0.328340 -1.130208    -0.058605             -0.155918

## Operating-condition analysis
Per-condition results are in `operating_condition_results.csv` where supported. No fault-mode classification is claimed.

## Leakage audit and limitations
See `leakage_audit.json`. FD001–FD004 are benchmark datasets, health is RUL-derived, and no operational or flight-certification claim is made. The source compatibility map is required because the accepted FD001 preprocessing scaler was not serialized; this is documented rather than silently replaced.

## Conclusion
The measured transfer rows determine whether generalization is strong, moderate, weak, or severe. This study does not tune the transfer model and does not advance the roadmap.
