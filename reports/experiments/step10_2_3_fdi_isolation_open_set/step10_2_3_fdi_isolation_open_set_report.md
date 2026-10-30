# Step 10.2.3 Sensor Isolation and Open-Set FDI Fix

## Detection reference

The Step 10.2.2 soft temporal adaptive detector was frozen and re-executed
without changing its feature path or threshold-selection protocol.

{
  "detector": "frozen_soft_temporal_adaptive",
  "threshold": 0.6113829673454165,
  "precision": 0.6984126984126984,
  "recall": 1.0,
  "f1": 0.822429906542056,
  "fpr": 0.028941355674028942,
  "roc_auc": 0.9843176625354843,
  "pr_auc": 0.7155209101040626,
  "training_time_seconds": 1.4984777999925427
}

## Diagnosis

Step 10.2.2 ranked sensors from a generic per-sensor classifier without
training-only peer residuals or fault-type conditioning. Composite scenarios
also exposed a single-target-label limitation. Step 10.2.2 open-set scoring
used confidence alone, which accepted unknown morphologies.

## New isolation path

The new path preserves the sensor axis, adds peer prediction residuals fit
only on X_train, uses balanced pairwise faulty-vs-healthy sensor training,
and reports complete sensor rankings. Results are in `isolation_results.csv`
and `sensor_breakdown.csv`.

                 model     top1     top3     top5    top10      mrr  mean_rank
pairwise_random_forest 0.090909 0.238636 0.397727 0.852273 0.258389   6.534091

## Known-fault and open-set results

{
  "threshold": 0.3103960145870998,
  "unknown_precision": 0.5454545454545454,
  "unknown_recall": 1.0,
  "unknown_f1": 0.7058823529411765,
  "unknown_auroc": 0.6567708333333333,
  "unknown_aupr": 0.7473709591508177,
  "false_acceptance_rate": 0.0,
  "known_macro_f1": 0.06666666666666667
}

Unknown types were excluded from training: SPIKE, SATURATION, GAIN_ERROR, DEADBAND, BURST_NOISE, COMPOSITE.

## Historical comparison

See `historical_comparison.csv` for Steps 10 through 10.2.3. The detector
metrics must match the Step 10.2.2 reference; localization and open-set
components are evaluated separately.

## Limitations

Only FD001 was available. Unknown AUROC improved above random, while sensor
isolation regressed slightly versus Step 10.2.2 (pairwise RF Top-1 9.09%
versus 10.23%). Unknown AUROC and sensor isolation may remain weak;
no claim of aerospace operational readiness or certification is made. The
roadmap remains unchanged and Step 10 is not marked complete.
