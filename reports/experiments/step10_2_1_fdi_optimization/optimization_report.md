# Step 10.2.1 FDI Precision, Isolation, and Open-Set Optimization

## Decision

This is a new isolated optimization experiment. Step 10 remains unmarked and
the roadmap remains unchanged. Only C-MAPSS FD001 was available; FD002-FD004
and ESA telemetry were not used.

## Why the previous system failed

Step 10.2 used attribution from global detector scores as a proxy for sensor
isolation, trained fault classification on aggregate features, and had no
explicit unknown rejection. Its 21% FPR and 0% unknown recall therefore did
not establish reliable FDI.

## Final locked detection result

{
  "detector": "persistent_adaptive_statistical",
  "threshold": 27.673930872590585,
  "precision": 0.058823529411764705,
  "recall": 0.05,
  "f1": 0.05405405405405406,
  "fpr": 0.048743335872048744,
  "roc_auc": 0.8585491241431836,
  "pr_auc": 0.2381188118811881,
  "detection_rate": 0.05
}

## Comparison matrix

See `comparison_matrix.csv` for the Step 10 through Step 10.2.1 comparison.
The selected detector prioritizes F1, then PR-AUC, then low FPR; isolation
and unknown rejection are reported separately rather than hidden in a single
aggregate score.

Threshold selection used normal calibration and development known-fault data
only, with a persistence-aware score and an FPR objective of at most 10%.

## Dedicated sensor classifiers

{
  "random_forest": {
    "top1": 0.125,
    "top3": 0.2375,
    "top5": 0.4,
    "mrr": 0.27181225024975025,
    "mean_rank": 7.0875,
    "set_precision": 0.1,
    "set_recall": 0.1,
    "jaccard": 0.09166666666666666
  },
  "gradient_boosting": {
    "top1": 0.025,
    "top3": 0.1625,
    "top5": 0.425,
    "mrr": 0.21123699217449216,
    "mean_rank": 6.875,
    "set_precision": 0.03125,
    "set_recall": 0.03125,
    "jaccard": 0.029166666666666664
  }
}

The sensor classifiers consume one feature vector per sensor and are evaluated
on disjoint final scenarios. Detailed per-scenario ranks and sensor sets are
in `sensor_isolation.csv`.

## Fault classification and open-set rejection

{
  "known_macro_f1": 0.06666666666666667,
  "unknown_recall": 1.0,
  "unknown_precision": 0.5,
  "unknown_auroc": 0.4975,
  "unknown_aupr": 0.5064710910770024,
  "false_acceptance_rate": 0.0
}

Unknown classes were excluded from training: SPIKE, SATURATION, GAIN_ERROR, MISSING_BLOCK, BURST_NOISE. The
rejection rule is confidence-based and its threshold is derived from the
training-known confidence distribution. Results are reported even if weak.

## Limitations

This remains a synthetic-fault FD001 study. Cross-condition and cross-dataset
generalization cannot be measured without the unavailable datasets. The
classifier does not establish physical causality or operational aerospace
readiness. Historical Step 10, Step 10.1, and Step 10.2 artifacts were not
overwritten.
