# Step 10.2.2 Temporal Sensor Isolation and Adaptive FDI

## Executive summary

This isolated experiment tests causal temporal features, adaptive soft
evidence, dedicated sensor classifiers, and explicit open-set rejection.
Only C-MAPSS FD001 was available. No roadmap state or historical experiment
was changed.

## Final detection

{
  "threshold": 0.6113829673454165,
  "precision": 0.6984126984126984,
  "recall": 1.0,
  "f1": 0.822429906542056,
  "fpr": 0.028941355674028942,
  "fnr": 0.0,
  "roc_auc": 0.9843176625354843,
  "pr_auc": 0.7155209101040626,
  "detection_rate": 1.0,
  "detector": "soft_temporal_adaptive",
  "training_time_seconds": 2.7865936999733094
}

## Sensor isolation

[
  {
    "model": "random_forest",
    "top1": 0.10227272727272728,
    "top3": 0.18181818181818182,
    "top5": 0.3409090909090909,
    "mrr": 0.2458224855952129,
    "mean_rank": 7.5227272727272725
  },
  {
    "model": "gradient_boosting",
    "top1": 0.022727272727272728,
    "top3": 0.17045454545454544,
    "top5": 0.375,
    "mrr": 0.19230996276450824,
    "mean_rank": 7.6022727272727275
  }
]

## Fault classification and open set

{
  "known_macro_f1": 0.06666666666666667,
  "unknown_recall": 0.0,
  "unknown_precision": 0.0,
  "unknown_f1": 0.0,
  "unknown_auroc": 0.46328125000000003,
  "unknown_aupr": 0.5283721265847608,
  "false_acceptance_rate": 1.0
}

## Leakage controls

Training uses normal training telemetry and development known-fault scenarios.
Calibration uses normal calibration and development scenarios. Final scenarios
use disjoint windows, seeds, and unknown fault types. Final labels were not
used for threshold selection.

## Interpretation

The detector uses soft evidence rather than the hard persistence gate that
collapsed Step 10.2.1 recall. These metrics are a measured result, not a
success claim. Cross-dataset and real-fault validation remain unavailable.
