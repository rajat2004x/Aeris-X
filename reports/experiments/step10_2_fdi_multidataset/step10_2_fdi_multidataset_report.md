# Step 10.2 Multi-Dataset FDI Upgrade

## Scope and data availability

This is an isolated upgrade of the historical Step 10.2 experiment. The
authoritative roadmap was not changed: Step 7 remains the current authorized
milestone. Only C-MAPSS FD001 was available locally. FD002-FD004 and the ESA
anomaly dataset were not used, so cross-condition and cross-dataset results
are unavailable rather than fabricated.

## Architecture

The pipeline is `quality-aware telemetry -> robust statistical, temporal,
cross-sensor, and autoencoder scores -> hybrid decision -> sensor isolation ->
fault classifier -> open-set analysis`. The autoencoder and classifier were
trained only on development/normal data. Final scenarios use distinct source
windows and scenario identities.

## Final locked evaluation

    detector  threshold  precision  recall       f1  balanced_accuracy       mcc      fpr   fnr  roc_auc   pr_auc  detection_rate
 statistical  72.484465   0.225352   1.000 0.367816           0.895278  0.422082 0.209444 0.000 0.895278 0.225352           1.000
 autoencoder   0.011388   0.107143   0.075 0.088235           0.518460  0.043729 0.038081 0.925 0.780760 0.161009           0.075
    temporal   7.116917   0.225352   1.000 0.367816           0.895278  0.422082 0.209444 0.000 0.895278 0.225352           1.000
cross_sensor   6.481810   0.000000   0.000 0.000000           0.465727 -0.064873 0.068545 1.000 0.927618 0.418690           0.000
      hybrid   3.000000   0.224719   1.000 0.366972           0.894897  0.421286 0.210206 0.000 0.945811 0.523691           1.000

## Fault classification and open set

{
  "known_fault_macro_f1": 0.13636363636363638,
  "unknown_fault_count": 40,
  "unknown_detection_recall": 0.0,
  "unknown_precision": 0.0,
  "known_unknown_confusion_matrix": [
    [
      40,
      0
    ],
    [
      40,
      0
    ]
  ]
}

Unknown types were absent from classifier training: SPIKE, SATURATION, GAIN_ERROR, MISSING_BLOCK, BURST_NOISE.
The classifier is not allowed to call an unseen label unknown without an
explicit rejection policy; therefore the reported unknown metrics measure
actual rejection behavior and may be zero.

## Isolation

The full per-scenario isolation table is in `isolation_results.csv`, including
Top-1, Top-3, Top-5, MRR, rank, and sensor identity. No single aggregate is
presented as proof of successful isolation.

## Leakage audit

See `leakage_audit.json`. Training uses FD001 X_train, calibration uses only
the first half of normal validation, and final normal/fault evaluation uses
the second half with disjoint scenario identities. Thresholds were selected
without final fault labels. Existing scalers and prepared arrays were not
refit.

## Cost and limitations

Device: `cuda`. Autoencoder training time: `18.38` seconds.
The detector remains a controlled synthetic-fault study. Natural degradation
is represented by held-out normal FD001 validation windows, but this is not
equivalent to operational fleet telemetry. Cross-condition, cross-dataset,
and real-fault conclusions remain unavailable.
