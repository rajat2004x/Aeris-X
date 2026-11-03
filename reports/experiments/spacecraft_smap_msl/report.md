# AERIS-X SMAP/MSL Spacecraft Anomaly Detection

## 1. Objective

Evaluate leakage-controlled classical anomaly detectors on the acquired Telemanom SMAP/MSL package.

## 2. Dataset

82 train files, 82 test files, 55 SMAP channels, 27 MSL channels, and 82 label records. The CSV contains 81 unique IDs because `P-2` is duplicated; it was retained.

## 3. Dataset Audit

The audit found no missing or infinite values. Train/test IDs match, label IDs are present, and anomaly sequences parse successfully.

## 4. Leakage-Controlled Protocol

Each channel uses train-only mean/std/norm summary preprocessing. The first 80% of train telemetry fits detectors and the remaining 20% calibrates a frozen 99th-percentile threshold. Test labels are used only for final evaluation.

## 5. Detectors

Robust median/MAD, rolling deviation, PCA reconstruction, and Isolation Forest were evaluated. The autoencoder was explicitly skipped to maintain the laptop runtime bound.

## 6. Point-Level Results

See `detector_comparison.csv` and `baseline_results.json` for precision, recall, F1, accuracy, FPR, FNR, ROC-AUC, PR-AUC, and coverage.

## 7. Event-Level Results

Events are the provided anomaly ranges; predicted points are grouped with a two-sample temporal gap rule fixed before evaluation. See `event_metrics.json`.

## 8. Channel Isolation

Scores are ranked across channels at anomaly midpoints. See `channel_isolation.json`; this is channel isolation, not fault classification.

## 9. SMAP vs MSL

Per-spacecraft rows are available in `detector_comparison.csv` and `event_metrics.json`. No significance claim is made.

## 10. Robustness

Deterministic evaluation-only bias, drift, spike, stuck-at, noise, and dropout perturbations are in `robustness_results.json`; raw data was not modified.

## 11. Cross-Domain Comparison with C-MAPSS

See `cross_domain_comparison.json` and `cross_domain_comparison.md`. Step 11 artifacts were not modified.

## 12. Explainability

Per-channel scores, thresholds, and event ranges provide reproducible evidence. No unsupported causal explanation is claimed.

## 13. Failure Analysis

Aggregate false-positive and false-negative counts are in `failure_analysis.json`. Morphology-specific classification is limited by the source labels.

## 14. Best Research Reference Detector

Selection rule was fixed as PR-AUC, then event recall, F1, normal FPR, and detection delay. The selected reference is **robust**. It is not certified for operations.

## 15. Limitations

This is a benchmark analysis, not NASA validation, flight certification, or operational deployment. The summary representation compresses native feature vectors, and no direct cross-domain transfer was tested.

## 16. Reproducibility

Seed: 42. Runtime and environment details are in `reproducibility_manifest.json`.

## 17. Conclusion

The pipeline provides bounded, leakage-controlled spacecraft telemetry evidence, but detector quality and channel isolation must be interpreted from the complete metrics rather than accuracy alone.
