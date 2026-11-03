# Step 12 — Health-State Estimation

## Research question
What does AERIS-X know about the current system health from temporal telemetry?

## Health definition
Health is continuous normalized remaining useful life: `clip(RUL / 332.0, 0, 1)`.
This reuses the repository's RUL-derived health convention without inventing labels.
Health is primary; anomaly and sensor-fault outputs remain Step 11 and Step 10 concerns.

## Dataset and leakage controls
The evaluation uses the existing processed FD001 arrays (`(30, 15)` windows). Validation
engines were recovered from the existing raw FD001 file solely for ownership and split
metadata. Training uses 15104 windows; development uses engines [1, 11, 19, 23, 31, 34, 40, 45, 46];
the frozen final holdout uses engines [54, 71, 74, 81, 84, 91]. See `leakage_audit.json`.

## Models evaluated
Mean health, Linear Regression, the verified Step 7 XGBoost artifact, and the existing
Step 8 Transformer artifact. The final model was selected only by development MAE/RMSE.

## Final results

| Metric | Value |
|---|---:|
| Selected model | transformer |
| MAE | 0.055055 |
| RMSE | 0.079593 |
| R² | 0.817601 |
| Pearson correlation | 0.918930 |
| Spearman correlation | 0.930399 |
| Smoothness mean absolute delta | 0.014062 |
| Monotonicity violations | 479 |
| Direction consistency | 0.568306 |
| Early-life MAE | 0.104508 |
| Late-life MAE | 0.010781 |

## Health trajectories and error analysis
Trajectory plots are provided for representative final-holdout engines. The diagnostics
show whether the estimator tracks degradation, reacts too slowly/aggressively, or
predicts healthy systems as degraded. Continuous health is not forcibly monotonic.

## Uncertainty
No uncertainty is claimed. The previous uncertainty experiment had poor empirical
coverage, so this milestone reports point estimates only.

## Limitations and conclusion
FD001-only scope for this milestone; FD002–FD004 support is not claimed. Prepared
arrays do not contain original engine IDs, so ownership is recovered from the existing
raw FD001 file without changing features. The final result answers health only in the
normalized RUL sense and does not establish operational health certification.

**What AERIS-X knows:** the selected estimator provides a continuous estimate of
remaining-life-derived health for frozen FD001 engines, with the measured accuracy and
trajectory consistency above.

**Where it fails:** errors, monotonicity violations, and late/early-life MAE identify
where telemetry-to-health mapping is unreliable; no universal superiority is claimed.
