# Step 12 Improvement Experiment

## Existing baseline
The immutable control is the existing Step 12 Transformer. Its final metrics are
preserved in `baseline_results.csv` and the original Step 12 directory.

## Weaknesses and hypotheses
The baseline had 479 monotonicity violations, 0.568306 direction consistency, and
early-life MAE 0.104508 versus late-life MAE 0.010781. Candidate A tests lifecycle
balanced Huber loss. Candidate B tests a 0.10 temporal consistency penalty only for
same-engine adjacent windows whose true health is non-increasing. Neither forces all
local predictions to decrease.

## Experiments and selection
Only the two predeclared candidates were trained. Selection used development engines
[1, 11, 19, 23, 31, 34, 40, 45, 46] and the stated MAE/RMSE/model-name rule. Final engines [54, 71, 74, 81, 84, 91]
were frozen until after selection.

## Results

               model split      mae     rmse        r2  correlation  spearman_correlation  smoothness_mean_abs_delta  monotonicity_violations  monotonicity_violation_rate  degradation_direction_consistency  early_life_mae  mid_life_mae  late_life_mae  engines
baseline_transformer final 0.055055 0.079593  0.817601     0.918930              0.930399                   0.014062                      479                     0.436248                           0.568306        0.104508      0.082982       0.010781        6
     lifecycle_huber final 0.158764 0.199314 -0.143777     0.659347              0.684026                   0.042011                      532                     0.484517                           0.520036        0.289735      0.204088       0.100681        6

## Monotonicity and lifecycle analysis
Candidate metrics include violations, direction consistency, smoothness, and
early/mid/late-life MAE. A lower error candidate is not accepted if trajectory
consistency materially worsens.

## Robustness

               model                 corruption  clean_mae  corrupted_mae  mae_degradation  prediction_stability healthy_as_degraded_rate  healthy_windows  affected_windows
baseline_transformer sensor_0_bias_severity_1.5   0.055055       0.067100         0.012045              0.017234                     None                0               275
  selected_candidate sensor_0_bias_severity_1.5   0.158764       0.159747         0.000982              0.001861                     None                0               275

The corruption experiment is evaluation-only and does not establish FDI or anomaly
 detection capability. The frozen final subset contains no windows at the fixed
 healthy threshold (0.75), so healthy-as-degraded rate is not applicable there.

## Uncertainty
Not operationally calibrated. No new uncertainty claim is made.

## Leakage and reproducibility
See `leakage_audit.json`, `config.json`, and `experiment_manifest.json`. Training,
development, and frozen-final engine boundaries are preserved. Seed: 42.

## Final decision
**No candidate materially improved the existing Step 12 baseline; baseline retained.**

Baseline Step 12 artifacts were not overwritten. Step 13 was not implemented and the
roadmap was not advanced.
