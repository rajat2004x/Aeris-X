# Step 9: Uncertainty Quantification

## Objective

Evaluate whether the existing AERIS-X Transformer provides useful uncertainty
estimates using Monte Carlo Dropout.

## Method and integrity

The existing Transformer checkpoint was loaded without retraining. The
validation arrays from Step 6 were reused unchanged. The model was kept in
evaluation mode except for its dropout layers, which were selectively enabled
for `100` stochastic forward passes. No preprocessing, scaler,
split, architecture, or previous benchmark artifact was changed.

The interval is an **approximate 95% prediction interval based on MC Dropout
predictive variability**, calculated as mean +/- 1.96 * sample standard
deviation. It is not a formally guaranteed 95% confidence interval.

## Environment and reproducibility

- Checkpoint: `C:\Users\rajat\AERIS-X\reports\experiments\step8_temporal_dl\gpu\models\transformer_gpu.pt`
- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 3050 4GB Laptop GPU`
- PyTorch: `2.11.0+cu128`
- CUDA: `12.8`
- Seed: `42`
- MC samples: `100`

## Point prediction

- MAE: `17.321881`
- RMSE: `24.980096`
- PHM: `61844.862716`

## Uncertainty and coverage

- Mean predictive standard deviation: `2.982096`
- Median predictive standard deviation: `2.761896`
- Average interval width: `11.689815`
- Median interval width: `10.826634`
- Nominal coverage: `95.00%`
- Observed coverage: `34.91%`
- Coverage error: `-60.09%`
- Pearson uncertainty/error correlation: `-0.064166`
- Spearman uncertainty/error correlation: `0.032783`

## Uncertainty vs error

The highest-uncertainty group had MAE `22.937596`,
while the largest grouped MAE was `29.018569` in
`lowest_20`. This is descriptive evidence about the
relationship in this validation set, not
a claim of universal calibration or deployment reliability.

## RUL regimes

regime  sample_count       mae      rmse  mean_uncertainty  mean_interval_width  coverage
  0-20           300  2.520093  3.107402          2.414346             9.464234  0.860000
 20-40           300  5.938193  7.132626          2.540541             9.958921  0.483333
 40-60           300  6.993533  9.341173          2.869750            11.249419  0.523333
 60-80           300 13.583386 19.276585          3.697513            14.494252  0.430000
   80+          1427 25.784177 32.242178          3.067499            12.024595  0.159776

## Limitations and conclusion

MC Dropout is an approximate Bayesian-style method. The interval assumes an
approximately Gaussian predictive distribution, and empirical coverage may
differ substantially from 95%. C-MAPSS FD001 is a benchmark engine-degradation
dataset, not operational aerospace telemetry. This research implementation is
not a flight-certified safety system; further validation is required for real
aerospace deployment.

The results should be interpreted as a first uncertainty experiment. Future
work may compare additional validated uncertainty methods, but deep ensembles,
conformal prediction, quantile regression, Bayesian neural networks, anomaly
detection, FDI, and later milestones are outside Step 9.
