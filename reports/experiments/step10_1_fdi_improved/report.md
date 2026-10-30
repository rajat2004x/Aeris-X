# Step 10.1: Improved Sensor FDI and Anomaly Detection

## Baseline diagnosis

The original Step 10 used a global mean reconstruction score and mean
per-sensor contribution. Short tail faults were diluted across the window,
and sensor-specific evidence was lost. The original baseline remains preserved
at `reports/experiments/step10_fdi/`.

## Improvements

This experiment uses robust per-sensor median/MAD scores with maximum temporal
deviation, an improved 64-unit AdamW autoencoder trained for 20 epochs, and
sensor rankings from maximum per-sensor contributions. Thresholds were fixed
at the 95th percentile using normal calibration windows only. Scenarios were
split deterministically 70/30 before final evaluation; final labels were not
used for tuning.

## Final held-out comparison

 precision   recall       f1  accuracy      fpr      fnr  roc_auc   pr_auc  detection_rate    detector  threshold     top1     top3     top5
  0.080000 0.585366 0.140762  0.783764 0.210046 0.414634 0.709517 0.065858        0.585366 statistical  72.484465 0.073171 0.439024 0.512195
  0.000000 0.000000 0.000000  0.925461 0.045662 1.000000 0.535657 0.034202        0.000000 autoencoder   0.003987 0.048780 0.170732 0.292683
  0.137931 0.097561 0.114286  0.954244 0.019026 0.902439 0.634722 0.085356        0.097561    combined   1.768324 0.048780 0.170732 0.292683

### Fault-type breakdown

          fault_type  sample_count  detection_rate  mean_statistical_score
                BIAS             9        0.666667               49.219880
               DRIFT            10        0.800000               58.527164
INTERMITTENT_DROPOUT             8        0.000000                2.288940
               NOISE             4        0.000000                2.649681
            STUCK_AT            10        1.000000               72.484465

### Severity breakdown

severity_name  sample_count  detection_rate  mean_statistical_score
         high            14        0.357143               27.437459
          low            14        0.642857               47.507485
       medium            13        0.769231               56.367402

### Onset breakdown

onset_name  sample_count  detection_rate  mean_statistical_score
     early            10        0.700000               51.496658
      late            14        0.642857               47.452933
    middle            17        0.470588               35.452812

### Sensor breakdown

 sensor_index  sample_count  detection_rate  mean_statistical_score
            0            12        0.666667               49.160197
            1            13        0.538462               40.232650
            2            16        0.562500               41.816164

Normal-degradation false-positive rate on held-out normal validation windows:
`21.00%`. This is a
distribution-shift diagnostic, not a tuned final-fault threshold.

Training device: `cuda`. Autoencoder training time: `24.31` seconds.
The statistical detector is the primary improvement because it directly
captures distributional sensor excursions without reconstruction dilution.

## Leakage audit

- Existing scaler and prepared arrays were reused unchanged.
- The autoencoder trained only on normal `X_train`.
- Thresholds used only normal calibration windows.
- Fault severity used training-window standard deviations.
- Final fault scenarios were held out before evaluation.
- No final evaluation labels selected thresholds, architecture, or weights.

## Limitations

Faults are controlled synthetic simulations, not real aerospace fault labels.
C-MAPSS FD001 is a benchmark engine-degradation dataset. Detection is not
diagnosis, and synthetic isolation does not establish operational FDI
capability or aerospace safety.
