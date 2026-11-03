# Final Step 10: Sensor Health and FDI

## Executive summary
This isolated milestone artifact packages the defensible condition-aware FDI evidence already produced by the Step 10.4 final experiment. No model was retrained and no historical Step 10.x result was overwritten.
The integrated framework provides validation, condition-residual and temporal-peer evidence, thresholded detection, sensor ranking, fault-type evidence, open-set summaries, and calibration/uncertainty records.

## Research question
Can AERIS-X detect abnormal sensor behavior, distinguish it from normal degradation, identify likely sensors, and report confidence across operating conditions?

## Final integrated architecture
Telemetry -> validation -> condition residuals -> temporal evidence and peer residuals -> normal-only threshold -> FDI decision -> sensor ranking -> fault/open-set evidence -> evaluation.
The selected detector is the historical condition-residual + soft temporal evidence + peer-consistency ensemble. The final artifact is an evidence integration layer, not a new training run.

## Final integrated detection results

| Dataset | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| FD001 | 0.0510 | 0.4167 | 0.0909 | 0.0528 | 0.8579 | 0.2396 |
| FD002 | 0.0295 | 0.9167 | 0.0571 | 0.0810 | 0.9309 | 0.9069 |
| FD003 | 0.0282 | 0.2083 | 0.0498 | 0.0400 | 0.7955 | 0.2199 |
| FD004 | 0.0305 | 0.9167 | 0.0590 | 0.0681 | 0.9604 | 0.8684 |

## Sensor isolation

| Dataset | Top-1 | Top-3 | Top-5 | Top-10 | MRR |
|---|---:|---:|---:|---:|---:|
| FD001 | 0.5833 | 0.6667 | 0.7083 | 0.8333 | 0.6484 |
| FD002 | 0.0833 | 0.1667 | 0.4167 | 0.5417 | 0.2179 |
| FD003 | 0.5000 | 0.5417 | 0.6667 | 0.7500 | 0.5783 |
| FD004 | 0.0833 | 0.1667 | 0.4167 | 0.5000 | 0.2227 |

## Fault classification
Classification is evidence only; it is not presented as a validated universal classifier.

| Dataset | Known scenarios | Exact fault-type accuracy |
|---|---:|---:|
| FD001 | N/A | N/A |
| FD002 | N/A | N/A |
| FD003 | N/A | N/A |
| FD004 | N/A | N/A |

## Open-set and calibration evidence

```text
dataset  unknown_precision  unknown_recall  unknown_f1  unknown_auroc  unknown_aupr  false_acceptance_rate
  FD001                0.6             0.6    0.600000       0.835714      0.808357                    0.4
  FD002                1.0             1.0    0.625000       0.578571      0.467905                    0.0
  FD003                0.4             0.4    0.533333       0.778571      0.786395                    0.6
  FD004                1.0             1.0    0.625000       0.564286      0.466062                    0.0
```

```text
metric    value
 brier 0.003897
   ece 0.060478
```

## Historical findings and limitations
- The selected final artifact contains sensor-fault detection rows but no actual-versus-predicted fault-type fields; fault-type classification accuracy is therefore not claimed.
- FD001–FD004 results are not interchangeable with the strong historical FD001-only result from Step 10.2.2.
- The final integrated detector has very low precision and F1 on all four datasets despite useful ranking metrics in some datasets.
- Normal degradation and sensor anomalies remain difficult to separate; a changing telemetry channel is not itself proof of a sensor fault.
- Fault-type performance varies substantially, especially for drift, stuck-at, noise, and intermittent dropout.
- Open-set scores are controlled synthetic-fault evidence, not general unknown-fault detection.
- Calibration metrics and uncertainty records are reported as evidence, not proof of calibrated operational probabilities.
- These are research results on public telemetry and synthetic faults, not production, flight-certified, or safety-certified capability.

## Reproducibility
Use `scripts/run_step10_sensor_health_fdi.py` with the existing Step 10.4 final artifact directory. The manifest records source paths, seed, detector, fault types, and frozen-holdout status.

## Conclusion
Step 10 establishes a reproducible sensor-health/FDI evidence framework and documents the central limitation: strong in-domain or ranking metrics do not translate into reliable cross-dataset detection precision. The limitation is a result, not a defect to conceal.
