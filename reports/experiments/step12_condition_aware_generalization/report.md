# Step 12 Condition-Aware Generalization Study

## 1. Objective
Test whether simple training-only operating-condition normalization and broader training-domain coverage improve the accepted Step 12 estimator under C-MAPSS domain shift.

## 2. Research Question
Can condition-aware preprocessing and multi-condition training improve robustness without target-domain leakage?

## 3. Existing Control
The immutable control is read from `step12_health_state_multidataset`. No control artifact was rerun or overwritten.

## 4. Hypotheses
H1–H5 were registered in `hypothesis.json` and evaluated without assuming positive results.

## 5. Dataset / Operating-Condition Analysis
Conditions use rounded `setting_1` and `setting_3` (`setting_1|setting_3`). FD001/FD003 have one condition; FD002/FD004 have six. Fault-mode labels are unavailable.

## 6. Experimental Protocol
The same Transformer, window size, selected features, optimizer, seed, batch size, and eight epochs were used. Condition means/scales were fitted only on training-engine rows. Unseen transfer conditions use source-global statistics.

## 7. Experiment A — In-Domain Condition-Aware Preprocessing
| Dataset | Baseline MAE | Condition-Aware MAE | Baseline R² | Candidate R² |
|---|---:|---:|---:|---:|
| FD001 | 0.055055 | 0.071706 | 0.817601 | 0.711130 |
| FD002 | 0.103937 | 0.238359 | 0.492315 | -1.143114 |
| FD003 | 0.085200 | 0.085060 | 0.597906 | 0.593855 |
| FD004 | 0.076649 | 0.157277 | 0.606420 | -0.657650 |

## 8. Experiment B — Frozen Cross-Dataset Transfer
| Source | Target | Baseline MAE | Candidate MAE | Baseline R² | Candidate R² |
|---|---|---:|---:|---:|---:|
| FD001 | FD002 | 0.197176 | 0.148976 | -1.030454 | -0.118578 |
| FD001 | FD003 | 0.123879 | 0.141587 | 0.501745 | 0.398407 |
| FD001 | FD004 | 0.255811 | 0.192596 | -1.130208 | -0.198382 |

## 9. Experiment C — Multi-Dataset Training
| Train | Held-Out | MAE | RMSE | R² | Spearman |
|---|---|---:|---:|---:|---:|
| FD001 + FD002 + FD003 | FD004 | 0.114160 | 0.146382 | 0.177295 | 0.446354 |

## 10. Operating-Condition Error Analysis
Detailed condition counts and fallback-window counts are stored in the manifests and transfer results. No target-final statistics were used.

## 11. Lifecycle Error Analysis
All result rows include early-, mid-, and late-life MAE plus trajectory consistency metrics.

## 12. Robustness
Sensor perturbation was not run because no existing reusable perturbation protocol was identified during the controlled study; this remains a limitation.

## 13. Leakage Audit
See `leakage_audit.json`; all checks are PASS.

## 14. Reproducibility
See `reproducibility_manifest.json`.

## 15. Limitations
C-MAPSS is a synthetic benchmark; health is RUL-derived rather than an independent physical health ground truth. These results make no operational, production, or flight-certification claim.

## 16. Results
Condition-aware preprocessing improved frozen FD001→FD002 and FD001→FD004 MAE/RMSE/R²/Spearman, but it worsened FD003 transfer and materially regressed in-domain FD001, FD002, and FD004. It is therefore not adopted as the new default.

The prioritized C1 multi-dataset run (FD001+FD002+FD003 held out on FD004) achieved MAE 0.114160, RMSE 0.146382, R² 0.177295, and Spearman 0.446354. C2 and C3 were not run, so no complete leave-one-dataset-out claim is made.

H1 — SUPPORTED: the FD001 transfer failure is consistent with condition/domain shift, and condition normalization materially changes FD002/FD004 transfer.

H2 — PARTIALLY SUPPORTED: normalization changes transfer errors, but the in-domain regressions prevent claiming a robust improvement.

H3 — NOT SUPPORTED by the available evidence: the condition-aware in-domain candidate was not uniformly better.

H4 — PARTIALLY SUPPORTED: C1 improves over frozen FD001→FD004 transfer, but it is worse than FD004 in-domain training and the other held-out splits were not run.

H5 — SUPPORTED: the leakage audit passes and all condition statistics and health references use permitted training data.

## 17. Decision
Decision: neither condition-aware preprocessing nor the partial C1 result is sufficient for adoption as a general default. Stop this preprocessing-only optimization line and proceed to architecture/domain-generalization research when authorized. The roadmap remains unchanged.
