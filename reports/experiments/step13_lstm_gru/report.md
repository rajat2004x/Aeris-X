# Step 13 — LSTM / GRU Temporal Evaluation

## Objective

Determine whether compact recurrent models meaningfully improve FD001 RUL estimation over the accepted Transformer reference.

## Dataset

The existing Step 6 processed FD001 contract was reused: `X_train` and `X_validation` windows of shape `(30, 15)`. No preprocessing or historical artifact was changed.

## Experimental Protocol

Seed 42, Adam, learning rate 1e-3, batch size 128, MSE loss, 30-epoch maximum, validation-loss checkpointing, patience 5, and the same device selection utility were used. The compact recurrent models use hidden size 64, one layer, and dropout 0.2.

## Models

LSTM and GRU were trained in the isolated Step 13 namespace. CPU and GPU Transformer results and XGBoost are immutable references loaded from Step 7/8 artifacts.

## Results

See `model_comparison.csv` and `model_comparison.json`.

| Model | MAE | RMSE | PHM | Train seconds | Inference seconds | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | 19.9439 | 29.5300 | 144880.22 | 49.15 | 0.4507 | 20801 |
| GRU | 19.6595 | 28.8863 | 158344.10 | 45.42 | 0.1535 | 15617 |

## Lifecycle Analysis

See `lifecycle_error_comparison.csv` and `plots/lifecycle_mae.png`.

## Runtime Analysis

Training and inference timings are recorded per model. The runner is resumable and stores checkpoints and histories.

## Comparison With Transformer

The accepted CPU and GPU Transformer values are preserved as references. Percentage comparisons use `(reference - candidate) / reference * 100`; lower MAE/RMSE/PHM is better.

## Failure Analysis

This single FD001 validation protocol does not support statistical significance or broad generalization claims. Recurrent results must be interpreted with runtime and parameter costs.

## Leakage Audit

See `leakage_audit.json`. Validation data was used for checkpoint selection only; validation labels were not used for fitting.

## Reproducibility

See `reproducibility_manifest.json`.

## Limitations

The benchmark uses one processed FD001 validation split and one seed. It is not an untouched test estimate, certification, or production evaluation.

## Decision

`NO_CLEAR_WINNER`. The result does not justify replacing the Transformer based on FD001 alone.

## Recommendation

Retain the Transformer as the reference architecture. Keep LSTM/GRU as alternative compact architectures for future multi-seed or resource-constrained studies; do not claim universal superiority.
