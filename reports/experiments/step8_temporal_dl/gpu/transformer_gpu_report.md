# Step 8.5 GPU Transformer Rerun

## GPU verification

- PyTorch: `2.11.0+cu128`
- CUDA runtime: `12.8`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 3050 4GB Laptop GPU`
- Selected device: `cuda`
- CUDA synchronization was used before and after training and inference timing.

## Configuration

The existing Step 8 `TransformerRegressor` and training configuration were reused unchanged:
`input_size=15`, `d_model=64`, `nhead=4`, `num_layers=2`, `dim_feedforward=128`,
`dropout=0.1`, `max_length=30`, seed `42`, batch size `128`, epochs `30`,
Adam learning rate `1e-3`, MSELoss, gradient clipping `1.0`, and checkpoint
patience `5`. The existing saved train/validation arrays were used unchanged.

Command:

```powershell
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe scripts\run_step8_5_gpu_transformer.py
```

## Metrics

| Metric | CPU reference | GPU rerun |
|---|---:|---:|
| MAE | 17.210884 | 17.793437 |
| RMSE | 24.590212 | 25.123281 |
| PHM score | 98955.540074 | 61942.234481 |
| Training time (seconds) | 380.929213 | 29.905384 |
| Inference time (seconds) | 0.199135 | 0.078060 |
| Parameter count | 68033 | 68033 |

## Comparison

- Training speedup: `12.738x`
- Inference speedup: `2.551x`
- MAE difference: `0.582553` (`3.385%`)
- RMSE difference: `0.533068` (`2.168%`)
- PHM difference: `-37013.305593` (`-37.404%`)

Metrics are expected to differ slightly because CPU and CUDA kernels can
produce different floating-point reduction orderings. No model or
hyperparameter changes were made to force metric agreement.
