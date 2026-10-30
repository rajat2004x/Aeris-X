"""Run the existing Step 8 Transformer benchmark on CUDA only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aeris.evaluation.benchmark import load_ml_ready_dataset
from aeris.evaluation.metrics import mean_absolute_error, phm_score, root_mean_squared_error
from aeris.models.temporal import TransformerRegressor
from aeris.training.datasets import TemporalDataset
from aeris.training.trainer import Trainer, seed_everything


CPU_REFERENCE = {
    "mae": 17.21088359943124,
    "rmse": 24.590212497395246,
    "phm_score": 98955.54007371175,
    "training_time_seconds": 380.92921259999275,
    "inference_time_seconds": 0.19913540000561625,
    "parameter_count": 68033,
}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/processed/cmapss_fd001")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/experiments/step8_temporal_dl/gpu",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Step 8.5 requires CUDA, but CUDA is unavailable.")

    seed_everything(42)
    device = torch.device("cuda")
    output = args.output_dir
    models_dir = output / "models"
    output.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU name: {torch.cuda.get_device_name(device)}")
    print(f"Selected device: {device}")

    X_train, y_train, X_validation, y_validation = load_ml_ready_dataset(args.data_dir)
    train_loader = DataLoader(
        TemporalDataset(X_train, y_train),
        batch_size=128,
        shuffle=True,
    )
    validation_loader = DataLoader(
        TemporalDataset(X_validation, y_validation),
        batch_size=128,
        shuffle=False,
    )

    model = TransformerRegressor(
        input_size=15,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_length=30,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        loss_fn=nn.MSELoss(),
        device=device,
        checkpoint_path=models_dir / "transformer_gpu.pt",
        patience=5,
    )

    synchronize(device)
    training_started = time.perf_counter()
    history = trainer.fit(train_loader, validation_loader, epochs=30)
    synchronize(device)
    training_seconds = time.perf_counter() - training_started

    synchronize(device)
    inference_started = time.perf_counter()
    predictions, _ = trainer.predict(validation_loader)
    synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    predictions = np.asarray(predictions, dtype=float)
    np.savetxt(
        output / "transformer_gpu_predictions.csv",
        np.column_stack((y_validation, predictions)),
        delimiter=",",
        header="actual_rul,predicted_rul",
        comments="",
    )

    gpu_metrics = {
        "model": "Transformer",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "parameter_count": parameter_count,
        "training_time_seconds": training_seconds,
        "inference_time_seconds": inference_seconds,
        "inference_time_per_sample": inference_seconds / len(y_validation),
        "mae": mean_absolute_error(y_validation, predictions),
        "rmse": root_mean_squared_error(y_validation, predictions),
        "phm_score": phm_score(y_validation, predictions),
        "seed": 42,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "epochs": 30,
        "optimizer": "Adam",
        "loss": "MSELoss",
        "window_size": 30,
        "num_features": 15,
        "checkpoint": str((models_dir / "transformer_gpu.pt").resolve()),
        "command": "$env:PYTHONPATH=\"src\"; .\\.venv\\Scripts\\python.exe scripts\\run_step8_5_gpu_transformer.py",
    }
    comparison = {
        "training_speedup": CPU_REFERENCE["training_time_seconds"] / training_seconds,
        "inference_speedup": CPU_REFERENCE["inference_time_seconds"] / inference_seconds,
        "mae_absolute_difference": gpu_metrics["mae"] - CPU_REFERENCE["mae"],
        "mae_percentage_difference": (gpu_metrics["mae"] - CPU_REFERENCE["mae"]) / CPU_REFERENCE["mae"] * 100,
        "rmse_absolute_difference": gpu_metrics["rmse"] - CPU_REFERENCE["rmse"],
        "rmse_percentage_difference": (gpu_metrics["rmse"] - CPU_REFERENCE["rmse"]) / CPU_REFERENCE["rmse"] * 100,
        "phm_absolute_difference": gpu_metrics["phm_score"] - CPU_REFERENCE["phm_score"],
        "phm_percentage_difference": (gpu_metrics["phm_score"] - CPU_REFERENCE["phm_score"]) / CPU_REFERENCE["phm_score"] * 100,
    }
    result = {"gpu": gpu_metrics, "cpu_reference": CPU_REFERENCE, "comparison": comparison}
    (output / "transformer_gpu_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(8, 5))
    labels = ["MAE", "RMSE", "PHM / 10000"]
    cpu_values = [CPU_REFERENCE["mae"], CPU_REFERENCE["rmse"], CPU_REFERENCE["phm_score"] / 10000]
    gpu_values = [gpu_metrics["mae"], gpu_metrics["rmse"], gpu_metrics["phm_score"] / 10000]
    x = np.arange(len(labels))
    axis.bar(x - 0.18, cpu_values, 0.36, label="CPU reference")
    axis.bar(x + 0.18, gpu_values, 0.36, label="GPU rerun")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Metric value")
    axis.set_title("Step 8.5 Transformer: CPU vs GPU")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "transformer_gpu_metrics.png", dpi=150)
    plt.close(figure)

    report = f"""# Step 8.5 GPU Transformer Rerun

## GPU verification

- PyTorch: `{gpu_metrics["pytorch_version"]}`
- CUDA runtime: `{gpu_metrics["cuda_version"]}`
- CUDA available: `{torch.cuda.is_available()}`
- GPU: `{gpu_metrics["gpu_name"]}`
- Selected device: `{gpu_metrics["device"]}`
- CUDA synchronization was used before and after training and inference timing.

## Configuration

The existing Step 8 `TransformerRegressor` and training configuration were reused unchanged:
`input_size=15`, `d_model=64`, `nhead=4`, `num_layers=2`, `dim_feedforward=128`,
`dropout=0.1`, `max_length=30`, seed `42`, batch size `128`, epochs `30`,
Adam learning rate `1e-3`, MSELoss, gradient clipping `1.0`, and checkpoint
patience `5`. The existing saved train/validation arrays were used unchanged.

Command:

```powershell
{gpu_metrics["command"]}
```

## Metrics

| Metric | CPU reference | GPU rerun |
|---|---:|---:|
| MAE | {CPU_REFERENCE["mae"]:.6f} | {gpu_metrics["mae"]:.6f} |
| RMSE | {CPU_REFERENCE["rmse"]:.6f} | {gpu_metrics["rmse"]:.6f} |
| PHM score | {CPU_REFERENCE["phm_score"]:.6f} | {gpu_metrics["phm_score"]:.6f} |
| Training time (seconds) | {CPU_REFERENCE["training_time_seconds"]:.6f} | {training_seconds:.6f} |
| Inference time (seconds) | {CPU_REFERENCE["inference_time_seconds"]:.6f} | {inference_seconds:.6f} |
| Parameter count | {CPU_REFERENCE["parameter_count"]} | {parameter_count} |

## Comparison

- Training speedup: `{comparison["training_speedup"]:.3f}x`
- Inference speedup: `{comparison["inference_speedup"]:.3f}x`
- MAE difference: `{comparison["mae_absolute_difference"]:.6f}` (`{comparison["mae_percentage_difference"]:.3f}%`)
- RMSE difference: `{comparison["rmse_absolute_difference"]:.6f}` (`{comparison["rmse_percentage_difference"]:.3f}%`)
- PHM difference: `{comparison["phm_absolute_difference"]:.6f}` (`{comparison["phm_percentage_difference"]:.3f}%`)

Metrics are expected to differ slightly because CPU and CUDA kernels can
produce different floating-point reduction orderings. No model or
hyperparameter changes were made to force metric agreement.
"""
    (output / "transformer_gpu_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Outputs written to {output}")


if __name__ == "__main__":
    main()
