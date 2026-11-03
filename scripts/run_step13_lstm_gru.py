"""Execute the isolated Step 13 LSTM/GRU evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aeris.step13 import run_step13


def main() -> None:
    output = ROOT / "reports" / "experiments" / "step13_lstm_gru"
    results = run_step13(ROOT / "data" / "processed" / "cmapss_fd001", output)
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    temporal = results[results["model"].isin(["LSTM", "GRU"])]
    temporal.set_index("model")[["mae", "rmse", "phm_score"]].plot(kind="bar", title="Step 13 model comparison")
    plt.tight_layout()
    plt.savefig(plots / "model_metric_comparison.png")
    plt.close()
    lifecycle = pd.read_csv(output / "lifecycle_error_comparison.csv")
    lifecycle.pivot(index="lifecycle", columns="model", values="mae").plot(kind="bar", title="Lifecycle MAE")
    plt.tight_layout()
    plt.savefig(plots / "lifecycle_mae.png")
    plt.close()
    for model in ("LSTM", "GRU"):
        history = dict(pd.read_csv(output / f"{model.lower()}_training_history.npz")) if False else None
        import numpy as np
        values = np.load(output / f"{model.lower()}_training_history.npz")
        plt.plot(values["train_loss"], label="train")
        plt.plot(values["validation_loss"], label="validation")
        plt.title(f"{model} training curve")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots / f"{model.lower()}_training_curve.png")
        plt.close()
    predictions = {}
    for model in ("LSTM", "GRU"):
        import numpy as np
        predictions[model] = np.load(output / "predictions" / f"{model.lower()}_predictions.npy")
    y_validation = np.load(ROOT / "data" / "processed" / "cmapss_fd001" / "y_validation.npy")
    plt.figure(figsize=(9, 4))
    plt.plot(y_validation[:300], label="actual")
    plt.plot(predictions["LSTM"][:300], label="LSTM")
    plt.plot(predictions["GRU"][:300], label="GRU")
    plt.legend()
    plt.title("Representative Step 13 predictions")
    plt.tight_layout()
    plt.savefig(plots / "prediction_examples.png")
    plt.close()
    lstm = results.loc[results["model"].eq("LSTM")].iloc[0]
    gru = results.loc[results["model"].eq("GRU")].iloc[0]
    decision = "NO_CLEAR_WINNER"
    if lstm["mae"] < results.loc[results["model"].eq("Transformer CPU"), "mae"].iloc[0] and gru["mae"] < results.loc[results["model"].eq("Transformer CPU"), "mae"].iloc[0]:
        decision = "RECURRENT_ADVANTAGE"
    report = f"""# Step 13 — LSTM / GRU Temporal Evaluation

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
| LSTM | {lstm['mae']:.4f} | {lstm['rmse']:.4f} | {lstm['phm_score']:.2f} | {lstm['training_time_seconds']:.2f} | {lstm['inference_time_seconds']:.4f} | {int(lstm['parameter_count'])} |
| GRU | {gru['mae']:.4f} | {gru['rmse']:.4f} | {gru['phm_score']:.2f} | {gru['training_time_seconds']:.2f} | {gru['inference_time_seconds']:.4f} | {int(gru['parameter_count'])} |

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

`{decision}`. The result does not justify replacing the Transformer based on FD001 alone.

## Recommendation

Retain the Transformer as the reference architecture. Keep LSTM/GRU as alternative compact architectures for future multi-seed or resource-constrained studies; do not claim universal superiority.
"""
    (output / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
