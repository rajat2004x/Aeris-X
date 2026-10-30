"""Run Step 9 Monte Carlo Dropout uncertainty quantification."""

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
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aeris.evaluation.benchmark import load_ml_ready_dataset
from aeris.evaluation.metrics import mean_absolute_error, phm_score, root_mean_squared_error
from aeris.evaluation.uncertainty import interval_coverage, relative_uncertainty, summarize_mc_predictions
from aeris.models.temporal import TransformerRegressor
from aeris.training.datasets import TemporalDataset
from aeris.training.trainer import seed_everything, select_device


def enable_dropout(model: torch.nn.Module) -> None:
    """Enable only dropout modules while keeping the model otherwise in eval mode."""
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
    # TransformerEncoder uses a training-dependent fast path around its
    # dropout submodules. There are no normalization layers in this model,
    # so enabling only the encoder preserves evaluation behavior elsewhere
    # while ensuring its configured dropout is sampled.
    if hasattr(model, "encoder"):
        model.encoder.train()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    pearson = float(np.corrcoef(left, right)[0, 1]) if np.std(left) and np.std(right) else float("nan")
    spearman = float(spearmanr(left, right).statistic) if np.std(left) and np.std(right) else float("nan")
    return pearson, spearman


def _quantile_analysis(frame: pd.DataFrame) -> list[dict[str, float | int | str]]:
    frame = frame.copy()
    frame["uncertainty_quantile"] = pd.qcut(
        frame["predictive_std"].rank(method="first"),
        5,
        labels=["lowest_20", "20_40", "40_60", "60_80", "highest_20"],
    )
    grouped = frame.groupby("uncertainty_quantile", observed=True)
    return [
        {
            "group": str(name),
            "sample_count": int(len(group)),
            "mean_uncertainty": float(group["predictive_std"].mean()),
            "mae": float(group["absolute_error"].mean()),
            "rmse": float(np.sqrt(np.mean(group["absolute_error"] ** 2))),
        }
        for name, group in grouped
    ]


def _regime_analysis(frame: pd.DataFrame) -> list[dict[str, float | int | str]]:
    bins = [-np.inf, 20, 40, 60, 80, np.inf]
    labels = ["0-20", "20-40", "40-60", "60-80", "80+"]
    frame = frame.copy()
    frame["rul_regime"] = pd.cut(frame["true_rul"], bins=bins, labels=labels, right=False)
    grouped = frame.groupby("rul_regime", observed=True)
    return [
        {
            "regime": str(name),
            "sample_count": int(len(group)),
            "mae": float(group["absolute_error"].mean()),
            "rmse": float(np.sqrt(np.mean(group["absolute_error"] ** 2))),
            "mean_uncertainty": float(group["predictive_std"].mean()),
            "mean_interval_width": float(group["interval_width"].mean()),
            "coverage": float(group["covered"].mean()),
        }
        for name, group in grouped
    ]


def _plots(frame: pd.DataFrame, output: Path, quantiles: list[dict], regimes: list[dict], nominal_coverage: float) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    ordered = frame.reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(ordered.index, ordered["true_rul"], label="True RUL", linewidth=1)
    axis.plot(ordered.index, ordered["mean_prediction"], label="Mean prediction", linewidth=1)
    axis.fill_between(ordered.index, ordered["lower_interval"], ordered["upper_interval"], alpha=0.2, label="Approx. 95% interval")
    axis.set(xlabel="Validation sample", ylabel="RUL", title="RUL predictions with MC Dropout uncertainty")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "predictions_with_intervals.png", dpi=150)
    plt.close(figure)

    for name, x, y, xlabel, ylabel in (
        ("uncertainty_vs_absolute_error", frame["predictive_std"], frame["absolute_error"], "Predictive standard deviation", "Absolute error"),
        ("uncertainty_distribution", frame["predictive_std"], None, "Predictive standard deviation", "Count"),
    ):
        figure, axis = plt.subplots()
        if y is None:
            axis.hist(x, bins=40)
        else:
            axis.scatter(x, y, alpha=0.25, s=10)
        axis.set(xlabel=xlabel, ylabel=ylabel, title=name.replace("_", " ").title())
        figure.tight_layout()
        figure.savefig(plots / f"{name}.png", dpi=150)
        plt.close(figure)

    figure, axis = plt.subplots()
    coverage = frame["covered"].astype(int).rolling(50, min_periods=1).mean()
    axis.plot(coverage, label="Rolling empirical coverage")
    axis.axhline(nominal_coverage, linestyle="--", color="black", label="Nominal 95%")
    axis.set(xlabel="Validation sample", ylabel="Coverage", title="Prediction interval coverage")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "prediction_interval_coverage.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots()
    axis.bar([item["regime"] for item in regimes], [item["mean_uncertainty"] for item in regimes])
    axis.set(xlabel="True RUL regime", ylabel="Mean predictive standard deviation", title="Uncertainty by RUL regime")
    figure.tight_layout()
    figure.savefig(plots / "uncertainty_by_rul_regime.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots()
    axis.bar([item["group"] for item in quantiles], [item["mae"] for item in quantiles])
    axis.set(xlabel="Uncertainty quantile", ylabel="MAE", title="Error by uncertainty quantile")
    figure.tight_layout()
    figure.savefig(plots / "error_by_uncertainty_quantile.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/processed/cmapss_fd001")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "reports/experiments/step8_temporal_dl/gpu/models/transformer_gpu.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/experiments/step9_uncertainty")
    parser.add_argument("--mc-samples", type=int, default=100)
    args = parser.parse_args()
    if args.mc_samples < 2:
        raise ValueError("mc-samples must be at least 2.")

    seed_everything(42)
    device = select_device()
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Selected device: {device}")
    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(device)}")

    _, _, X_validation, y_validation = load_ml_ready_dataset(args.data_dir)
    loader = DataLoader(TemporalDataset(X_validation, y_validation), batch_size=128, shuffle=False)
    model = TransformerRegressor(input_size=15, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, max_length=30)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.to(device)
    enable_dropout(model)

    passes: list[np.ndarray] = []
    _synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.mc_samples):
            batch_predictions = []
            for features, _ in loader:
                batch_predictions.append(model(features.to(device)).cpu().numpy())
            passes.append(np.concatenate(batch_predictions))
    _synchronize(device)
    inference_seconds = time.perf_counter() - started
    mc_predictions = np.stack(passes, axis=1)
    summary = summarize_mc_predictions(mc_predictions)
    frame = pd.DataFrame({
        "sample_index": np.arange(len(y_validation)),
        "true_rul": y_validation,
        **summary,
    })
    frame["absolute_error"] = np.abs(frame["mean_prediction"] - frame["true_rul"])
    frame["relative_uncertainty"] = relative_uncertainty(frame["mean_prediction"], frame["predictive_std"])
    frame["covered"] = (frame["true_rul"] >= frame["lower_interval"]) & (frame["true_rul"] <= frame["upper_interval"])

    nominal = 0.95
    coverage = interval_coverage(y_validation, frame["lower_interval"], frame["upper_interval"])
    pearson, spearman = _correlation(frame["predictive_std"].to_numpy(), frame["absolute_error"].to_numpy())
    quantiles = _quantile_analysis(frame)
    regimes = _regime_analysis(frame)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "predictions.csv", index=False)
    np.save(output / "mc_predictions.npy", mc_predictions)
    metrics = {
        "model": "Transformer",
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "seed": 42,
        "mc_samples": args.mc_samples,
        "inference_time_seconds": inference_seconds,
        "point_prediction": {
            "mae": mean_absolute_error(y_validation, frame["mean_prediction"].to_numpy()),
            "rmse": root_mean_squared_error(y_validation, frame["mean_prediction"].to_numpy()),
            "phm_score": phm_score(y_validation, frame["mean_prediction"].to_numpy()),
        },
        "uncertainty": {
            "mean_predictive_std": float(frame["predictive_std"].mean()),
            "median_predictive_std": float(frame["predictive_std"].median()),
            "average_interval_width": float(frame["interval_width"].mean()),
            "median_interval_width": float(frame["interval_width"].median()),
            "min_interval_width": float(frame["interval_width"].min()),
            "max_interval_width": float(frame["interval_width"].max()),
            "nominal_coverage": nominal,
            "observed_coverage": coverage,
            "coverage_error": coverage - nominal,
            "pearson_uncertainty_absolute_error": pearson,
            "spearman_uncertainty_absolute_error": spearman,
        },
        "uncertainty_quantiles": quantiles,
        "rul_regimes": regimes,
    }
    (output / "results.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics["point_prediction"] | metrics["uncertainty"]]).to_csv(output / "results.csv", index=False)
    (output / "config.json").write_text(json.dumps({
        "seed": 42, "mc_samples": args.mc_samples, "batch_size": 128,
        "interval": "mean +/- 1.96 * sample standard deviation",
        "architecture": {"input_size": 15, "d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.1, "max_length": 30},
    }, indent=2), encoding="utf-8")
    _plots(frame, output, quantiles, regimes, nominal)
    highest_uncertainty_group = next(item for item in quantiles if item["group"] == "highest_20")
    largest_error_group = max(quantiles, key=lambda item: item["mae"])
    report = f"""# Step 9: Uncertainty Quantification

## Objective

Evaluate whether the existing AERIS-X Transformer provides useful uncertainty
estimates using Monte Carlo Dropout.

## Method and integrity

The existing Transformer checkpoint was loaded without retraining. The
validation arrays from Step 6 were reused unchanged. The model was kept in
evaluation mode except for its dropout layers, which were selectively enabled
for `{args.mc_samples}` stochastic forward passes. No preprocessing, scaler,
split, architecture, or previous benchmark artifact was changed.

The interval is an **approximate 95% prediction interval based on MC Dropout
predictive variability**, calculated as mean +/- 1.96 * sample standard
deviation. It is not a formally guaranteed 95% confidence interval.

## Environment and reproducibility

- Checkpoint: `{args.checkpoint}`
- Device: `{device}`
- GPU: `{metrics["gpu_name"]}`
- PyTorch: `{torch.__version__}`
- CUDA: `{torch.version.cuda}`
- Seed: `42`
- MC samples: `{args.mc_samples}`

## Point prediction

- MAE: `{metrics["point_prediction"]["mae"]:.6f}`
- RMSE: `{metrics["point_prediction"]["rmse"]:.6f}`
- PHM: `{metrics["point_prediction"]["phm_score"]:.6f}`

## Uncertainty and coverage

- Mean predictive standard deviation: `{metrics["uncertainty"]["mean_predictive_std"]:.6f}`
- Median predictive standard deviation: `{metrics["uncertainty"]["median_predictive_std"]:.6f}`
- Average interval width: `{metrics["uncertainty"]["average_interval_width"]:.6f}`
- Median interval width: `{metrics["uncertainty"]["median_interval_width"]:.6f}`
- Nominal coverage: `95.00%`
- Observed coverage: `{coverage:.2%}`
- Coverage error: `{coverage - nominal:.2%}`
- Pearson uncertainty/error correlation: `{pearson:.6f}`
- Spearman uncertainty/error correlation: `{spearman:.6f}`

## Uncertainty vs error

The highest-uncertainty group had MAE `{highest_uncertainty_group["mae"]:.6f}`,
while the largest grouped MAE was `{largest_error_group["mae"]:.6f}` in
`{largest_error_group["group"]}`. This is descriptive evidence about the
relationship in this validation set, not
a claim of universal calibration or deployment reliability.

## RUL regimes

{pd.DataFrame(regimes).to_string(index=False)}

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
"""
    (output / "uncertainty_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
