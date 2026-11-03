"""Reproducible comparison of completed AERIS-X model experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_COLUMNS = [
    "model",
    "variant",
    "model_family",
    "approach",
    "mae",
    "rmse",
    "phm_score",
    "training_time_seconds",
    "inference_time_seconds",
    "inference_time_per_sample",
    "parameter_count",
    "model_size_mb",
    "device",
]


def _number(value: Any) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def _load_rows(step7_dir: Path, step8_dir: Path) -> list[dict[str, Any]]:
    classical = pd.read_csv(step7_dir / "comparison_results.csv")
    temporal = json.loads((step8_dir / "results.json").read_text(encoding="utf-8"))
    gpu = json.loads((step8_dir / "gpu" / "transformer_gpu_results.json").read_text(encoding="utf-8"))["gpu"]
    rows: list[dict[str, Any]] = []

    classical_labels = {
        "mean_rul": ("Mean RUL", "classical", "classical"),
        "linear_regression": ("Linear Regression", "classical", "classical"),
        "random_forest": ("Random Forest", "classical", "classical"),
        "xgboost": ("XGBoost", "classical", "classical"),
    }
    for record in classical.to_dict(orient="records"):
        label, family, approach = classical_labels[record["model"]]
        model_path = step7_dir / "models" / f"{record['model']}.joblib"
        rows.append({
            "model": label,
            "variant": "reference",
            "model_family": family,
            "approach": approach,
            "mae": _number(record["mae"]),
            "rmse": _number(record["rmse"]),
            "phm_score": _number(record["phm_score"]),
            "training_time_seconds": _number(record["training_time_seconds"]),
            "inference_time_seconds": _number(record["inference_time_seconds"]),
            "inference_time_per_sample": _number(record["inference_time_seconds"] / 2627),
            "parameter_count": None,
            "model_size_mb": _number(model_path.stat().st_size / (1024**2)),
            "device": "reference",
        })

    for record in temporal:
        if record["model"] == "XGBoost":
            continue
        rows.append({
            "model": record["model"],
            "variant": "cpu",
            "model_family": "temporal",
            "approach": "temporal",
            **{column: _number(record.get(column)) for column in (
                "mae", "rmse", "phm_score", "training_time_seconds",
                "inference_time_seconds", "inference_time_per_sample",
                "parameter_count", "model_size_mb",
            )},
            "device": record["device"],
        })

    gpu_path = step8_dir / "gpu" / "models" / "transformer_gpu.pt"
    rows.append({
        "model": "Transformer",
        "variant": "gpu",
        "model_family": "temporal",
        "approach": "temporal",
        "mae": _number(gpu["mae"]),
        "rmse": _number(gpu["rmse"]),
        "phm_score": _number(gpu["phm_score"]),
        "training_time_seconds": _number(gpu["training_time_seconds"]),
        "inference_time_seconds": _number(gpu["inference_time_seconds"]),
        "inference_time_per_sample": _number(gpu["inference_time_per_sample"]),
        "parameter_count": _number(gpu["parameter_count"]),
        "model_size_mb": _number(gpu_path.stat().st_size / (1024**2)),
        "device": gpu["gpu_name"],
    })
    return rows


def _rankings(frame: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "mae_ascending": frame.sort_values(["mae", "model", "variant"]).apply(
            lambda row: f"{row['model']} ({row['variant']})", axis=1
        ).tolist(),
        "rmse_ascending": frame.sort_values(["rmse", "model", "variant"]).apply(
            lambda row: f"{row['model']} ({row['variant']})", axis=1
        ).tolist(),
        "phm_ascending": frame.sort_values(["phm_score", "model", "variant"]).apply(
            lambda row: f"{row['model']} ({row['variant']})", axis=1
        ).tolist(),
        "training_time_ascending": frame.sort_values(
            ["training_time_seconds", "model", "variant"]
        ).apply(lambda row: f"{row['model']} ({row['variant']})", axis=1).tolist(),
        "inference_time_per_sample_ascending": frame.sort_values(
            ["inference_time_per_sample", "model", "variant"]
        ).apply(lambda row: f"{row['model']} ({row['variant']})", axis=1).tolist(),
    }


def _plot(frame: pd.DataFrame, output: Path) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    labels = frame.apply(lambda row: f"{row['model']}\n{row['variant']}", axis=1)
    for metric, filename, title in (
        ("mae", "accuracy_mae.png", "MAE (lower is better)"),
        ("rmse", "accuracy_rmse.png", "RMSE (lower is better)"),
        ("phm_score", "accuracy_phm.png", "PHM score (lower is better)"),
        ("training_time_seconds", "training_cost.png", "Training time (seconds)"),
        ("inference_time_per_sample", "inference_cost.png", "Inference time per sample (seconds)"),
    ):
        figure, axis = plt.subplots(figsize=(11, 5))
        axis.bar(labels, frame[metric])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=55)
        figure.tight_layout()
        figure.savefig(plots / filename, dpi=150)
        plt.close(figure)


def _write_report(payload: dict[str, Any], output: Path) -> None:
    records = payload["records"]
    lines = [
        "# Step 9: Experiment Comparison and Analysis",
        "",
        "## Scope",
        "This report compares completed Step 7 and Step 8 artifacts. It does not retrain models or modify prior experiment directories.",
        "",
        "## Results",
        "",
        "| Model | Variant | Family | MAE | RMSE | PHM | Train seconds | Inference seconds/sample | Parameters | Device |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in records:
        lines.append(
            f"| {row['model']} | {row['variant']} | {row['model_family']} | "
            f"{row['mae']:.6f} | {row['rmse']:.6f} | {row['phm_score']:.2f} | "
            f"{row['training_time_seconds']:.6f} | {row['inference_time_per_sample']:.9f} | "
            f"{row['parameter_count'] if row['parameter_count'] is not None else 'N/A'} | {row['device']} |"
        )
    derived = payload["derived_comparisons"]
    lines.extend([
        "",
        "## Findings",
        "",
        f"- Best MAE and RMSE: {payload['interpretation']['best_mae']}.",
        f"- Best PHM: {payload['interpretation']['best_phm']}.",
        f"- CPU Transformer versus XGBoost MAE relative change: {derived['transformer_cpu_vs_xgboost_mae_relative_change']:.4%}.",
        f"- Historical GPU Transformer training speedup versus CPU: {derived['transformer_gpu_training_speedup_vs_cpu']:.3f}x.",
        f"- Historical GPU Transformer inference speedup versus CPU: {derived['transformer_gpu_inference_speedup_vs_cpu']:.3f}x.",
        "",
        "## Interpretation",
        "",
        payload["interpretation"]["accuracy_efficiency_note"],
        payload["interpretation"]["gpu_note"],
        "",
        "## Limitations",
        "",
    ])
    lines.extend(f"- {item}" for item in payload["limitations"])
    (output / "FINAL_STEP9_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_comparison(
    step7_dir: str | Path = "reports/experiments/step7_classical_ml",
    step8_dir: str | Path = "reports/experiments/step8_temporal_dl",
    output_dir: str | Path = "reports/experiments/step9_comparison",
) -> pd.DataFrame:
    """Read completed experiment artifacts and write a deterministic comparison."""
    step7_path, step8_path, output = Path(step7_dir), Path(step8_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(step7_path, step8_path)
    frame = pd.DataFrame(rows, columns=MODEL_COLUMNS)
    frame.to_csv(output / "comparison_results.csv", index=False)
    records = frame.to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if pd.isna(value):
                record[key] = None

    transformer_cpu = frame.query("model == 'Transformer' and variant == 'cpu'").iloc[0]
    transformer_gpu = frame.query("model == 'Transformer' and variant == 'gpu'").iloc[0]
    xgboost = frame.query("model == 'XGBoost'").iloc[0]
    payload = {
        "schema_version": 1,
        "source_artifacts": [
            str(step7_path / "comparison_results.csv"),
            str(step8_path / "results.json"),
            str(step8_path / "gpu" / "transformer_gpu_results.json"),
        ],
        "records": records,
        "rankings": _rankings(frame),
        "derived_comparisons": {
            "transformer_cpu_vs_xgboost_mae_relative_change": float(
                (transformer_cpu["mae"] - xgboost["mae"]) / xgboost["mae"]
            ),
            "transformer_cpu_vs_xgboost_rmse_relative_change": float(
                (transformer_cpu["rmse"] - xgboost["rmse"]) / xgboost["rmse"]
            ),
            "transformer_cpu_vs_xgboost_phm_relative_change": float(
                (transformer_cpu["phm_score"] - xgboost["phm_score"]) / xgboost["phm_score"]
            ),
            "transformer_gpu_training_speedup_vs_cpu": float(
                transformer_cpu["training_time_seconds"] / transformer_gpu["training_time_seconds"]
            ),
            "transformer_gpu_inference_speedup_vs_cpu": float(
                transformer_cpu["inference_time_seconds"] / transformer_gpu["inference_time_seconds"]
            ),
        },
        "interpretation": {
            "best_mae": "Transformer (cpu)",
            "best_rmse": "Transformer (cpu)",
            "best_phm": "Transformer (gpu)",
            "accuracy_efficiency_note": "Transformer (cpu) has the strongest CPU accuracy and the smallest temporal parameter count; XGBoost remains much faster to train.",
            "gpu_note": "The historical GPU run reduced Transformer training and inference time, but produced different validation metrics and is not a like-for-like accuracy win over the CPU run.",
        },
        "limitations": [
            "All model metrics use the same FD001 validation split, not an untouched test set.",
            "Timing values are historical single-run measurements and are hardware- and load-dependent.",
            "The GPU and CPU Transformer runs have different measured predictions, so speed comparisons do not imply identical accuracy.",
            "No statistical significance or repeated-seed analysis is claimed.",
        ],
    }
    (output / "comparison_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output / "experiment_manifest.json").write_text(json.dumps({
        "step": "9",
        "purpose": "Compare completed Step 7 and Step 8 experiments without retraining.",
        "dataset": "data/processed/cmapss_fd001",
        "validation_protocol": "Existing FD001 validation arrays and metrics from source artifacts.",
        "reproducible_inputs": payload["source_artifacts"],
        "models": [f"{record['model']} ({record['variant']})" for record in records],
        "derived_fields": list(payload["derived_comparisons"]),
    }, indent=2), encoding="utf-8")
    _write_report(payload, output)
    _plot(frame, output)
    return frame
