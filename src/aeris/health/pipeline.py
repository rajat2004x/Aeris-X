"""Step 12 health-state estimation benchmark."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aeris.data.cmapss import load_cmapss, make_windows
from aeris.health.estimator import MeanHealthEstimator
from aeris.health.evaluation import regression_metrics, trajectory_metrics
from aeris.health.labels import health_state, rul_to_health
from aeris.health.trajectory import early_late_errors, split_by_engine_order
from aeris.models.temporal import TransformerRegressor


def _owners(data_dir: Path, metadata: dict[str, Any], count: int) -> np.ndarray:
    raw_root = data_dir.parents[1] / "raw"
    dataset = load_cmapss(raw_root, "FD001")
    _, owners = make_windows(dataset.train, metadata["validation_engines"], metadata["feature_columns"], metadata["window_size"])
    if len(owners) != count:
        raise ValueError("Raw validation ownership does not align with processed arrays.")
    return owners


def _flatten(values: np.ndarray) -> np.ndarray:
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("temporal inputs must be finite 3-D arrays.")
    return values.reshape(len(values), -1)


def _predict_transformer(X: np.ndarray, checkpoint: Path) -> np.ndarray:
    model = TransformerRegressor(input_size=X.shape[2], max_length=X.shape[1])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32)).numpy()


def run_step12(
    data_dir: str | Path = "data/processed/cmapss_fd001",
    output_dir: str | Path = "reports/experiments/step12_health_state",
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate frozen health estimators with engine-disjoint development/final data."""
    np.random.seed(seed)
    data = Path(data_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    X_train = np.load(data / "X_train.npy")
    y_train_rul = np.load(data / "y_train.npy").astype(float)
    X_validation = np.load(data / "X_validation.npy")
    y_validation_rul = np.load(data / "y_validation.npy").astype(float)
    reference_max = float(np.max(y_train_rul))
    y_train = rul_to_health(y_train_rul, reference_max)
    y_validation = rul_to_health(y_validation_rul, reference_max)
    owners = _owners(data, metadata, len(X_validation))
    dev_mask, final_mask, dev_engines, final_engines = split_by_engine_order(owners)
    train_features = _flatten(X_train)
    validation_features = _flatten(X_validation)

    estimators: dict[str, Any] = {
        "mean_health": MeanHealthEstimator().fit(train_features, y_train),
        "linear_regression": LinearRegression().fit(train_features, y_train),
    }
    xgb_path = Path("reports/experiments/step7_classical_ml/models/xgboost.joblib")
    if not xgb_path.exists():
        raise FileNotFoundError(f"Required verified XGBoost artifact is missing: {xgb_path}")
    estimators["xgboost"] = joblib.load(xgb_path)
    transformer_path = Path("reports/experiments/step8_temporal_dl/models/transformer.pt")
    if not transformer_path.exists():
        raise FileNotFoundError(f"Required verified Transformer artifact is missing: {transformer_path}")

    def predict(name: str, features: np.ndarray, temporal: np.ndarray) -> np.ndarray:
        if name == "transformer":
            rul = _predict_transformer(temporal, transformer_path)
        else:
            rul = estimators[name].predict(features)
        return rul_to_health(np.asarray(rul, dtype=float), reference_max)

    names = ["mean_health", "linear_regression", "xgboost", "transformer"]
    predictions: dict[str, np.ndarray] = {}
    rows = []
    for name in names:
        started = time.perf_counter()
        predictions[name] = predict(name, validation_features, X_validation)
        elapsed = time.perf_counter() - started
        dev_metrics = regression_metrics(y_validation[dev_mask], predictions[name][dev_mask])
        rows.append({"model": name, "split": "development", **dev_metrics, "inference_time_seconds": elapsed})
    development = pd.DataFrame(rows)
    selected_name = str(development.sort_values(["mae", "rmse", "model"]).iloc[0]["model"])
    final_prediction = predictions[selected_name][final_mask]
    final_actual = y_validation[final_mask]
    final_metrics = regression_metrics(final_actual, final_prediction)
    final_trajectory = trajectory_metrics(final_actual, final_prediction)
    final_metrics.update(early_late_errors(final_actual, final_prediction))
    final_metrics.update({key: value for key, value in final_trajectory.items() if key not in final_metrics})
    final_metrics["model"] = selected_name
    final_metrics["development_engines"] = len(dev_engines)
    final_metrics["final_engines"] = len(final_engines)

    model_rows = []
    for name in names:
        model_rows.append({"model": name, "split": "final", **regression_metrics(y_validation[final_mask], predictions[name][final_mask])})
    comparison = pd.concat([development, pd.DataFrame(model_rows)], ignore_index=True)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    pd.DataFrame([final_metrics]).to_csv(output / "final_results.csv", index=False)

    trajectory_rows = []
    for engine_id in final_engines:
        selected = final_mask & (owners == engine_id)
        trajectory_rows.append({
            "engine_id": engine_id,
            "windows": int(selected.sum()),
            **trajectory_metrics(y_validation[selected], predictions[selected_name][selected]),
        })
    pd.DataFrame(trajectory_rows).to_csv(output / "trajectory_results.csv", index=False)
    state_frame = pd.DataFrame({
        "engine_id": owners[final_mask],
        "true_health": final_actual,
        "predicted_health": final_prediction,
        "state": health_state(final_prediction),
    })
    state_frame.to_csv(output / "health_state_results.csv", index=False)

    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    for engine_id in final_engines[:3]:
        selected = final_mask & (owners == engine_id)
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.plot(y_validation[selected], label="true health")
        axis.plot(predictions[selected_name][selected], label=f"predicted health ({selected_name})")
        axis.set(xlabel="engine cycle window order", ylabel="health (0–1)", title=f"FD001 engine {engine_id} health trajectory")
        axis.legend()
        figure.tight_layout()
        figure.savefig(plots / f"health_trajectory_engine_{engine_id}.png", dpi=150)
        plt.close(figure)
    axis = comparison[comparison["split"] == "final"].set_index("model")["mae"].plot(kind="bar", title="Final health MAE")
    axis.set_ylabel("MAE")
    axis.figure.tight_layout()
    axis.figure.savefig(plots / "model_comparison.png", dpi=150)
    plt.close(axis.figure)
    figure, axis = plt.subplots()
    axis.hist(final_prediction - final_actual, bins=30)
    axis.set(title="Final health error distribution", xlabel="predicted - true health", ylabel="windows")
    figure.tight_layout()
    figure.savefig(plots / "error_distribution.png", dpi=150)
    plt.close(figure)
    figure, axis = plt.subplots()
    axis.scatter(final_actual, final_prediction, alpha=0.25, s=8)
    axis.set(xlabel="true health", ylabel="predicted health", title="Final health correlation")
    figure.tight_layout()
    figure.savefig(plots / "health_correlation.png", dpi=150)
    plt.close(figure)
    trajectory_frame = pd.DataFrame(trajectory_rows)
    axis = trajectory_frame.set_index("engine_id")["degradation_direction_consistency"].plot(
        kind="bar", title="Degradation direction consistency by engine"
    )
    axis.set_ylabel("fraction of matching health direction")
    axis.figure.tight_layout()
    axis.figure.savefig(plots / "degradation_consistency.png", dpi=150)
    plt.close(axis.figure)

    manifest = {
        "step": "12",
        "dataset": "NASA C-MAPSS FD001",
        "scope": "FD001-only scope for this milestone.",
        "seed": seed,
        "health_definition": "clip(RUL / max(y_train_RUL), 0, 1); 1 is high remaining life and 0 is near failure.",
        "reference_max_rul": reference_max,
        "train_windows": len(X_train),
        "validation_windows": len(X_validation),
        "development_engines": dev_engines,
        "final_engines": final_engines,
        "selected_model": selected_name,
        "selection_rule": "lowest development MAE, then RMSE, then model name",
        "temporal_model": "existing frozen Step 8 Transformer checkpoint; no Step 8 retraining",
        "scaler": "prepared Step 6 arrays reused unchanged; no scaler refit on development/final data",
        "uncertainty": "not included; prior Step 9 coverage was poor and no calibrated uncertainty is claimed",
    }
    (output / "config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "leakage_audit.json").write_text(json.dumps({
        "fit_data": "X_train.npy and y_train.npy only",
        "development_data": "validation engines listed in development_engines",
        "final_data": "validation engines listed in final_engines",
        "reference_max_rul_fit_on": "y_train.npy only",
        "model_selection": "development MAE/RMSE only",
        "final_holdout_used_for_selection": False,
        "final_holdout_evaluated_after_freezing": True,
        "future_rul_or_health_features": False,
        "engine_overlap_train_development_final": False,
        "raw_data_used": "FD001 train file only to recover deterministic ownership metadata",
    }, indent=2), encoding="utf-8")
    (output / "FINAL_STEP12_REPORT.md").write_text(
        f"""# Step 12 — Health-State Estimation

## Research question
What does AERIS-X know about the current system health from temporal telemetry?

## Health definition
Health is continuous normalized remaining useful life: `clip(RUL / {reference_max:.1f}, 0, 1)`.
This reuses the repository's RUL-derived health convention without inventing labels.
Health is primary; anomaly and sensor-fault outputs remain Step 11 and Step 10 concerns.

## Dataset and leakage controls
The evaluation uses the existing processed FD001 arrays (`(30, 15)` windows). Validation
engines were recovered from the existing raw FD001 file solely for ownership and split
metadata. Training uses {len(X_train)} windows; development uses engines {dev_engines};
the frozen final holdout uses engines {final_engines}. See `leakage_audit.json`.

## Models evaluated
Mean health, Linear Regression, the verified Step 7 XGBoost artifact, and the existing
Step 8 Transformer artifact. The final model was selected only by development MAE/RMSE.

## Final results

| Metric | Value |
|---|---:|
| Selected model | {selected_name} |
| MAE | {final_metrics["mae"]:.6f} |
| RMSE | {final_metrics["rmse"]:.6f} |
| R² | {final_metrics["r2"]:.6f} |
| Pearson correlation | {final_metrics["correlation"]:.6f} |
| Spearman correlation | {final_metrics["spearman_correlation"]:.6f} |
| Smoothness mean absolute delta | {final_metrics["smoothness_mean_abs_delta"]:.6f} |
| Monotonicity violations | {final_metrics["monotonicity_violations"]} |
| Direction consistency | {final_metrics["degradation_direction_consistency"]:.6f} |
| Early-life MAE | {final_metrics["early_life_mae"]:.6f} |
| Late-life MAE | {final_metrics["late_life_mae"]:.6f} |

## Health trajectories and error analysis
Trajectory plots are provided for representative final-holdout engines. The diagnostics
show whether the estimator tracks degradation, reacts too slowly/aggressively, or
predicts healthy systems as degraded. Continuous health is not forcibly monotonic.

## Uncertainty
No uncertainty is claimed. The previous uncertainty experiment had poor empirical
coverage, so this milestone reports point estimates only.

## Limitations and conclusion
FD001-only scope for this milestone; FD002–FD004 support is not claimed. Prepared
arrays do not contain original engine IDs, so ownership is recovered from the existing
raw FD001 file without changing features. The final result answers health only in the
normalized RUL sense and does not establish operational health certification.

**What AERIS-X knows:** the selected estimator provides a continuous estimate of
remaining-life-derived health for frozen FD001 engines, with the measured accuracy and
trajectory consistency above.

**Where it fails:** errors, monotonicity violations, and late/early-life MAE identify
where telemetry-to-health mapping is unreliable; no universal superiority is claimed.
""",
        encoding="utf-8",
    )
    return {"selected_model": selected_name, "final_metrics": final_metrics, "manifest": manifest}
