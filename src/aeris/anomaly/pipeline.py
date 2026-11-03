"""Deterministic Step 11 anomaly-detection pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aeris.anomaly.events import event_metrics, persistence_flags
from aeris.anomaly.evaluation import detection_metrics, percentile_threshold
from aeris.anomaly.features import (
    fit_robust_baseline,
    sensor_anomaly_scores,
    system_anomaly_scores,
    top_sensor_indices,
)
from aeris.faults.injection import inject_fault


def _inject_multisensor_anomaly(
    window: np.ndarray,
    seed: int,
    training_std: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """Inject a system-level two-sensor deviation for evaluation only."""
    result = window.copy()
    targets = [0, 1]
    for offset, sensor in enumerate(targets):
        result, _ = inject_fault(
            result,
            sensor_index=sensor,
            fault_type="BIAS" if offset == 0 else "DRIFT",
            onset=0.5,
            severity=1.5,
            training_std=float(training_std[sensor]),
            seed=seed + offset,
        )
    return result, targets


def run_step11(
    data_dir: str | Path = "data/processed/cmapss_fd001",
    output_dir: str | Path = "reports/experiments/step11_anomaly_detection",
    seed: int = 42,
    threshold_percentile: float = 99.0,
    persistence: int = 3,
) -> dict[str, Any]:
    """Fit on Step 6 training windows and evaluate a frozen synthetic holdout."""
    rng = np.random.default_rng(seed)
    data = Path(data_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    X_train = np.load(data / "X_train.npy")
    X_validation = np.load(data / "X_validation.npy")
    baseline = fit_robust_baseline(X_train)
    development = X_validation[: len(X_validation) // 2]
    holdout = X_validation[len(X_validation) // 2 :]
    development_scores = system_anomaly_scores(sensor_anomaly_scores(development, baseline))
    threshold = percentile_threshold(development_scores, threshold_percentile)
    training_std = np.std(X_train.reshape(-1, X_train.shape[-1]), axis=0)

    sample_count = min(400, len(holdout))
    clean = holdout[:sample_count].copy()
    anomalous = clean.copy()
    labels = np.zeros(sample_count, dtype=int)
    anomaly_indices = np.arange(0, sample_count, 4)
    for index in anomaly_indices:
        anomalous[index], _ = _inject_multisensor_anomaly(anomalous[index], seed + int(index), training_std)
        labels[index] = 1
    scores_by_sensor = sensor_anomaly_scores(anomalous, baseline)
    scores = system_anomaly_scores(scores_by_sensor)
    raw_alarms = scores >= threshold
    alarms = persistence_flags(raw_alarms, persistence)
    metrics = detection_metrics(labels, scores, threshold)
    events = event_metrics(labels, alarms)
    normal_scores = system_anomaly_scores(sensor_anomaly_scores(clean[labels == 0], baseline))
    contributions = top_sensor_indices(scores_by_sensor, top_k=3)
    contribution_rows = []
    for index in anomaly_indices:
        contribution_rows.append({
            "sample_index": int(index),
            "label": int(labels[index]),
            "score": float(scores[index]),
            "top_sensor_1": int(contributions[index, 0]),
            "top_sensor_2": int(contributions[index, 1]),
            "top_sensor_3": int(contributions[index, 2]),
        })

    results = pd.DataFrame([{
        "dataset": "FD001",
        "detector": "robust_multisensor_mad",
        "threshold": threshold,
        "threshold_method": f"percentile_{threshold_percentile}_development_normal",
        "persistence": persistence,
        "development_windows": len(development),
        "holdout_windows": sample_count,
        "anomaly_windows": int(labels.sum()),
        **metrics,
        **events,
        "normal_holdout_fpr": float(np.mean(normal_scores >= threshold)),
        "false_alarms_per_engine": float(np.sum(alarms & (labels == 0)) / 15),
    }])
    results.to_csv(output / "final_results.csv", index=False)
    pd.DataFrame(contribution_rows).to_csv(output / "sensor_contribution.csv", index=False)
    pd.DataFrame([{
        "dataset": "FD001",
        "normal_holdout_windows": len(normal_scores),
        "normal_score_mean": float(np.mean(normal_scores)),
        "normal_score_p95": float(np.percentile(normal_scores, 95)),
        "normal_score_max": float(np.max(normal_scores)),
        "false_positive_rate": float(np.mean(normal_scores >= threshold)),
    }]).to_csv(output / "degradation_analysis.csv", index=False)
    pd.DataFrame([metrics | events]).to_csv(output / "event_results.csv", index=False)

    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.hist(normal_scores, bins=30, alpha=0.7, label="normal holdout")
    axis.axvline(threshold, color="red", linestyle="--", label="frozen threshold")
    axis.set(title="System anomaly score distribution", xlabel="score", ylabel="windows")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "score_distribution.png", dpi=150)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(scores, label="system score")
    axis.plot(labels * threshold, label="injected anomaly", alpha=0.8)
    axis.axhline(threshold, color="red", linestyle="--")
    axis.set(title="Frozen holdout anomaly evidence", xlabel="window", ylabel="score")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "representative_anomaly.png", dpi=150)
    plt.close(figure)

    manifest = {
        "step": "11",
        "dataset": "NASA C-MAPSS FD001",
        "data_dir": str(data),
        "train_windows": int(len(X_train)),
        "validation_windows": int(len(X_validation)),
        "window_shape": list(X_train.shape[1:]),
        "normal_training_definition": "All existing Step 6 X_train windows; no anomaly labels or injected scenarios used for fitting.",
        "development_definition": "First half of X_validation, used only for threshold calibration.",
        "final_holdout_definition": "Second half of X_validation; labels are generated only for controlled evaluation scenarios.",
        "detector": "per-sensor median/MAD robust scores aggregated as mean top-3 sensor evidence",
        "threshold": threshold,
        "threshold_method": f"{threshold_percentile}th percentile of development scores",
        "persistence": persistence,
        "seed": seed,
        "fault_protocol": "Two-sensor BIAS/DRIFT injection on every fourth holdout window; evaluation only.",
        "processed_fd002_fd004_available": False,
        "future_information_used": False,
        "final_labels_used_for_fit_or_threshold": False,
    }
    (output / "config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "leakage_audit.json").write_text(json.dumps({
        "fit_arrays": ["X_train.npy"],
        "threshold_arrays": ["first half of X_validation.npy"],
        "final_evaluation_arrays": ["second half of X_validation.npy"],
        "train_and_final_engine_split": "inherited from Step 6 prepared engine-level split",
        "anomaly_labels_used_for_fit": False,
        "anomaly_labels_used_for_threshold": False,
        "final_holdout_used_for_model_selection": False,
        "fault_injection_used_for_training": False,
        "future_cycles_used": False,
        "scope_note": "FD001 only; FD002-FD004 are unavailable in the processed evaluation contract.",
    }, indent=2), encoding="utf-8")
    (output / "FINAL_STEP11_REPORT.md").write_text(
        f"""# Step 11 — System Anomaly Detection

## Scope and protocol

This package evaluates a deterministic robust statistical detector on NASA C-MAPSS
FD001. The detector is fit on the existing Step 6 `X_train` windows only. The first
half of `X_validation` is development data for threshold selection; the second half
is a frozen holdout. Controlled two-sensor BIAS/DRIFT scenarios are injected only
for final evaluation. This is system-level abnormal-behavior detection, not Step 10
sensor FDI or fault-type classification.

The detector aggregates the mean of the three largest per-sensor median/MAD
deviations and applies a {persistence}-window persistence rule after a
{threshold_percentile}th-percentile development threshold.

## Results

| Metric | Value |
|---|---:|
| Threshold | {threshold:.6f} |
| Precision | {metrics["precision"]:.6f} |
| Recall | {metrics["recall"]:.6f} |
| F1 | {metrics["f1"]:.6f} |
| ROC-AUC | {metrics["roc_auc"]:.6f} |
| Normal holdout false-positive rate | {float(np.mean(normal_scores >= threshold)):.6f} |
| Event recall | {events["event_recall"]:.6f} |
| False alarms per engine | {float(np.sum(alarms & (labels == 0)) / 15):.6f} |

The low point recall and near-random ROC-AUC are retained as measured results; the
controlled injected scenarios do not establish production effectiveness. Event
recall is based on exact temporal overlap after persistence and should not be read
as evidence of broad anomaly-detection quality.

## Limitations

Only the processed FD001 contract is available, so cross-condition and cross-dataset
generalization to FD002–FD004 is not claimed. Normal degradation is measured on the
clean frozen holdout as a false-alarm analysis, not relabeled as abnormal. A
temporal autoencoder was not added because the available historical autoencoder
evidence documents poor practical detection and retraining would create a new
experiment outside this verification scope.

## Reproducibility and leakage

See `experiment_manifest.json`, `config.json`, and `leakage_audit.json`. No final
labels, injected anomalies, or future cycles are used for fitting or thresholding.
""",
        encoding="utf-8",
    )
    return {"manifest": manifest, "results": results.to_dict(orient="records")[0]}
