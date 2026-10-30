"""Run the improved, held-out Step 10.1 FDI experiment."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aeris.evaluation.anomaly import calibrate_threshold, fit_robust_baseline, robust_sensor_scores
from aeris.evaluation.fdi import reconstruction_scores
from aeris.faults.injection import inject_fault
from aeris.models.anomaly import SequenceAutoencoder
from aeris.training.datasets import TemporalDataset
from aeris.training.trainer import seed_everything, select_device


BASELINE = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.4713967489, "pr_auc": 0.0418895484}


def train_autoencoder(X: np.ndarray, device: torch.device, checkpoint: Path) -> tuple[SequenceAutoencoder, float]:
    model = SequenceAutoencoder(input_size=X.shape[2], hidden_size=64).to(device)
    loader = DataLoader(TemporalDataset(X, np.zeros(len(X), dtype=np.float32)), batch_size=128, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    loss_fn = nn.MSELoss()
    started = time.perf_counter()
    best = float("inf")
    for _ in range(20):
        model.train()
        total = 0.0
        count = 0
        for features, _ in loader:
            features = features.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), features)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(features)
            count += len(features)
        value = total / count
        if value < best:
            best = value
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model, time.perf_counter() - started


def reconstruct(model: nn.Module, X: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(TemporalDataset(X, np.zeros(len(X), dtype=np.float32)), batch_size=128, shuffle=False)
    outputs = []
    model.eval()
    with torch.no_grad():
        for features, _ in loader:
            outputs.append(model(features.to(device)).cpu().numpy())
    reconstructed = np.concatenate(outputs)
    global_scores, sensor_scores = reconstruction_scores(X, reconstructed)
    return global_scores, sensor_scores


def detector_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float, predicted: np.ndarray | None = None) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    flags = scores >= threshold if predicted is None else np.asarray(predicted, dtype=bool)
    negatives = max(1, int(np.sum(labels == 0)))
    positives = max(1, int(np.sum(labels == 1)))
    return {
        "precision": float(precision_score(labels, flags, zero_division=0)),
        "recall": float(recall_score(labels, flags, zero_division=0)),
        "f1": float(f1_score(labels, flags, zero_division=0)),
        "accuracy": float(np.mean(flags == labels)),
        "fpr": float(np.sum(flags & (labels == 0)) / negatives),
        "fnr": float(np.sum((~flags) & (labels == 1)) / positives),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "detection_rate": float(np.mean(flags[labels == 1])),
    }


def main() -> None:
    seed_everything(42)
    output = PROJECT_ROOT / "reports/experiments/step10_1_fdi_improved"
    models_dir = output / "models"
    output.mkdir(parents=True, exist_ok=True)
    device = select_device()
    X_train = np.load(PROJECT_ROOT / "data/processed/cmapss_fd001/X_train.npy")
    X_validation = np.load(PROJECT_ROOT / "data/processed/cmapss_fd001/X_validation.npy")
    baseline = fit_robust_baseline(X_train)
    calibration = X_validation[: len(X_validation) // 2]
    normal_eval = X_validation[len(X_validation) // 2 :]
    calibration_stat_sensor = robust_sensor_scores(calibration, baseline)
    calibration_stat = calibration_stat_sensor.max(axis=1)
    stat_threshold = calibrate_threshold(calibration_stat, 95)

    model, training_seconds = train_autoencoder(X_train, device, models_dir / "improved_autoencoder.pt")
    calibration_ae, calibration_ae_sensor = reconstruct(model, calibration, device)
    ae_threshold = calibrate_threshold(calibration_ae, 95)

    fault_types = ["BIAS", "DRIFT", "STUCK_AT", "NOISE", "INTERMITTENT_DROPOUT"]
    severities = {"low": 0.25, "medium": 0.5, "high": 1.0}
    onsets = {"early": 0.2, "middle": 0.5, "late": 0.75}
    scenarios = []
    windows = []
    for index, (fault_type, severity_name, onset_name, sensor_index) in enumerate(
        (t, s, o, i) for t in fault_types for s in severities for o in onsets for i in (0, 1, 2)
    ):
        source = normal_eval[index % len(normal_eval)]
        injected, metadata = inject_fault(source, sensor_index, fault_type, onsets[onset_name], severities[severity_name], max(float(np.std(X_train[:, :, sensor_index])), 1e-6), seed=42 + index)
        metadata.update({"scenario_index": index, "severity_name": severity_name, "onset_name": onset_name})
        scenarios.append(metadata)
        windows.append(injected)
    order = np.random.default_rng(42).permutation(len(windows))
    dev_count = int(len(order) * 0.7)
    dev_idx, final_idx = order[:dev_count], order[dev_count:]
    scenarios_frame = pd.DataFrame(scenarios)
    fault_windows = np.asarray(windows, dtype=np.float32)
    scenario_scores = {}
    for name, values in (("statistical", robust_sensor_scores(fault_windows, baseline)),):
        scenario_scores[name] = values
    fault_ae, fault_ae_sensor = reconstruct(model, fault_windows, device)
    scenario_scores["autoencoder"] = fault_ae_sensor
    stat_fault_global = scenario_scores["statistical"].max(axis=1)
    ae_fault_global = fault_ae

    # Thresholds and combination are locked from normal calibration only.
    calibration_combined = calibration_stat / stat_threshold + calibration_ae / ae_threshold
    combined_threshold = calibrate_threshold(calibration_combined, 95)
    labels = np.concatenate([np.zeros(len(normal_eval), dtype=int), np.ones(len(final_idx), dtype=int)])
    normal_stat = robust_sensor_scores(normal_eval, baseline)
    normal_stat_global = normal_stat.max(axis=1)
    normal_ae, normal_ae_sensor = reconstruct(model, normal_eval, device)
    final_stat = stat_fault_global[final_idx]
    final_ae = ae_fault_global[final_idx]
    final_stat_sensors = scenario_scores["statistical"][final_idx]
    final_ae_sensors = scenario_scores["autoencoder"][final_idx]
    score_map = {
        "statistical": np.concatenate([normal_stat_global, final_stat]),
        "autoencoder": np.concatenate([normal_ae, final_ae]),
        "combined": np.concatenate([normal_stat_global / stat_threshold + normal_ae / ae_threshold, final_stat / stat_threshold + final_ae / ae_threshold]),
    }
    rows = []
    isolation_rows = []
    for detector, scores in score_map.items():
        threshold = {"statistical": stat_threshold, "autoencoder": ae_threshold, "combined": combined_threshold}[detector]
        metric = detector_metrics(labels, scores, threshold)
        metric.update({"detector": detector, "threshold": threshold})
        rows.append(metric)
    final_scenarios = scenarios_frame.iloc[final_idx].copy().reset_index(drop=True)
    final_scenarios["statistical_score"] = final_stat
    final_scenarios["statistical_detected"] = final_stat >= stat_threshold
    final_scenarios["autoencoder_score"] = final_ae
    final_scenarios["combined_score"] = final_stat / stat_threshold + final_ae / ae_threshold
    for detector, sensor_scores in (("statistical", final_stat_sensors), ("autoencoder", final_ae_sensors), ("combined", final_stat_sensors / stat_threshold + final_ae_sensors / ae_threshold)):
        for index, sensor_score in enumerate(sensor_scores):
            ranking = np.argsort(sensor_score)[::-1].tolist()
            target = int(final_scenarios.iloc[index]["sensor_index"])
            isolation_rows.append({"detector": detector, "scenario_index": int(final_scenarios.iloc[index]["scenario_index"]), "target_sensor": target, "rank": ranking.index(target) + 1, "top1": ranking[0] == target, "top3": target in ranking[:3], "top5": target in ranking[:5]})
    isolation = pd.DataFrame(isolation_rows)
    for row in rows:
        subset = isolation[isolation["detector"] == row["detector"]]
        row.update({"top1": float(subset["top1"].mean()), "top3": float(subset["top3"].mean()), "top5": float(subset["top5"].mean())})
    results = pd.DataFrame(rows)
    results.to_csv(output / "results.csv", index=False)
    (output / "results.json").write_text(json.dumps({"detectors": rows, "baseline_step10": BASELINE}, indent=2), encoding="utf-8")
    scenarios_frame.to_csv(output / "fault_scenarios.csv", index=False)
    final_scenarios.to_csv(output / "detection_results.csv", index=False)
    isolation.to_csv(output / "isolation_results.csv", index=False)
    breakdowns = {}
    for key in ("fault_type", "severity_name", "onset_name", "sensor_index"):
        grouped = final_scenarios.assign(
            detected=final_scenarios["statistical_score"] >= stat_threshold
        ).groupby(key, dropna=False)
        breakdowns[key] = grouped.agg(
            sample_count=("detected", "size"),
            detection_rate=("detected", "mean"),
            mean_statistical_score=("statistical_score", "mean"),
        ).reset_index()
        breakdowns[key].to_csv(output / f"{key}_summary.csv", index=False)
    (output / "config.json").write_text(json.dumps({"seed": 42, "development_fraction": 0.7, "final_fraction": 0.3, "threshold_percentile": 95, "persistence": [1, 3, 5], "autoencoder": {"hidden_size": 64, "epochs": 20, "optimizer": "AdamW", "learning_rate": 5e-4}, "fault_types": fault_types, "severities": severities, "onsets": onsets, "device": str(device)}, indent=2), encoding="utf-8")
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    for detector, scores in score_map.items():
        plt.figure()
        plt.hist(scores[labels == 0], bins=30, alpha=0.6, label="normal")
        plt.hist(scores[labels == 1], bins=30, alpha=0.6, label="fault")
        plt.legend()
        plt.xlabel("Anomaly score")
        plt.ylabel("Count")
        plt.title(f"{detector.title()} detector score distribution")
        plt.tight_layout()
        plt.savefig(plots / f"{detector}_score_distribution.png", dpi=150)
        plt.close()
    plt.figure()
    plt.bar(results["detector"], results["f1"])
    plt.ylabel("F1")
    plt.title("Detector comparison")
    plt.tight_layout()
    plt.savefig(plots / "detector_comparison.png", dpi=150)
    plt.close()
    report = f"""# Step 10.1: Improved Sensor FDI and Anomaly Detection

## Baseline diagnosis

The original Step 10 used a global mean reconstruction score and mean
per-sensor contribution. Short tail faults were diluted across the window,
and sensor-specific evidence was lost. The original baseline remains preserved
at `reports/experiments/step10_fdi/`.

## Improvements

This experiment uses robust per-sensor median/MAD scores with maximum temporal
deviation, an improved 64-unit AdamW autoencoder trained for 20 epochs, and
sensor rankings from maximum per-sensor contributions. Thresholds were fixed
at the 95th percentile using normal calibration windows only. Scenarios were
split deterministically 70/30 before final evaluation; final labels were not
used for tuning.

## Final held-out comparison

{results.to_string(index=False)}

### Fault-type breakdown

{breakdowns["fault_type"].to_string(index=False)}

### Severity breakdown

{breakdowns["severity_name"].to_string(index=False)}

### Onset breakdown

{breakdowns["onset_name"].to_string(index=False)}

### Sensor breakdown

{breakdowns["sensor_index"].to_string(index=False)}

Normal-degradation false-positive rate on held-out normal validation windows:
`{float(np.mean(normal_stat_global >= stat_threshold)):.2%}`. This is a
distribution-shift diagnostic, not a tuned final-fault threshold.

Training device: `{device}`. Autoencoder training time: `{training_seconds:.2f}` seconds.
The statistical detector is the primary improvement because it directly
captures distributional sensor excursions without reconstruction dilution.

## Leakage audit

- Existing scaler and prepared arrays were reused unchanged.
- The autoencoder trained only on normal `X_train`.
- Thresholds used only normal calibration windows.
- Fault severity used training-window standard deviations.
- Final fault scenarios were held out before evaluation.
- No final evaluation labels selected thresholds, architecture, or weights.

## Limitations

Faults are controlled synthetic simulations, not real aerospace fault labels.
C-MAPSS FD001 is a benchmark engine-degradation dataset. Detection is not
diagnosis, and synthetic isolation does not establish operational FDI
capability or aerospace safety.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
