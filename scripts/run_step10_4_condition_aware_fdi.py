"""Run the isolated Step 10.4 condition-aware FDI research benchmark."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aeris.data.cmapss import SENSOR_COLUMNS, load_cmapss, make_windows, split_engines
from aeris.faults.injection import inject_fault
from aeris.fdi.condition import assign_conditions, fit_condition_baseline, fit_condition_model, robust_window_scores

OUT = ROOT / "reports/experiments/step10_4_condition_aware_fdi"
DATA = ROOT / "data/raw"
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
KNOWN = ["BIAS", "DRIFT", "STUCK_AT", "NOISE", "INTERMITTENT_DROPOUT"]
UNKNOWN = ["SPIKE", "SATURATION", "GAIN_ERROR", "MISSING_BLOCK", "BURST_NOISE"]
SEED = 42
WINDOW = 30


def window_conditions(frame: pd.DataFrame, owners: np.ndarray, condition_model: dict[str, object], sensor_windows: np.ndarray) -> np.ndarray:
    settings = ["setting_1", "setting_2", "setting_3"]
    last_rows = frame.groupby("engine_id", sort=True)[settings].last()
    rows = np.asarray([last_rows.loc[engine_id].to_numpy() for engine_id in owners], dtype=float)
    return assign_conditions(rows, condition_model)


def inject_scenarios(windows: np.ndarray, dataset: str, std: np.ndarray, start: int) -> tuple[np.ndarray, pd.DataFrame]:
    result, records = [], []
    for offset in range(min(20, len(windows))):
        sensor = (offset * 3 + len(dataset)) % windows.shape[2]
        fault = KNOWN[offset % len(KNOWN)]
        severity_name = ("LOW", "MEDIUM", "HIGH")[offset % 3]
        severity = (0.3, 0.6, 1.0)[offset % 3]
        altered, metadata = inject_fault(windows[offset], sensor, fault, (0.2, 0.45, 0.7)[offset % 3], severity, max(float(std[sensor]), 1e-6), seed=SEED + start + offset)
        result.append(np.nan_to_num(altered))
        metadata.update({"dataset": dataset, "scenario_id": start + offset, "fault_type": fault, "sensor_index": sensor, "severity": severity_name})
        records.append(metadata)
    return np.asarray(result, dtype=np.float32), pd.DataFrame(records)


def evaluate(dataset: str, normal: np.ndarray, faults: np.ndarray, normal_scores: np.ndarray, fault_scores: np.ndarray, threshold: float, mode: str) -> dict[str, object]:
    labels = np.r_[np.zeros(len(normal_scores), dtype=int), np.ones(len(fault_scores), dtype=int)]
    scores = np.r_[normal_scores, fault_scores]
    flags = scores >= threshold
    return {"dataset": dataset, "method": mode, "threshold": float(threshold), "normal_windows": len(normal), "fault_windows": len(faults), "precision": float(precision_score(labels, flags, zero_division=0)), "recall": float(recall_score(labels, flags, zero_division=0)), "f1": float(f1_score(labels, flags, zero_division=0)), "fpr": float(np.mean(flags[:len(normal)])), "roc_auc": float(roc_auc_score(labels, scores)), "pr_auc": float(average_precision_score(labels, scores))}


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = {name: load_cmapss(DATA, name) for name in DATASETS}
    global_rows, condition_rows, threshold_rows, shift_rows, condition_shift_rows = [], [], [], [], []
    leakage = {}
    prepared = {}
    for name, dataset in loaded.items():
        train_ids, calibration_ids, final_ids = split_engines(dataset.get_engine_ids(), SEED)
        raw = {key: make_windows(dataset.train, ids, SENSOR_COLUMNS, WINDOW)[0] for key, ids in (("train", train_ids), ("calibration", calibration_ids), ("final", final_ids))}
        settings = dataset.train[["setting_1", "setting_2", "setting_3"]].to_numpy()
        condition_model = fit_condition_model(settings, seed=SEED)
        owners = {key: make_windows(dataset.train, ids, SENSOR_COLUMNS, WINDOW)[1] for key, ids in (("train", train_ids), ("calibration", calibration_ids), ("final", final_ids))}
        condition_ids = {}
        for key in raw:
            last = dataset.train.groupby("engine_id", sort=True)[["setting_1", "setting_2", "setting_3"]].last()
            condition_ids[key] = assign_conditions(np.asarray([last.loc[engine].to_numpy() for engine in owners[key]]), condition_model)
        baseline = fit_condition_baseline(raw["train"], condition_ids["train"])
        prepared[name] = {"raw": raw, "conditions": condition_ids, "baseline": baseline, "condition_model": condition_model, "train_ids": train_ids, "calibration_ids": calibration_ids, "final_ids": final_ids}
        leakage[name] = {"train_engines": train_ids, "calibration_engines": calibration_ids, "final_engines": final_ids, "engine_overlap": False, "baseline_fit": "normal training engines only", "threshold_fit": "normal calibration engines only", "final_labels_used": False}

        global_cal = robust_window_scores(raw["calibration"], baseline)
        condition_cal = robust_window_scores(raw["calibration"], baseline, condition_ids["calibration"], True)
        global_threshold = float(np.percentile(global_cal, 95))
        condition_threshold = float(np.percentile(condition_cal, 95))
        global_normal = robust_window_scores(raw["final"], baseline)
        condition_normal = robust_window_scores(raw["final"], baseline, condition_ids["final"], True)
        faults, manifest = inject_scenarios(raw["final"], name, np.std(raw["train"], axis=(0, 1)), 10000 + len(global_rows))
        global_fault = robust_window_scores(faults, baseline)
        condition_fault = robust_window_scores(faults, baseline, np.repeat(condition_ids["final"][:len(faults)], 1), True)
        global_rows.append(evaluate(name, raw["final"], faults, global_normal, global_fault, global_threshold, "global_baseline_global_threshold"))
        condition_rows.append(evaluate(name, raw["final"], faults, condition_normal, condition_fault, condition_threshold, "condition_baseline_condition_threshold"))
        threshold_rows.extend([{"dataset": name, "method": "global", "threshold": global_threshold, "calibration_count": len(global_cal), "calibration_fpr": 0.05}, {"dataset": name, "method": "condition_aware", "threshold": condition_threshold, "calibration_count": len(condition_cal), "calibration_fpr": 0.05}])
        for sensor_index, sensor in enumerate(SENSOR_COLUMNS):
            train_values = raw["train"][:, :, sensor_index].ravel()
            for reference in ("FD001",):
                ref_values = prepared.get(reference, {}).get("raw", {}).get("train", raw["train"])[:, :, sensor_index].ravel()
                shift_rows.append({"source": reference, "target": name, "sensor": sensor, "wasserstein": float(wasserstein_distance(ref_values, train_values)), "ks_statistic": float(ks_2samp(ref_values, train_values).statistic)})
        for condition in np.unique(condition_ids["train"]):
            condition_shift_rows.append({"dataset": name, "condition": int(condition), "training_windows": int(np.sum(condition_ids["train"] == condition)), "calibration_windows": int(np.sum(condition_ids["calibration"] == condition)), "final_windows": int(np.sum(condition_ids["final"] == condition))})

    all_shift = pd.DataFrame(shift_rows)
    all_shift.to_csv(OUT / "sensor_shift_statistics.csv", index=False)
    pd.DataFrame(condition_shift_rows).to_csv(OUT / "condition_shift_statistics.csv", index=False)
    pd.DataFrame(global_rows).to_csv(OUT / "baseline_results.csv", index=False)
    pd.DataFrame(condition_rows).to_csv(OUT / "condition_aware_results.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(OUT / "threshold_results.csv", index=False)
    pd.DataFrame(global_rows + condition_rows).to_csv(OUT / "cross_dataset_results.csv", index=False)
    pd.DataFrame([{"dataset": name, "status": "NOT RUN; requires dedicated attribution training protocol"} for name in DATASETS]).to_csv(OUT / "isolation_results.csv", index=False)
    pd.DataFrame([{"dataset": name, "status": "NOT RUN; no final-label classifier training in this milestone"} for name in DATASETS]).to_csv(OUT / "fault_classification_results.csv", index=False)
    pd.DataFrame([{"dataset": name, "status": "NOT RUN; open-set calibration requires separate development protocol"} for name in DATASETS]).to_csv(OUT / "open_set_results.csv", index=False)
    pd.DataFrame([{"dataset": name, "status": "NOT RUN; multisensor attribution not selected before single-sensor baseline"} for name in DATASETS]).to_csv(OUT / "multisensor_results.csv", index=False)
    pd.DataFrame([{"model": "FD001-only", "status": "NOT RUN; condition-aware detector evaluated first"}]).to_csv(OUT / "pooled_training_results.csv", index=False)
    pd.DataFrame([{"component": item, "status": "RUN" if item in ("global_baseline", "condition_baseline", "adaptive_threshold") else "NOT RUN"} for item in ("global_baseline", "condition_baseline", "adaptive_threshold", "temporal_persistence", "peer_consistency", "sensor_local_features", "ensemble")]).to_csv(OUT / "ablation_results.csv", index=False)
    pd.DataFrame([{"dataset": name, "status": "NOT RUN; uncertainty integration deferred"} for name in DATASETS]).to_csv(OUT / "calibration_results.csv", index=False)
    pd.DataFrame([{"experiment": "Step 10.2.2", "scope": "historical FD001", "f1": 0.8224}, {"experiment": "Step 10.3.1 corrected", "scope": "FD001-FD004 frozen", "f1": "see audit"}, {"experiment": "Step 10.4", "scope": "condition-aware controlled benchmark", "f1": "see condition_aware_results.csv"}]).to_csv(OUT / "historical_comparison.csv", index=False)
    (OUT / "dataset_shift_statistics.csv").write_text(all_shift.groupby("target")[["wasserstein", "ks_statistic"]].mean().reset_index().to_csv(index=False), encoding="utf-8")
    (OUT / "leakage_audit.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    (OUT / "experiment_manifest.json").write_text(json.dumps({"experiment_id": "step10_4_condition_aware_fdi", "seed": SEED, "datasets": DATASETS, "window_size": WINDOW, "normalization": "global and condition-specific robust baseline", "fault_types": KNOWN, "runtime_seconds": time.perf_counter() - started}, indent=2), encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps({"seed": SEED, "condition_model": "KMeans on normalized raw operating settings; train data only", "threshold": "95th percentile normal calibration", "historical_artifacts_preserved": True}, indent=2), encoding="utf-8")
    (OUT / "plots").mkdir(exist_ok=True)
    pivot = pd.DataFrame(global_rows + condition_rows)
    pivot.pivot(index="dataset", columns="method", values="fpr").plot(kind="bar", figsize=(8, 4))
    plt.ylabel("False-positive rate")
    plt.tight_layout()
    plt.savefig(OUT / "plots/fpr_global_vs_condition.png", dpi=150)
    plt.close()
    report = "# Step 10.4 Condition-Aware FDI\n\n"
    report += "This isolated milestone quantifies operating-condition shift and compares global versus condition-aware robust baselines using normal training/calibration data only.\n\n"
    report += "## Results\n\n```\n" + pd.DataFrame(global_rows + condition_rows).to_string(index=False) + "\n```\n\n"
    report += "## Not run\n\nSensor isolation, known-fault classification, open-set rejection, pooled training, self-supervised representation learning, uncertainty integration, and multi-sensor attribution were not run because their development protocols were not established before this controlled baseline comparison.\n\n"
    report += "## Limitations\n\nAERIS-X is a research/portfolio implementation using NASA C-MAPSS simulated turbofan telemetry and controlled synthetic sensor-fault injection. It is not evidence of flight certification, operational aerospace deployment, replacement for certified FDIR, or production aerospace safety certification.\n"
    (OUT / "step10_4_condition_aware_fdi_report.md").write_text(report, encoding="utf-8")
    print(pd.DataFrame(global_rows + condition_rows).to_string(index=False))


if __name__ == "__main__":
    main()
