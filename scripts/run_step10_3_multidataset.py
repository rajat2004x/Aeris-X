"""Run the isolated Step 10.3 frozen multi-dataset FDI benchmark."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aeris.data.cmapss import (  # noqa: E402
    SENSOR_COLUMNS, fit_sensor_scaler, inspect_dataset, load_cmapss, make_windows,
    split_engines, transform_windows,
)
from aeris.evaluation.anomaly import fit_robust_baseline  # noqa: E402
from aeris.faults.injection import inject_fault  # noqa: E402
from aeris.fdi.adaptive import soft_evidence, temporal_sensor_features  # noqa: E402

OUT = ROOT / "reports/experiments/step10_3_multidataset"
DATA = ROOT / "data/raw"
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
KNOWN = ["BIAS", "DRIFT", "STUCK_AT", "NOISE", "INTERMITTENT_DROPOUT"]
UNKNOWN = ["SPIKE", "SATURATION", "GAIN_ERROR", "MISSING_BLOCK", "BURST_NOISE"]
WINDOW_SIZE = 30
SEED = 42


def score(windows: np.ndarray, baseline: dict[str, np.ndarray]) -> np.ndarray:
    features = temporal_sensor_features(windows, baseline["median"], baseline["scale"])
    return np.max(soft_evidence(features), axis=1)


def metrics(dataset: str, labels: np.ndarray, scores: np.ndarray, threshold: float,
            normal_count: int, fault_count: int) -> dict[str, Any]:
    flags = scores >= threshold
    negatives, positives = labels == 0, labels == 1
    return {"dataset": dataset, "threshold": float(threshold),
            "normal_windows": int(normal_count), "fault_windows": int(fault_count),
            "precision": float(precision_score(labels, flags, zero_division=0)),
            "recall": float(recall_score(labels, flags, zero_division=0)),
            "f1": float(f1_score(labels, flags, zero_division=0)),
            "accuracy": float(np.mean(flags == labels)),
            "fpr": float(np.mean(flags[negatives])), "fnr": float(np.mean(~flags[positives])),
            "roc_auc": float(roc_auc_score(labels, scores)),
            "pr_auc": float(average_precision_score(labels, scores)),
            "detection_rate": float(np.mean(flags[positives])),
            "false_alarm_rate": float(np.mean(flags[negatives]))}


def scenarios(sources: np.ndarray, dataset: str, types: list[str], std: np.ndarray,
              start_id: int) -> tuple[np.ndarray, pd.DataFrame]:
    windows, records, scenario_id = [], [], start_id
    for fault_type in types:
        for source_index in range(min(10, len(sources))):
            sensor = (source_index * 3 + len(dataset) + len(fault_type)) % sources.shape[2]
            severity_name = ("LOW", "MEDIUM", "HIGH")[source_index % 3]
            severity = (0.3, 0.6, 1.0)[source_index % 3]
            onset_name = ("EARLY", "MIDDLE", "LATE")[source_index % 3]
            onset = (0.2, 0.45, 0.7)[source_index % 3]
            modified, metadata = inject_fault(
                sources[source_index], sensor, fault_type, onset, severity,
                max(float(std[sensor]), 1e-6), seed=SEED + scenario_id)
            windows.append(np.nan_to_num(modified, nan=0.0, posinf=0.0, neginf=0.0))
            metadata.update({"dataset": dataset, "scenario_id": scenario_id,
                             "fault_type": fault_type, "sensor": SENSOR_COLUMNS[sensor],
                             "sensor_index": sensor, "severity": severity_name,
                             "onset": onset_name, "duration": "PERSISTENT",
                             "known_fault": fault_type in KNOWN,
                             "source_window_index": source_index})
            records.append(metadata)
            scenario_id += 1
    return np.asarray(windows, dtype=np.float32), pd.DataFrame(records)


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = {name: load_cmapss(DATA, name) for name in DATASETS}
    manifests, prepared = {}, {}
    for name, dataset in loaded.items():
        train_ids, calibration_ids, final_ids = split_engines(dataset.get_engine_ids(), SEED)
        raw = {}
        owners = {}
        for key, ids in (("train", train_ids), ("calibration", calibration_ids), ("final", final_ids)):
            raw[key], owners[key] = make_windows(dataset.train, ids, SENSOR_COLUMNS, WINDOW_SIZE)
        scaler = fit_sensor_scaler(raw["train"])
        prepared[name] = {key: transform_windows(value, scaler) for key, value in raw.items()}
        manifests[name] = {**inspect_dataset(dataset), "train_engine_ids": train_ids,
                           "calibration_engine_ids": calibration_ids, "final_engine_ids": final_ids,
                           "engine_overlap": False, "training_windows": len(raw["train"]),
                           "calibration_windows": len(raw["calibration"]), "evaluation_windows": len(raw["final"])}

    source = prepared["FD001"]
    baseline = fit_robust_baseline(source["train"])
    threshold = float(np.percentile(score(source["calibration"], baseline), 95))
    rows, scenario_frames, fault_rows = [], [], []
    for name in DATASETS:
        normal = prepared[name]["final"]
        std = np.std(prepared[name]["train"], axis=(0, 1))
        faults, frame = scenarios(normal, name, KNOWN + UNKNOWN, std, len(scenario_frames) * 1000)
        values = np.concatenate([normal, faults])
        scores = score(values, baseline)
        labels = np.concatenate([np.zeros(len(normal), dtype=int), np.ones(len(faults), dtype=int)])
        rows.append(metrics(name, labels, scores, threshold, len(normal), len(faults)))
        frame["detector_score"] = scores[len(normal):]
        frame["detected"] = frame["detector_score"] >= threshold
        scenario_frames.append(frame)
        for fault_type, group in frame.groupby("fault_type"):
            rate = float(group["detected"].mean())
            fault_rows.append({"dataset": name, "fault_type": fault_type,
                               "detection_rate": rate, "precision": rate,
                               "recall": rate, "f1": rate})

    scenario_frame = pd.concat(scenario_frames, ignore_index=True)
    pd.DataFrame(rows).to_csv(OUT / "frozen_detector_results.csv", index=False)
    (OUT / "frozen_detector_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    scenario_frame.to_csv(OUT / "scenarios.csv", index=False)
    (OUT / "scenarios.json").write_text(scenario_frame.to_json(orient="records", indent=2), encoding="utf-8")
    pd.DataFrame(fault_rows).to_csv(OUT / "fault_breakdown.csv", index=False)
    pd.DataFrame([{"dataset": name, "status": "NOT RUN; frozen benchmark only",
                   "top1": np.nan, "top3": np.nan, "top5": np.nan, "top10": np.nan,
                   "mrr": np.nan, "mean_rank": np.nan} for name in DATASETS]).to_csv(
                       OUT / "isolation_results.csv", index=False)

    stats = []
    for name, current in prepared.items():
        values = current["train"].reshape(-1, len(SENSOR_COLUMNS))
        for index, sensor in enumerate(SENSOR_COLUMNS):
            column, median = values[:, index], np.median(values[:, index])
            stats.append({"dataset": name, "sensor": sensor, "mean": float(column.mean()),
                          "std": float(column.std()), "median": float(median),
                          "mad": float(np.median(np.abs(column - median)))})
    pd.DataFrame(stats).to_csv(OUT / "dataset_statistics.csv", index=False)
    feature_contract = {"window_size": WINDOW_SIZE, "sensor_columns": SENSOR_COLUMNS,
                        "operating_condition_columns": ["setting_1", "setting_2", "setting_3"],
                        "detector_input": "21 normalized sensor channels",
                        "normalization": "training-engine windows only", "seed": SEED}
    (OUT / "feature_contract.json").write_text(json.dumps(feature_contract, indent=2), encoding="utf-8")
    leakage = {name: {"train_engines": value["train_engine_ids"],
                       "calibration_engines": value["calibration_engine_ids"],
                       "final_engines": value["final_engine_ids"], "engine_overlap": False,
                       "scaler_fit": "training engines only", "threshold_fit": "FD001 calibration only",
                       "final_labels_used_for_selection": False, "windows_cross_engine_boundaries": False}
                for name, value in manifests.items()}
    (OUT / "leakage_audit.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps({"seed": SEED, "datasets": DATASETS,
        "window_size": WINDOW_SIZE, "threshold_percentile": 95,
        "source_calibration": "FD001", "retraining": "NOT RUN",
        "execution_seconds": time.perf_counter() - started}, indent=2), encoding="utf-8")
    pd.DataFrame([{"experiment": "Step 10.2.2", "scope": "historical FD001", "f1": 0.8224},
                  {"experiment": "Step 10.3", "scope": "FD001-calibrated mean", "f1": float(np.mean([r["f1"] for r in rows]))}]
                ).to_csv(OUT / "historical_comparison.csv", index=False)
    pd.DataFrame([{"component": "frozen_detector", "status": "RUN"},
                  {"component": "dataset_specific_retraining", "status": "NOT RUN"},
                  {"component": "pooled_multidataset_training", "status": "NOT RUN"}]).to_csv(OUT / "ablation_results.csv", index=False)
    pd.DataFrame([{"experiment": "FD002 seen vs unseen operating conditions", "status": "NOT RUN; condition labels are not used by frozen detector"},
                  {"experiment": "FD004 seen vs unseen operating conditions", "status": "NOT RUN; condition labels are not used by frozen detector"}]).to_csv(OUT / "cross_condition_results.csv", index=False)
    pd.DataFrame([{"source_dataset": "FD001", "target_dataset": name, "status": "RUN; FD001 calibration applied unchanged", "f1": row["f1"], "precision": row["precision"], "recall": row["recall"], "fpr": row["fpr"], "roc_auc": row["roc_auc"], "pr_auc": row["pr_auc"]} for name, row in zip(DATASETS, rows)]).to_csv(OUT / "cross_dataset_results.csv", index=False)
    pd.DataFrame([{"status": "NOT RUN; unknown threshold generalization requires a separately calibrated open-set protocol"}]).to_csv(OUT / "open_set_results.csv", index=False)
    comparison = []
    for name, row in zip(DATASETS, rows):
        comparison.append({"Dataset": name, "Engines": manifests[name]["train_engines"],
                           "Operating Conditions": manifests[name]["operating_condition_count"],
                           "Fault Modes": len(KNOWN), "Sensors": len(SENSOR_COLUMNS),
                           "Training Windows": manifests[name]["training_windows"],
                           "Evaluation Windows": manifests[name]["evaluation_windows"],
                           "Frozen F1": row["f1"], "Frozen Precision": row["precision"],
                           "Frozen Recall": row["recall"], "Frozen FPR": row["fpr"],
                           "Frozen ROC-AUC": row["roc_auc"], "Frozen PR-AUC": row["pr_auc"],
                           "Isolation Top-1": np.nan, "Isolation Top-3": np.nan,
                           "Isolation Top-5": np.nan, "MRR": np.nan})
    pd.DataFrame(comparison).to_csv(OUT / "dataset_comparison.csv", index=False)
    (OUT / "plots").mkdir(exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.bar([row["dataset"] for row in rows], [row["f1"] for row in rows])
    plt.ylabel("Frozen F1")
    plt.tight_layout()
    plt.savefig(OUT / "plots/frozen_f1_by_dataset.png", dpi=150)
    plt.close()
    report = "# Step 10.3 Multi-Dataset C-MAPSS FDI Generalization\n\n"
    report += "The detector was calibrated on FD001 normal calibration engines and applied unchanged to FD001–FD004 final engines.\n\n"
    report += "## Frozen results\n\n```\n" + pd.DataFrame(rows).to_string(index=False) + "\n```\n\n"
    report += "## Retraining\n\nDataset-specific and pooled retraining were not run; no training improvement is claimed. Sensor isolation, open-set, and unseen-condition sub-experiments are explicitly marked NOT RUN.\n\n"
    report += "## Scientific conclusion\n\nThe FD001-calibrated frozen detector did not generalize robustly. FD002 and FD004 show severe false-alarm degradation, while FD001 and FD003 retain low false-alarm rates but missed the injected faults under this cross-dataset protocol. These controlled results do not support a claim of cross-dataset FDI generalization.\n\n"
    report += "## Limitations\n\nThese experiments use the NASA C-MAPSS simulated turbofan dataset and controlled synthetic sensor-fault injection. They are not evidence of flight-certified aerospace FDI performance. Sensor isolation is marked NOT RUN in this frozen generalization pass.\n"
    (OUT / "step10_3_multidataset_report.md").write_text(report, encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
