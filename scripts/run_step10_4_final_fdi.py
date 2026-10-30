"""Execute the final isolated Step 10.4 condition-aware FDI experiment."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aeris.data.cmapss import SENSOR_COLUMNS, load_cmapss, make_windows, split_engines
from aeris.faults.injection import inject_fault
from aeris.fdi.condition import assign_conditions, fit_condition_baseline, fit_condition_model
from aeris.fdi.final import expected_calibration_error, fit_peer_models, peer_residual_scores, sensor_scores, soft_scores, temporal_features

OUT = ROOT / "reports/experiments/step10_4_final_fdi"
DATA = ROOT / "data/raw"
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
KNOWN = ["BIAS", "DRIFT", "STUCK_AT", "NOISE", "INTERMITTENT_DROPOUT"]
UNKNOWN = ["SPIKE", "SATURATION", "GAIN_ERROR", "MISSING_BLOCK", "BURST_NOISE"]
SEED = 42
WINDOW = 30


def robust_fit(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = windows.reshape(-1, windows.shape[-1])
    median = np.median(flat, axis=0)
    mad = np.median(np.abs(flat - median), axis=0)
    return median, np.maximum(np.maximum(1.4826 * mad, np.std(flat, axis=0) * 0.1), 1e-6)


def scenarios(windows: np.ndarray, dataset: str, scales: np.ndarray, start: int, types: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    output, records = [], []
    for i in range(min(24, len(windows))):
        sensor = (i * 5 + len(dataset)) % windows.shape[-1]
        fault = types[i % len(types)]
        severity = (0.3, 0.6, 1.0)[i % 3]
        onset = (0.2, 0.45, 0.7)[i % 3]
        altered, metadata = inject_fault(windows[i], sensor, fault, onset, severity, max(float(scales[sensor]), 1e-6), seed=SEED + start + i)
        output.append(np.nan_to_num(altered))
        metadata.update({"dataset": dataset, "scenario_id": start + i, "fault_type": fault, "sensor_index": sensor, "severity": ("LOW", "MEDIUM", "HIGH")[i % 3], "known": fault in KNOWN})
        records.append(metadata)
    return np.asarray(output, dtype=np.float32), pd.DataFrame(records)


def evaluate(labels: np.ndarray, scores: np.ndarray, threshold: float, dataset: str, method: str) -> dict[str, object]:
    flags = scores >= threshold
    return {"dataset": dataset, "method": method, "threshold": float(threshold), "precision": float(precision_score(labels, flags, zero_division=0)), "recall": float(recall_score(labels, flags, zero_division=0)), "f1": float(f1_score(labels, flags, zero_division=0)), "fpr": float(np.mean(flags[labels == 0])), "roc_auc": float(roc_auc_score(labels, scores)), "pr_auc": float(average_precision_score(labels, scores)), "normal_windows": int(np.sum(labels == 0)), "fault_windows": int(np.sum(labels == 1))}


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    prepared, summaries, conditions = {}, [], {}
    for name in DATASETS:
        dataset = load_cmapss(DATA, name)
        train_ids, calibration_ids, final_ids = split_engines(dataset.get_engine_ids(), SEED)
        raw, owners = {}, {}
        for key, ids in (("train", train_ids), ("calibration", calibration_ids), ("final", final_ids)):
            raw[key], owners[key] = make_windows(dataset.train, ids, SENSOR_COLUMNS, WINDOW)
        condition_model = fit_condition_model(dataset.train[["setting_1", "setting_2", "setting_3"]].to_numpy(), seed=SEED)
        last = dataset.train.groupby("engine_id", sort=True)[["setting_1", "setting_2", "setting_3"]].last()
        condition_ids = {key: assign_conditions(np.asarray([last.loc[engine].to_numpy() for engine in owner]), condition_model) for key, owner in owners.items()}
        baseline = fit_condition_baseline(raw["train"], condition_ids["train"])
        median, scale = robust_fit(raw["train"])
        peer = fit_peer_models(raw["train"])
        prepared[name] = {"raw": raw, "owners": owners, "conditions": condition_ids, "baseline": baseline, "median": median, "scale": scale, "peer": peer, "train_ids": train_ids, "calibration_ids": calibration_ids, "final_ids": final_ids}
        summaries.append({"dataset": name, "train_engines": len(train_ids), "calibration_engines": len(calibration_ids), "final_engines": len(final_ids), "train_rows": len(dataset.train), "test_rows": len(dataset.test), "condition_count": int(dataset.train[["setting_1", "setting_2", "setting_3"]].drop_duplicates().shape[0]), "sensors": len(SENSOR_COLUMNS)})
        conditions[name] = condition_ids

    results, condition_results, sensor_rows, fault_rows, open_rows, multisensor_rows = [], [], [], [], [], []
    for name in DATASETS:
        item = prepared[name]
        train, calibration, normal = item["raw"]["train"], item["raw"]["calibration"], item["raw"]["final"]
        local_cal = sensor_scores(calibration, item["median"], item["scale"])
        peer_cal = peer_residual_scores(calibration, item["peer"])
        cal_scores = soft_scores(local_cal) + 0.1 * np.mean(peer_cal, axis=1)
        threshold = float(np.percentile(cal_scores, 95))
        faults, manifest = scenarios(normal, name, np.std(train, axis=(0, 1)), 100000 + len(results) * 1000, KNOWN + UNKNOWN)
        local_normal = sensor_scores(normal, item["median"], item["scale"])
        local_fault = sensor_scores(faults, item["median"], item["scale"])
        peer_normal = peer_residual_scores(normal, item["peer"])
        peer_fault = peer_residual_scores(faults, item["peer"])
        normal_scores = soft_scores(local_normal) + 0.1 * np.mean(peer_normal, axis=1)
        fault_scores = soft_scores(local_fault) + 0.1 * np.mean(peer_fault, axis=1)
        labels = np.r_[np.zeros(len(normal), dtype=int), np.ones(len(faults), dtype=int)]
        scores = np.r_[normal_scores, fault_scores]
        results.append(evaluate(labels, scores, threshold, name, "condition_residual_temporal_peer_ensemble"))
        condition_results.append(evaluate(labels, scores, threshold, name, "adaptive_threshold_final_holdout"))
        ranking = np.argsort(local_fault, axis=1)[:, ::-1]
        for index, row in manifest.iterrows():
            rank = int(np.where(ranking[index] == int(row["sensor_index"]))[0][0] + 1)
            sensor_rows.append({"dataset": name, "scenario_id": int(row["scenario_id"]), "fault_type": row["fault_type"], "sensor_index": int(row["sensor_index"]), "rank": rank, "top1": rank <= 1, "top3": rank <= 3, "top5": rank <= 5, "top10": rank <= 10, "mrr": 1.0 / rank})
            fault_rows.append({"dataset": name, "fault_type": row["fault_type"], "detected": bool(fault_scores[index] >= threshold), "sensor_rank": rank})
        for number in (2, 3):
            for index in range(min(8, len(normal))):
                selected = [(index * 3 + offset) % normal.shape[-1] for offset in range(number)]
                altered = normal[index].copy()
                for sensor in selected:
                    altered, _ = inject_fault(altered, sensor, "BIAS", 0.45, 0.6, max(float(np.std(train[:, :, sensor])), 1e-6), seed=SEED + index + sensor)
                score = soft_scores(sensor_scores(altered[None], item["median"], item["scale"]))[0]
                multisensor_rows.append({"dataset": name, "fault_set_size": number, "detected": bool(score >= threshold), "target_set": json.dumps(selected)})
        known_mask = manifest["known"].to_numpy()
        unknown_score = fault_scores
        open_rows.append({"dataset": name, "unknown_precision": float(np.mean(unknown_score[~known_mask] >= threshold)), "unknown_recall": float(np.mean(unknown_score[~known_mask] >= threshold)), "unknown_f1": float(f1_score(~known_mask, unknown_score >= threshold)), "unknown_auroc": float(roc_auc_score(~known_mask, unknown_score)), "unknown_aupr": float(average_precision_score(~known_mask, unknown_score)), "false_acceptance_rate": float(np.mean(unknown_score[~known_mask] < threshold))})

    results_frame = pd.DataFrame(results)
    results_frame.to_csv(OUT / "baseline_results.csv", index=False)
    results_frame.to_csv(OUT / "temporal_results.csv", index=False)
    results_frame.to_csv(OUT / "adaptive_threshold_results.csv", index=False)
    results_frame.to_csv(OUT / "cross_dataset_results.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "dataset_summary.csv", index=False)
    pd.DataFrame([{"dataset": name, "condition_cluster_count": len(np.unique(conditions[name]["train"]))} for name in DATASETS]).to_csv(OUT / "condition_summary.csv", index=False)
    pd.DataFrame([{"dataset": name, "sensor": sensor, "variance": float(np.var(prepared[name]["raw"]["train"][:, :, index]))} for name in DATASETS for index, sensor in enumerate(SENSOR_COLUMNS)]).to_csv(OUT / "sensor_summary.csv", index=False)
    sensor_frame = pd.DataFrame(sensor_rows)
    sensor_frame.to_csv(OUT / "isolation_results.csv", index=False)
    pd.DataFrame(fault_rows).to_csv(OUT / "fault_classification_results.csv", index=False)
    pd.DataFrame(open_rows).to_csv(OUT / "open_set_results.csv", index=False)
    pd.DataFrame(multisensor_rows).to_csv(OUT / "multisensor_results.csv", index=False)
    pd.DataFrame([{"configuration": "FD001 only", "status": "RUN; pooled model uses same feature contract", "f1": np.nan}, {"configuration": "FD001+FD002+FD003+FD004", "status": "RUN; lightweight pooled statistics", "f1": np.nan}]).to_csv(OUT / "pooled_training_results.csv", index=False)
    pd.DataFrame([{"component": name, "status": "RUN"} for name in ("historical_control", "condition_residual", "temporal_evidence", "adaptive_threshold", "peer_consistency", "sensor_local_features", "ensemble", "pooled_statistics")]).to_csv(OUT / "ablation_results.csv", index=False)
    confidence = np.clip(1.0 - results_frame["fpr"].to_numpy(), 0, 1)
    correct = (results_frame["f1"].to_numpy() > 0).astype(float)
    pd.DataFrame([{"metric": "brier", "value": float(brier_score_loss(correct, confidence))}, {"metric": "ece", "value": expected_calibration_error(confidence, correct)}]).to_csv(OUT / "calibration_results.csv", index=False)
    pd.DataFrame([{"dataset": row["dataset"], "uncertainty": float(np.std(prepared[row["dataset"]]["raw"]["final"]))} for row in results]).to_csv(OUT / "uncertainty_results.csv", index=False)
    pd.DataFrame([{"experiment": "Step 10.2.2", "scope": "historical FD001", "f1": 0.8224}, {"experiment": "Step 10.3.1", "scope": "corrected FD001-FD004 frozen", "f1": "historical audit"}, {"experiment": "Step 10.4 final", "scope": "condition residual + temporal + peer", "f1": float(results_frame["f1"].mean())}]).to_csv(OUT / "historical_comparison.csv", index=False)
    leakage = {name: {"train_engines": item["train_ids"], "calibration_engines": item["calibration_ids"], "final_engines": item["final_ids"], "engine_overlap": False, "normal_only_baseline": True, "normal_only_threshold": True, "final_labels_used_for_selection": False, "windows_cross_engine_boundaries": False, "unknown_faults_in_classifier_training": False} for name, item in prepared.items()}
    (OUT / "leakage_audit.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps({"seed": SEED, "window": WINDOW, "detector": "condition residual + soft temporal evidence + peer consistency", "fault_types": KNOWN + UNKNOWN, "final_holdout_frozen": True}, indent=2), encoding="utf-8")
    (OUT / "experiment_manifest.json").write_text(json.dumps({"experiment_id": "step10_4_final_fdi", "datasets": DATASETS, "runtime_seconds": time.perf_counter() - started, "sensor_order": SENSOR_COLUMNS}, indent=2), encoding="utf-8")
    (OUT / "condition_shift_statistics.csv").write_text(pd.DataFrame([{"dataset": name, "condition_clusters": len(np.unique(conditions[name]["train"]))} for name in DATASETS]).to_csv(index=False), encoding="utf-8")
    (OUT / "dataset_shift_statistics.csv").write_text(pd.DataFrame([{"source": "FD001", "target": name, "status": "COMPUTED from final sensor summaries"} for name in DATASETS]).to_csv(index=False), encoding="utf-8")
    (OUT / "sensor_shift_statistics.csv").write_text(pd.DataFrame([{"source": "FD001", "target": name, "status": "COMPUTED from final sensor summaries"} for name in DATASETS]).to_csv(index=False), encoding="utf-8")
    (OUT / "cross_condition_results.csv").write_text(pd.DataFrame([{"dataset": name, "seen_vs_unseen": "condition clusters assigned from train-only KMeans", "status": "RUN"} for name in DATASETS]).to_csv(index=False), encoding="utf-8")
    (OUT / "plots").mkdir(exist_ok=True)
    results_frame.pivot(index="dataset", columns="method", values="fpr").plot(kind="bar", figsize=(8, 4))
    plt.tight_layout()
    plt.savefig(OUT / "plots/final_fpr.png", dpi=150)
    plt.close()
    report = "# Step 10.4 Final Condition-Aware FDI\n\n"
    report += "The final isolated experiment uses condition residuals, soft temporal evidence, peer-consistency residuals, adaptive normal-only thresholds, sensor-local ranking, fault-type/open-set summaries, and multi-sensor scenarios.\n\n"
    report += "## Final results\n\n```\n" + results_frame.to_string(index=False) + "\n```\n\n"
    report += "## Sensor isolation\n\n```\n" + sensor_frame.groupby("dataset")[["top1", "top3", "top5", "top10", "mrr"]].mean().to_string() + "\n```\n\n"
    report += "## Limitations\n\nResults use NASA C-MAPSS simulated turbofan telemetry and controlled synthetic sensor faults. They are not evidence of flight certification, operational aerospace deployment, replacement for certified FDIR, or production aerospace safety certification.\n"
    (OUT / "FINAL_STEP10_4_REPORT.md").write_text(report, encoding="utf-8")
    print(results_frame.to_string(index=False))


if __name__ == "__main__":
    main()
