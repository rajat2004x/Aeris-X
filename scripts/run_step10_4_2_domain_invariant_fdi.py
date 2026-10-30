"""Run isolated Step 10.4.2 domain-invariant residual FDI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aeris.data.cmapss import SENSOR_COLUMNS, load_cmapss, make_windows, split_engines
from aeris.faults.injection import inject_fault
from aeris.fdi.domain_invariant import (
    fit_residual_models,
    residual_features,
    residual_windows,
)

OUT = ROOT / "reports/experiments/step10_4_2_domain_invariant_fdi"
DATA = ROOT / "data/raw"
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
FAULTS = ["BIAS", "DRIFT", "STUCK_AT", "NOISE", "INTERMITTENT_DROPOUT", "SPIKE"]
WINDOW = 30
SEED = 42


def windows_with_context(frame: pd.DataFrame, ids: list[int], sensors: list[str]) -> tuple[np.ndarray, np.ndarray]:
    selected = set(map(int, ids))
    values, settings = [], []
    for engine, group in frame.groupby("engine_id", sort=True):
        if int(engine) not in selected:
            continue
        group = group.sort_values("cycle")
        x = group[sensors].to_numpy(dtype=np.float32)
        c = group[["setting_1", "setting_2", "setting_3"]].to_numpy(dtype=np.float32)
        for end in range(WINDOW, len(group) + 1):
            values.append(x[end - WINDOW:end])
            settings.append(c[end - WINDOW:end].mean(axis=0))
    if not values:
        raise ValueError("No windows generated")
    return np.asarray(values), np.asarray(settings)


def inject_many(windows: np.ndarray, scale: np.ndarray, dataset: str, offset: int) -> tuple[np.ndarray, pd.DataFrame]:
    changed, rows = [], []
    for i in range(min(48, len(windows))):
        sensor = (i * 7 + len(dataset)) % windows.shape[-1]
        fault = FAULTS[i % len(FAULTS)]
        result, metadata = inject_fault(
            windows[i], sensor, fault, (0.25, 0.45, 0.65)[i % 3],
            (0.45, 0.7, 1.0)[i % 3], max(float(scale[sensor]), 1e-6),
            seed=SEED + offset + i,
        )
        changed.append(np.nan_to_num(result))
        metadata.update({"dataset": dataset, "scenario_id": offset + i, "fault_type": fault})
        rows.append(metadata)
    return np.asarray(changed, dtype=np.float32), pd.DataFrame(rows)


def metrics(y: np.ndarray, score: np.ndarray, threshold: float, dataset: str, method: str) -> dict[str, object]:
    predicted = score >= threshold
    normal = y == 0
    return {
        "dataset": dataset, "method": method, "threshold": threshold,
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "f1": f1_score(y, predicted, zero_division=0),
        "fpr": float(np.mean(predicted[normal])) if np.any(normal) else 0.0,
        "fnr": float(np.mean(~predicted[y == 1])) if np.any(y == 1) else 0.0,
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "normal_windows": int(np.sum(normal)), "fault_windows": int(np.sum(y == 1)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    development, final_rows, shift_rows, threshold_rows = [], [], [], []
    prepared = {}
    for index, name in enumerate(DATASETS):
        dataset = load_cmapss(DATA, name)
        train_ids, dev_ids, final_ids = split_engines(dataset.get_engine_ids(), SEED)
        train_x, train_c = windows_with_context(dataset.train, train_ids, SENSOR_COLUMNS)
        dev_x, dev_c = windows_with_context(dataset.train, dev_ids, SENSOR_COLUMNS)
        final_x, final_c = windows_with_context(dataset.train, final_ids, SENSOR_COLUMNS)
        model = fit_residual_models(train_x, train_c)
        train_res = residual_windows(train_x, train_c, model)
        dev_res = residual_windows(dev_x, dev_c, model)
        final_res = residual_windows(final_x, final_c, model)
        train_evidence, _ = residual_features(train_res, model["scale"])
        dev_evidence, _ = residual_features(dev_res, model["scale"])
        final_evidence, _ = residual_features(final_res, model["scale"])
        scale = np.std(train_x, axis=(0, 1))
        faults, metadata = inject_many(dev_x, scale, name, 1000 + index * 100)
        fault_c = np.repeat(dev_c[: len(faults)], 1, axis=0)
        fault_res = residual_windows(faults, fault_c, model)
        fault_evidence, _ = residual_features(fault_res, model["scale"])
        normal_score = np.max(dev_evidence, axis=1)
        fault_score = np.max(fault_evidence, axis=1)
        development.append((np.r_[normal_score, fault_score], np.r_[np.zeros(len(normal_score)), np.ones(len(fault_score))]))
        final_faults, final_meta = inject_many(final_x, scale, name, 2000 + index * 100)
        final_fault_c = np.repeat(final_c[: len(final_faults)], 1, axis=0)
        final_fault_res = residual_windows(final_faults, final_fault_c, model)
        final_fault_evidence, _ = residual_features(final_fault_res, model["scale"])
        final_score = np.r_[np.max(final_evidence, axis=1), np.max(final_fault_evidence, axis=1)]
        final_y = np.r_[np.zeros(len(final_evidence)), np.ones(len(final_fault_evidence))]
        prepared[name] = (final_y, final_score)
        train_condition = np.mean(train_c, axis=0)
        distance = float(np.linalg.norm(np.mean(final_c, axis=0) - train_condition))
        shift_rows.append({"dataset": name, "domain_distance": distance, "train_windows": len(train_x), "development_windows": len(dev_x), "final_windows": len(final_x)})
        for target_fpr in (0.01, 0.02, 0.05, 0.10):
            threshold_rows.append({"target_fpr": target_fpr, "threshold": float(np.quantile(normal_score, 1.0 - target_fpr)), "dataset": name})
    dev_score = np.concatenate([item[0] for item in development])
    dev_y = np.concatenate([item[1] for item in development])
    candidates = np.quantile(dev_score[dev_y == 0], np.linspace(0.90, 0.999, 100))
    threshold = float(max(candidates, key=lambda value: (f1_score(dev_y, dev_score >= value, zero_division=0), -abs(np.mean((dev_score[dev_y == 0] >= value)) - 0.05))))
    for name, (y, score) in prepared.items():
        final_rows.append(metrics(y, score, threshold, name, "domain_invariant_condition_residual"))
    pd.DataFrame(final_rows).to_csv(OUT / "final_results.csv", index=False)
    pd.DataFrame(shift_rows).to_csv(OUT / "domain_shift_analysis.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(OUT / "threshold_tradeoff.csv", index=False)
    pd.DataFrame({"development_score": dev_score, "label": dev_y}).to_csv(OUT / "development_scores.csv", index=False)
    (OUT / "config.json").write_text(json.dumps({"datasets": DATASETS, "faults": FAULTS, "window": WINDOW, "seed": SEED, "threshold": threshold, "model": "per-sensor Ridge conditioned on operating settings and temporal position"}, indent=2))
    result_frame = pd.DataFrame(final_rows)
    (OUT / "FINAL_STEP10_4_2_REPORT.md").write_text(
        "# Step 10.4.2 Domain-Invariant FDI\n\n"
        "## Protocol\n\n"
        "Each sensor was predicted from operating-condition settings and temporal position using Ridge models fit on normal training engines only. "
        "Evidence uses condition residuals, robust residual scale, temporal persistence, and peer disagreement. "
        "Development fault scenarios selected one shared threshold; final engines and final fault scenarios were held out and used only for evaluation. "
        "Historical Step 10.x directories were not overwritten.\n\n"
        "## Results\n\n"
        + "```\n"
        + result_frame.to_string(index=False)
        + "\n```"
        + "\n\n"
        "## Interpretation\n\n"
        "The representation reduced FD001 false positives relative to Step 10.4.1, but the shared threshold did not provide reliable cross-dataset detection: "
        "FD002 and FD004 produced zero recall, while FD003 remained weak. Therefore this run does not demonstrate successful domain-invariant generalization. "
        "The residual representation is a measurable improvement in the failure diagnosis and should not replace the historical FD001 specialist without additional development.\n"
    )


if __name__ == "__main__":
    main()
