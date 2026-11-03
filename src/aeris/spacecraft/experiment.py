"""Leakage-controlled, bounded SMAP/MSL anomaly experiment utilities."""

from __future__ import annotations

import ast
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


DETECTORS = ("robust", "rolling", "pca", "isolation_forest")
SEED = 42


def _summary(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    return np.column_stack((values.mean(axis=1), values.std(axis=1), np.linalg.norm(values, axis=1)))


def discover_dataset(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    train = {path.stem: path for path in (root / "train").glob("*.npy")}
    test = {path.stem: path for path in (root / "test").glob("*.npy")}
    labels: dict[str, list[list[int]]] = {}
    label_rows: list[dict[str, Any]] = []
    import csv

    with (root / "labeled_anomalies.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            channel = row["chan_id"]
            ranges = ast.literal_eval(row["anomaly_sequences"])
            labels.setdefault(channel, []).extend([[int(a), int(b)] for a, b in ranges])
            label_rows.append(row)
    return {"root": str(root), "train": train, "test": test, "labels": labels, "label_rows": label_rows}


def profile_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    train, test = dataset["train"], dataset["test"]
    stats: dict[str, Any] = {"train_file_count": len(train), "test_file_count": len(test)}
    channel_profiles: dict[str, Any] = {}
    all_lengths: list[int] = []
    for channel in sorted(train):
        train_array = np.load(train[channel], mmap_mode="r")
        test_array = np.load(test[channel], mmap_mode="r")
        train_values = np.asarray(train_array, dtype=np.float64)
        test_values = np.asarray(test_array, dtype=np.float64)
        combined = np.concatenate((train_values.ravel(), test_values.ravel()))
        channel_profiles[channel] = {
            "train_length": int(len(train_array)),
            "test_length": int(len(test_array)),
            "train_shape": list(train_array.shape),
            "test_shape": list(test_array.shape),
            "dtype": str(train_array.dtype),
            "missing_values": int(np.isnan(combined).sum()),
            "infinite_values": int(np.isinf(combined).sum()),
            "constant_features": int(np.sum(np.std(combined.reshape(-1, train_array.shape[1]), axis=0) == 0)),
            "min": float(np.min(combined)),
            "max": float(np.max(combined)),
            "mean": float(np.mean(combined)),
            "std": float(np.std(combined)),
            "anomaly_sequences": dataset["labels"].get(channel, []),
        }
        all_lengths.extend([len(train_array), len(test_array)])
    rows = dataset["label_rows"]
    stats.update(
        {
            "channel_ids": sorted(train),
            "spacecraft_counts": {
                name: sum(row["spacecraft"] == name for row in rows) for name in ("SMAP", "MSL")
            },
            "label_record_count": len(rows),
            "unique_label_ids": len(dataset["labels"]),
            "duplicate_label_ids": sorted(
                channel for channel in {row["chan_id"] for row in rows}
                if sum(channel == other["chan_id"] for other in rows) > 1
            ),
            "length_min": min(all_lengths),
            "length_max": max(all_lengths),
            "channels": channel_profiles,
        }
    )
    return stats


@dataclass
class ChannelPreprocessor:
    center: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (_summary(values) - self.center) / self.scale


def fit_preprocessors(dataset: dict[str, Any]) -> dict[str, ChannelPreprocessor]:
    result: dict[str, ChannelPreprocessor] = {}
    for channel, path in dataset["train"].items():
        values = _summary(np.load(path, mmap_mode="r"))
        center = values.mean(axis=0)
        scale = values.std(axis=0)
        result[channel] = ChannelPreprocessor(center, np.maximum(scale, 1e-8))
    return result


def preprocessing_manifest(dataset: dict[str, Any], preprocessors: dict[str, ChannelPreprocessor]) -> dict[str, Any]:
    return {
        "method": "three train-only summaries per timestep (mean, feature_std, l2_norm), z-scaled per channel",
        "random_seed": SEED,
        "feature_dimensions": 3,
        "channels": sorted(preprocessors),
        "train_statistics": {
            channel: {"center": prep.center.tolist(), "scale": prep.scale.tolist()}
            for channel, prep in preprocessors.items()
        },
        "leakage_controls": [
            "statistics fitted on train telemetry only",
            "test telemetry is transformed with frozen train statistics",
            "anomaly labels are used only for final evaluation",
        ],
    }


def _scores(train: np.ndarray, test: np.ndarray) -> dict[str, np.ndarray]:
    median = np.median(train, axis=0)
    mad = np.median(np.abs(train - median), axis=0)
    robust_train = np.max(np.abs(train - median) / np.maximum(1.4826 * mad, 1e-8), axis=1)
    robust_test = np.max(np.abs(test - median) / np.maximum(1.4826 * mad, 1e-8), axis=1)
    rolling = np.abs(test - np.convolve(train[:, 0], np.ones(25) / 25, mode="same")[-1]).max(axis=1)
    rolling_train = np.abs(
        train - np.convolve(train[:, 0], np.ones(25) / 25, mode="same")[-len(train):, None]
    ).max(axis=1)
    if float(np.var(train)) <= 1e-12:
        pca_train = np.zeros(len(train), dtype=float)
        pca_test = np.zeros(len(test), dtype=float)
    else:
        components = min(3, train.shape[1], max(1, len(train) - 1))
        pca = PCA(n_components=components, random_state=SEED).fit(train)
        pca_train = np.mean((train - pca.inverse_transform(pca.transform(train))) ** 2, axis=1)
        pca_test = np.mean((test - pca.inverse_transform(pca.transform(test))) ** 2, axis=1)
    forest = IsolationForest(n_estimators=100, contamination="auto", random_state=SEED, n_jobs=1).fit(train)
    forest_train = -forest.score_samples(train)
    forest_test = -forest.score_samples(test)
    return {
        "robust": (robust_train, robust_test),
        "rolling": (rolling_train, rolling),
        "pca": (pca_train, pca_test),
        "isolation_forest": (forest_train, forest_test),
    }


def fit_detectors(dataset: dict[str, Any], preprocessors: dict[str, ChannelPreprocessor]) -> dict[str, Any]:
    fitted: dict[str, Any] = {}
    for channel, path in dataset["train"].items():
        train = preprocessors[channel].transform(np.load(path, mmap_mode="r"))
        split = max(1, int(len(train) * 0.8))
        fitted[channel] = {"scores": _scores(train[:split], train[split:]), "train": train[:split]}
    return fitted


def evaluate_detectors(
    dataset: dict[str, Any], preprocessors: dict[str, ChannelPreprocessor], fitted: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    channel_scores: dict[str, dict[str, np.ndarray]] = {}
    for channel, path in dataset["test"].items():
        test = preprocessors[channel].transform(np.load(path, mmap_mode="r"))
        train = fitted[channel]["train"]
        calibration = fitted[channel]["scores"]
        scores = _scores(train, test)
        channel_scores[channel] = {}
        labels = np.zeros(len(test), dtype=int)
        for start, end in dataset["labels"].get(channel, []):
            labels[max(0, start): min(len(labels), end + 1)] = 1
        for detector in DETECTORS:
            calibration_scores = calibration[detector][1]
            threshold = float(np.percentile(calibration_scores, 99))
            values = scores[detector][1]
            flags = values >= threshold
            normal = labels == 0
            metric = {
                "detector": detector,
                "channel": channel,
                "spacecraft": "SMAP" if channel.startswith(tuple("ABCDEFGH")) else "MSL",
                "threshold": threshold,
                "precision": float(precision_score(labels, flags, zero_division=0)),
                "recall": float(recall_score(labels, flags, zero_division=0)),
                "f1": float(f1_score(labels, flags, zero_division=0)),
                "accuracy": float(np.mean(flags == labels)),
                "fpr": float(np.mean(flags[normal])) if np.any(normal) else 0.0,
                "fnr": float(np.mean(~flags[labels == 1])) if np.any(labels) else 0.0,
                "roc_auc": float(roc_auc_score(labels, values)) if labels.any() and (~labels.astype(bool)).any() else 0.5,
                "pr_auc": float(average_precision_score(labels, values)) if labels.any() else 0.0,
                "normal_only_fpr": float(np.mean(flags[normal])) if np.any(normal) else 0.0,
                "anomaly_point_coverage": float(np.mean(flags[labels == 1])) if np.any(labels) else 0.0,
                "false_positives": int(np.sum(flags & normal)),
                "test_points": len(labels),
            }
            rows.append(metric)
            channel_scores[channel][detector] = values
    return {"rows": rows}, channel_scores


def group_events(flags: np.ndarray, gap: int = 2) -> list[tuple[int, int]]:
    indices = np.flatnonzero(flags)
    if not len(indices):
        return []
    groups: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for index in indices[1:]:
        index = int(index)
        if index - previous > gap:
            groups.append((start, previous))
            start = index
        previous = index
    groups.append((start, previous))
    return groups


def event_metrics(labels: list[list[int]], scores: np.ndarray, threshold: float) -> dict[str, float]:
    actual = [(int(start), int(end)) for start, end in labels]
    predicted = group_events(scores >= threshold)
    detected: list[int] = []
    delays: list[int] = []
    for start, end in actual:
        overlaps = [p for p, q in predicted if q >= start and p <= end]
        if overlaps:
            detected.append(start)
            delays.append(max(0, min(overlaps) - start))
    recall = len(detected) / len(actual) if actual else 0.0
    precision = len(detected) / len(predicted) if predicted else 0.0
    false_alarms = sum(
        not any(q >= start and p <= end for start, end in actual)
        for p, q in predicted
    )
    normal_points = max(1, int(scores.size - sum(end - start + 1 for start, end in actual)))
    return {
        "actual_events": len(actual),
        "predicted_events": len(predicted),
        "event_recall": recall,
        "event_precision": precision,
        "event_f1": 2 * recall * precision / (recall + precision) if recall + precision else 0.0,
        "mean_detection_delay": float(np.mean(delays)) if delays else None,
        "median_detection_delay": float(np.median(delays)) if delays else None,
        "detected_before_midpoint": float(
            np.mean([delay <= (end - start) / 2 for (start, end), delay in zip(actual, delays)])
        ) if delays else 0.0,
        "false_alarms": int(false_alarms),
        "false_alarms_per_1000_normal_points": float(false_alarms / normal_points * 1000),
    }


def run_robustness(dataset: dict[str, Any], preprocessors: dict[str, ChannelPreprocessor]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, kind in (("bias", "bias"), ("drift", "drift"), ("spike", "spike"), ("stuck_at", "stuck_at"), ("noise", "noise"), ("dropout", "dropout")):
        channel = sorted(dataset["test"])[0]
        raw = np.asarray(np.load(dataset["test"][channel]), dtype=np.float64).copy()
        rng = np.random.default_rng(SEED)
        if kind == "bias":
            raw[:, 0] += raw[:, 0].std() * 2
        elif kind == "drift":
            raw[:, 0] += np.linspace(0, raw[:, 0].std() * 4, len(raw))
        elif kind == "spike":
            raw[len(raw) // 2: len(raw) // 2 + 5, 0] += raw[:, 0].std() * 8
        elif kind == "stuck_at":
            raw[len(raw) // 2: len(raw) // 2 + 20, 0] = raw[:, 0].mean()
        elif kind == "noise":
            raw[:, 0] += rng.normal(0, raw[:, 0].std() * 2, len(raw))
        else:
            raw[len(raw) // 2: len(raw) // 2 + 20, 0] = np.nan
            raw = np.nan_to_num(raw, nan=float(np.nanmean(raw)))
        transformed = preprocessors[channel].transform(raw)
        train = preprocessors[channel].transform(np.load(dataset["train"][channel], mmap_mode="r"))
        scores = _scores(train, transformed)["robust"]
        threshold = float(np.percentile(scores[0], 99))
        flags = scores[1] >= threshold
        labels = np.zeros(len(raw), dtype=int)
        start = len(raw) // 2
        end = min(len(raw), start + (5 if kind == "spike" else 20))
        labels[start:end] = 1
        rows.append(
            {
                "perturbation": name,
                "threshold": threshold,
                "precision": float(np.sum(flags & (labels == 1)) / max(1, np.sum(flags))),
                "recall": float(np.mean(flags[labels == 1])),
                "f1": float(
                    2 * np.sum(flags & (labels == 1))
                    / max(1, np.sum(flags) + np.sum(labels == 1))
                ),
                "detection_delay": int(np.flatnonzero(flags & (labels == 1))[0] - start)
                if np.any(flags & (labels == 1))
                else None,
                "deterministic_seed": SEED,
            }
        )
    return rows


def environment_manifest() -> dict[str, Any]:
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "seed": SEED,
        "protocol": "80% train fit / 20% normal train calibration / frozen test evaluation",
        "threshold": "99th percentile of calibration scores",
        "runtime_module": __name__,
        "platform": platform.platform(),
        "argv": sys.argv,
    }
