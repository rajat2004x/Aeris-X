import csv
from pathlib import Path

import numpy as np

from aeris.spacecraft.experiment import (
    discover_dataset,
    event_metrics,
    fit_preprocessors,
    group_events,
    preprocessing_manifest,
    profile_dataset,
    run_robustness,
)


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "spacecraft"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()
    rng = np.random.default_rng(42)
    for channel in ("A-1", "P-2"):
        np.save(root / "train" / f"{channel}.npy", rng.normal(size=(20, 4)))
        np.save(root / "test" / f"{channel}.npy", rng.normal(size=(24, 4)))
    with (root / "labeled_anomalies.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["chan_id", "spacecraft", "anomaly_sequences"])
        writer.writeheader()
        writer.writerow({"chan_id": "A-1", "spacecraft": "SMAP", "anomaly_sequences": "[[3, 5]]"})
        writer.writerow({"chan_id": "P-2", "spacecraft": "MSL", "anomaly_sequences": "[[8, 10]]"})
    return root


def test_profile_and_duplicate_safe_discovery(tmp_path):
    dataset = discover_dataset(_fixture(tmp_path))
    profile = profile_dataset(dataset)
    assert profile["train_file_count"] == 2
    assert profile["label_record_count"] == 2
    assert profile["unique_label_ids"] == 2


def test_preprocessing_uses_train_only_statistics(tmp_path):
    dataset = discover_dataset(_fixture(tmp_path))
    preprocessors = fit_preprocessors(dataset)
    manifest = preprocessing_manifest(dataset, preprocessors)
    assert manifest["leakage_controls"]
    assert len(manifest["train_statistics"]["A-1"]["center"]) == 3


def test_event_grouping_and_metrics():
    assert group_events(np.array([False, True, True, False, False, True])) == [(1, 2), (5, 5)]
    metrics = event_metrics([[1, 3]], np.array([0.0, 2.0, 2.0, 0.0]), 1.0)
    assert metrics["event_recall"] == 1.0
    assert metrics["event_precision"] == 1.0


def test_robustness_is_deterministic(tmp_path):
    dataset = discover_dataset(_fixture(tmp_path))
    preprocessors = fit_preprocessors(dataset)
    assert run_robustness(dataset, preprocessors) == run_robustness(dataset, preprocessors)
