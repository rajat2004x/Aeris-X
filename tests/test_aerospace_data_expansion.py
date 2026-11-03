import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aeris.data.ncmapss import audit_n_cmapss
from aeris.data.registry import DATASET_REGISTRY
from aeris.digital_twin.evaluation import evaluate_synthetic
from aeris.digital_twin.generator import generate_scenario
from aeris.experiment.runtime import RuntimeGuard
from aeris.spacecraft.pipeline import audit_spacecraft_data


ROOT = Path(__file__).resolve().parents[1]


def test_registry_contains_domain_contracts():
    ids = {item["dataset_id"] for item in DATASET_REGISTRY}
    assert {"cmapss_fd001", "n_cmapss", "smap", "msl", "synthetic_aerospace"} <= ids


def test_missing_external_datasets_are_data_required(tmp_path):
    assert audit_n_cmapss(tmp_path)["status"] == "DATA_REQUIRED"
    assert audit_spacecraft_data(tmp_path)["status"] == "DATA_REQUIRED"


def test_acquired_spacecraft_package_is_validated():
    audit = audit_spacecraft_data(ROOT / "data" / "raw" / "SMAP_MSL")
    assert audit["status"] == "AVAILABLE"
    assert audit["train_files"] == 82
    assert audit["test_files"] == 82
    assert audit["spacecraft_counts"] == {"SMAP": 55, "MSL": 27}
    assert audit["channel_ids_match"] is True
    assert audit["label_ids_subset_of_files"] is True
    assert audit["unique_label_channels"] == 81
    assert audit["duplicate_label_channels"] == ["P-2"]
    assert audit["parseable_anomaly_sequences"] == 82
    assert audit["missing_values"] == 0


def test_synthetic_generation_is_deterministic_and_labeled():
    first, metadata = generate_scenario("BIAS", seed=7)
    second, metadata_again = generate_scenario("BIAS", seed=7)
    assert np.array_equal(first, second, equal_nan=True)
    assert metadata == metadata_again
    assert metadata["ground_truth"]["fault"] is True


def test_synthetic_evaluation_schema():
    result = evaluate_synthetic()
    assert len(result["rows"]) == 11
    assert {"top1", "top3", "top5"} <= set(result["rows"][1])


def test_runtime_guard_writes_status(tmp_path):
    guard = RuntimeGuard(tmp_path, "unit", 10)
    guard.complete({"status_value": "ok"})
    status = json.loads((tmp_path / "unit_status.json").read_text())
    assert status["status"] == "COMPLETED"
