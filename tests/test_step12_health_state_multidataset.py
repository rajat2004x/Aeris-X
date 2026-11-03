import json
from pathlib import Path

import numpy as np
import pytest

from aeris.health.multidataset import (
    FEATURE_COLUMNS,
    apply_affine,
    characterize_dataset,
    prepare_dataset,
    run_multidataset,
    source_affine_map,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/experiments/step12_health_state_multidataset"


@pytest.mark.parametrize("name", ["FD002", "FD003", "FD004"])
def test_additional_dataset_parsing_and_characterization(name):
    info = characterize_dataset(name, ROOT / "data" / "raw")
    assert info["selected_feature_count"] == 15
    assert info["train_engines"] >= 100
    assert info["operating_condition_count"] in {1, 6}
    assert info["fault_mode_count"] is None


def test_preparation_preserves_engine_boundaries_and_scaler_provenance(tmp_path):
    metadata = prepare_dataset("FD002", ROOT / "data" / "raw", tmp_path)
    assert metadata["scaler_fit_on"] == "train engines only"
    directory = tmp_path / "fd002"
    train_owners = np.load(directory / "owners_train.npy")
    final_owners = np.load(directory / "owners_final.npy")
    assert set(train_owners).isdisjoint(set(final_owners))
    assert np.isfinite(np.load(directory / "X_train.npy")).all()


def test_frozen_affine_transfer_mapping_and_dimensions():
    mapping = source_affine_map(ROOT / "data" / "raw", ROOT / "data" / "processed" / "cmapss_fd001")
    transformed = apply_affine(np.zeros((2, 30, 15)), mapping)
    assert transformed.shape == (2, 30, 15)
    assert len(FEATURE_COLUMNS) == 15
    with pytest.raises(ValueError):
        apply_affine(np.zeros((2, 30, 14)), mapping)


def test_multidataset_artifact_schema_and_roadmap_state_unchanged():
    state_before = (ROOT / ".aeris" / "state.json").read_bytes()
    result = run_multidataset(output_dir=OUTPUT)
    assert set(result["transfer"]["target"]) == {"FD001", "FD002", "FD003", "FD004"}
    assert (ROOT / ".aeris" / "state.json").read_bytes() == state_before
    audit = json.loads((OUTPUT / "leakage_audit.json").read_text(encoding="utf-8"))
    assert audit["engine_level_split"] == "PASS"
    assert audit["target_final_labels_used_for_selection"] == "PASS"
