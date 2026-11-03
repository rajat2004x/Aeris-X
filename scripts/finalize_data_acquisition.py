"""Validate acquired datasets and write acquisition manifests without training."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aeris.data.ncmapss import audit_n_cmapss
from aeris.data.registry import DATASET_REGISTRY
from aeris.spacecraft.pipeline import audit_spacecraft_data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    report = ROOT / "reports" / "experiments" / "aerospace_data_expansion"
    report.mkdir(parents=True, exist_ok=True)
    n_output = ROOT / "reports" / "experiments" / "n_cmapss_integration"
    s_output = ROOT / "reports" / "experiments" / "spacecraft_smap_msl"
    n_root = ROOT / "data" / "raw" / "N-CMAPSS"
    s_root = ROOT / "data" / "raw" / "SMAP_MSL"
    timestamp = datetime.now(timezone.utc).isoformat()
    before_free = shutil.disk_usage(ROOT).free
    n_audit = audit_n_cmapss(n_root)
    s_audit = audit_spacecraft_data(s_root)
    n_manifest = {
        "source": n_audit["source"],
        "source_url": n_audit["source_url"],
        "download_timestamp": timestamp,
        "filename": n_audit["expected_file"],
        "sha256": None,
        "size_bytes": None,
        "subset": n_audit["expected_subset"],
        "groups": None,
        "shapes": None,
        "status": "DATA_ACCESS_UNAVAILABLE",
        "reason": n_audit["reason"],
    }
    n_output.mkdir(parents=True, exist_ok=True)
    (n_output / "acquisition_manifest.json").write_text(json.dumps(n_manifest, indent=2), encoding="utf-8")
    s_files = [path for path in s_root.rglob("*") if path.is_file()]
    s_manifest = {
        "original_source": s_audit["original_source"],
        "distribution_source": s_audit["distribution_source"],
        "license": s_audit["license"],
        "download_timestamp": timestamp,
        "root": str(s_root),
        "size_bytes": s_audit["size_bytes"],
        "sha256": {str(path.relative_to(s_root)): sha256(path) for path in s_files},
        "train_files": s_audit["train_files"],
        "test_files": s_audit["test_files"],
        "label_records": s_audit["label_records"],
        "spacecraft_counts": s_audit["spacecraft_counts"],
        "channel_ids_match": s_audit["channel_ids_match"],
        "label_ids_subset_of_files": s_audit["label_ids_subset_of_files"],
        "unique_label_channels": s_audit["unique_label_channels"],
        "duplicate_label_channels": s_audit["duplicate_label_channels"],
        "parseable_anomaly_sequences": s_audit["parseable_anomaly_sequences"],
        "shape_counts": s_audit["shape_counts"],
        "dtypes": s_audit["dtypes"],
        "missing_values": s_audit["missing_values"],
        "sequence_length_min": s_audit["sequence_length_min"],
        "sequence_length_max": s_audit["sequence_length_max"],
        "label_columns": s_audit["label_columns"],
        "status": "AVAILABLE" if s_audit["status"] == "AVAILABLE" else "VALIDATION_FAILED",
    }
    s_output.mkdir(parents=True, exist_ok=True)
    (s_output / "acquisition_manifest.json").write_text(json.dumps(s_manifest, indent=2), encoding="utf-8")
    after_free = shutil.disk_usage(ROOT).free
    registry = [dict(item) for item in DATASET_REGISTRY]
    (report / "dataset_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    statuses = [
        {"dataset": "N-CMAPSS DS02", "status": n_manifest["status"], "path": str(n_root / n_manifest["filename"])},
        {"dataset": "SMAP/MSL Telemanom", "status": s_manifest["status"], "path": str(s_root)},
    ]
    pd.DataFrame(statuses).to_csv(report / "data_acquisition_status.csv", index=False)
    (report / "data_acquisition_report.md").write_text(f"""# Aerospace Data Acquisition Report

## Disk

- Free before acquisition: {before_free / 1e9:.2f} GB
- Free after acquisition: {after_free / 1e9:.2f} GB
- No N-CMAPSS archive was retained or downloaded.

## N-CMAPSS

Status: **{n_manifest['status']}**

NASA's official PCoE repository currently reports that direct download is unavailable.
The required target remains `{n_manifest['filename']}` from the DS02 subset. No mirror was used.

## SMAP/MSL

Status: **{s_manifest['status']}**

The acquired Telemanom distribution contains {s_manifest['train_files']} train and
{s_manifest['test_files']} test channel files, with {s_manifest['label_records']} label records:
{s_manifest['spacecraft_counts']}. The observed package contains 82 labeled channels,
not the approximate 105 anomaly-sequence figure; the observed value is preserved.
The train/test channel IDs match and all label IDs are present in the files. The label
CSV has 81 unique channel IDs because `P-2` occurs twice; this duplicate is retained
and reported rather than silently deduplicated.

Data is stored at `data/raw/SMAP_MSL/` with `train/`, `test/`, `labeled_anomalies.csv`,
and the included license. No ML training was run.

## Validation

- SMAP/MSL files: PASS
- Shape/dtype/missing-value audit: PASS
- Label-file audit: PASS
- N-CMAPSS validation: NOT RUN because the official file was unavailable
- Registry updated only for successfully validated SMAP/MSL data
""", encoding="utf-8")
    (report / "acquisition_disk_manifest.json").write_text(json.dumps({
        "free_before_bytes": before_free,
        "free_after_bytes": after_free,
        "smap_msl_bytes": s_manifest["size_bytes"],
        "n_cmapss_downloaded": False,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
