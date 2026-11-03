"""Schema-preserving SMAP/MSL integration audit without fabricated data."""

from __future__ import annotations

import json
import csv
import ast
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def audit_spacecraft_data(raw_root: str | Path) -> dict[str, Any]:
    root = Path(raw_root)
    files = sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()) if root.exists() else []
    train_files = sorted((root / "train").glob("*.npy")) if (root / "train").exists() else []
    test_files = sorted((root / "test").glob("*.npy")) if (root / "test").exists() else []
    label_path = root / "labeled_anomalies.csv"
    available = bool(train_files and test_files and label_path.exists())
    shape_counts: dict[str, int] = {}
    missing_values = 0
    lengths: list[int] = []
    dtypes: dict[str, int] = {}
    for path in train_files + test_files:
        array = np.load(path, mmap_mode="r")
        shape_counts[str(tuple(array.shape))] = shape_counts.get(str(tuple(array.shape)), 0) + 1
        dtypes[str(array.dtype)] = dtypes.get(str(array.dtype), 0) + 1
        lengths.append(int(array.shape[0]))
        if np.issubdtype(array.dtype, np.floating):
            missing_values += int(np.isnan(array).sum())
    label_rows = list(csv.DictReader(label_path.open(newline="", encoding="utf-8"))) if label_path.exists() else []
    spacecraft_counts: dict[str, int] = {}
    parseable_anomaly_sequences = 0
    for row in label_rows:
        spacecraft = row.get("spacecraft", "UNKNOWN")
        spacecraft_counts[spacecraft] = spacecraft_counts.get(spacecraft, 0) + 1
        try:
            sequences = ast.literal_eval(row.get("anomaly_sequences", ""))
            if isinstance(sequences, list) and all(
                isinstance(pair, list) and len(pair) == 2 and all(isinstance(value, int) for value in pair)
                for pair in sequences
            ):
                parseable_anomaly_sequences += 1
        except (SyntaxError, ValueError):
            pass
    train_ids = {path.stem for path in train_files}
    test_ids = {path.stem for path in test_files}
    label_ids = [row.get("chan_id", "") for row in label_rows]
    label_id_set = set(label_ids)
    train_test_ids_match = train_ids == test_ids
    label_ids_subset_of_files = label_id_set <= train_ids
    duplicate_label_channels = sorted(
        channel_id for channel_id, count in Counter(label_ids).items() if count > 1
    )
    total_size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    return {
        "status": "AVAILABLE" if available else "DATA_REQUIRED",
        "source": "Original NASA/JPL SMAP/MSL anomaly benchmark distributed via Telemanom",
        "original_source": "https://github.com/khundman/telemanom",
        "distribution_source": "https://huggingface.co/datasets/appleparan/telemanom",
        "root": str(root),
        "files": files,
        "train_files": len(train_files),
        "test_files": len(test_files),
        "label_records": len(label_rows),
        "spacecraft_counts": spacecraft_counts,
        "channel_ids_match": train_test_ids_match,
        "label_ids_subset_of_files": label_ids_subset_of_files,
        "unique_label_channels": len(label_id_set),
        "duplicate_label_channels": duplicate_label_channels,
        "parseable_anomaly_sequences": parseable_anomaly_sequences,
        "shape_counts": shape_counts,
        "dtypes": dtypes,
        "missing_values": missing_values,
        "sequence_length_min": min(lengths) if lengths else None,
        "sequence_length_max": max(lengths) if lengths else None,
        "size_bytes": total_size,
        "label_columns": list(label_rows[0]) if label_rows else None,
        "split": "temporal/channel-level; normal training only; validation threshold calibration",
        "license": "BSD-style license included in data/raw/SMAP_MSL/LICENSE.txt",
        "reason": (
            "Expected train/test NPY files and labeled_anomalies.csv were not found."
            if not available
            else None
        ),
    }


def write_spacecraft_audit(raw_root: str | Path, output: str | Path) -> dict[str, Any]:
    result = audit_spacecraft_data(raw_root)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
