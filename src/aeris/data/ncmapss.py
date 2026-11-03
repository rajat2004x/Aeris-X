"""Schema-first N-CMAPSS integration audit and preprocessing contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def audit_n_cmapss(raw_root: str | Path) -> dict[str, Any]:
    root = Path(raw_root)
    files = sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()) if root.exists() else []
    available = bool(files)
    return {
        "status": "AVAILABLE" if available else "DATA_REQUIRED",
        "source": "NASA Prognostics Data Repository, N-CMAPSS/C-MAPSS 2",
        "source_url": "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/",
        "expected_subset": "DS02",
        "expected_file": "N-CMAPSS_DS02-006.h5",
        "root": str(root),
        "files": files,
        "subsets": None,
        "engine_or_run_counts": None,
        "sensor_count": None,
        "operating_condition_variables": None,
        "sampling": None,
        "failure_labels": None,
        "health_semantics": None,
        "missing_values": None,
        "constant_channels": None,
        "reason": "NASA currently reports direct download unavailable; no schema or target is assumed." if not available else None,
    }


def write_n_cmapss_audit(raw_root: str | Path, output: str | Path) -> dict[str, Any]:
    result = audit_n_cmapss(raw_root)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
