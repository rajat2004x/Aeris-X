import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aeris.health.condition_aware import (
    FEATURE_COLUMNS,
    condition_key,
    fit_condition_statistics,
    transform_condition_windows,
)


ROOT = Path(__file__).resolve().parents[1]


def _frame() -> pd.DataFrame:
    rows = []
    for engine in (1, 2):
        for cycle in range(1, 5):
            rows.append({
                "engine_id": engine,
                "cycle": cycle,
                "setting_1": float(engine),
                "setting_2": 0.0,
                "setting_3": float(engine),
                **{name: float(engine + cycle) for name in FEATURE_COLUMNS},
            })
    return pd.DataFrame(rows)


def test_condition_identification_is_deterministic():
    keys = condition_key(_frame())
    assert keys.iloc[0] == "1.0|1.0"
    assert keys.iloc[-1] == "2.0|2.0"


def test_condition_statistics_use_training_engines_only():
    frame = _frame()
    stats = fit_condition_statistics(frame, [1])
    assert stats["fit_engines"] == [1]
    assert set(stats["conditions"]) == {"1.0|1.0", "__global__"}
    assert stats["conditions"]["__global__"]["mean"][0] == pytest.approx(3.5)


def test_unseen_condition_uses_global_fallback():
    frame = _frame()
    stats = fit_condition_statistics(frame, [1])
    windows = np.ones((1, 3, len(FEATURE_COLUMNS)), dtype=np.float32)
    transformed, provenance = transform_condition_windows(windows, np.array([2]), frame, stats)
    assert transformed.shape == windows.shape
    assert provenance["fallback_windows"] == 1


def test_result_schema_and_roadmap_files_are_available():
    assert (ROOT / ".aeris" / "state.json").exists()
    assert (ROOT / "reports" / "experiments" / "step12_health_state_multidataset" / "transfer" / "comparison.csv").exists()
