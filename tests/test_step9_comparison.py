import json
from pathlib import Path

import pandas as pd
import pytest

from aeris.evaluation.comparison import run_comparison


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/experiments/step9_comparison"


def test_comparison_artifacts_are_complete_and_json_nulls_are_used():
    frame = run_comparison(output_dir=OUTPUT)
    assert len(frame) == 9
    assert set(frame["model"]) == {"Mean RUL", "Linear Regression", "Random Forest", "XGBoost", "LSTM", "GRU", "TCN", "Transformer"}
    payload = json.loads((OUTPUT / "comparison_results.json").read_text(encoding="utf-8"))
    xgboost = next(row for row in payload["records"] if row["model"] == "XGBoost")
    assert xgboost["parameter_count"] is None
    assert xgboost["mae"] == pytest.approx(18.698258043515352)
    assert all(value is not None for value in payload["derived_comparisons"].values())


def test_rankings_are_deterministic_and_preserve_historical_artifacts():
    first = json.loads((OUTPUT / "comparison_results.json").read_text(encoding="utf-8"))
    second = json.loads((OUTPUT / "comparison_results.json").read_text(encoding="utf-8"))
    assert first["rankings"] == second["rankings"]
    assert first["rankings"]["mae_ascending"][0] == "Transformer (cpu)"
    assert (ROOT / "reports/experiments/step10_4_3_multiobjective_fdi/LEAKAGE_AUDIT.json").exists()
    assert pd.read_csv(ROOT / "reports/experiments/step7_classical_ml/comparison_results.csv").loc[
        lambda frame: frame["model"].eq("xgboost"), "mae"
    ].iloc[0] == pytest.approx(18.698258043515352)
