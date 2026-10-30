import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_build_gate import check
from validate_roadmap import validate


def test_step_7_is_current_and_allowed():
    assert validate()[0]
    assert json.loads((ROOT / ".aeris/state.json").read_text())["current_step"] == "7"
    assert check("7")[0]


def test_future_steps_are_blocked():
    assert not check("8")[0]
    assert not check("10")[0]


def test_historical_steps_do_not_advance_state():
    state = json.loads((ROOT / ".aeris/state.json").read_text())
    assert "7" not in state["completed_steps"]
    assert state["historical_out_of_order_steps"] == ["10.1", "10.2"]


def test_completed_foundation_steps_remain_complete():
    state = json.loads((ROOT / ".aeris/state.json").read_text())
    assert state["completed_steps"][:6] == ["1", "2", "3", "4", "5", "6"]


def test_invalid_state_fails(tmp_path):
    state_path = ROOT / ".aeris/state.json"
    original = state_path.read_text(encoding="utf-8")
    try:
        state = json.loads(original)
        state["current_step"] = "10"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        assert not validate()[0]
    finally:
        state_path.write_text(original, encoding="utf-8")


@pytest.mark.parametrize(
    "relative_path",
    [
        "reports/experiments/step10_1_fdi_improved/report.md",
        "reports/experiments/step10_2_fdi_multidataset/report.md",
        "data/processed/cmapss_fd001/X_train.npy",
    ],
)
def test_historical_artifacts_remain_present(relative_path):
    assert (ROOT / relative_path).exists()
