"""Run the Step 12 health-state estimation benchmark."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aeris.health.pipeline import run_step12


if __name__ == "__main__":
    result = run_step12()
    print({"selected_model": result["selected_model"], **result["final_metrics"]})
    print("Results written to reports/experiments/step12_health_state")
