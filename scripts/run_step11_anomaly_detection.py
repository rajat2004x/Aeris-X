"""Run the leakage-safe Step 11 system anomaly benchmark."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aeris.anomaly.pipeline import run_step11


if __name__ == "__main__":
    result = run_step11()
    print(result["results"])
    print("Results written to reports/experiments/step11_anomaly_detection")
