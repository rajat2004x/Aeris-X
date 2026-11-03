"""Build the isolated final Step 10 sensor-health/FDI evidence package."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aeris.fdi.integration import build_final_step10


if __name__ == "__main__":
    build_final_step10()
    print("Final Step 10 artifacts written to reports/experiments/step10_sensor_health_fdi")
