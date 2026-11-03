"""Run isolated FD001-FD004 Step 12 health-state evaluation."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aeris.health.multidataset import run_multidataset


if __name__ == "__main__":
    result = run_multidataset()
    print(result["transfer"][["source", "target", "mae", "rmse", "r2", "spearman_correlation"]].to_string(index=False))
