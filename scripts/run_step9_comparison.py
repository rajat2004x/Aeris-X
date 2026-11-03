"""Compare completed AERIS-X model experiment artifacts without retraining."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aeris.evaluation.comparison import run_comparison


if __name__ == "__main__":
    result = run_comparison()
    print(result.to_string(index=False))
    print("\nResults written to reports/experiments/step9_comparison")
