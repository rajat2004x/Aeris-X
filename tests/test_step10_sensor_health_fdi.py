import json
from pathlib import Path

import pandas as pd
from aeris.fdi.integration import build_final_step10


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/experiments/step10_sensor_health_fdi"


def test_final_step10_package_contains_all_datasets_and_metrics():
    manifest = build_final_step10(OUTPUT)
    assert manifest["final_holdout_frozen"] is True
    results = pd.read_csv(OUTPUT / "final_detection_results.csv")
    assert list(results["dataset"]) == ["FD001", "FD002", "FD003", "FD004"]
    assert {"precision", "recall", "f1", "fpr", "roc_auc", "pr_auc"} <= set(results.columns)
    assert (OUTPUT / "FINAL_STEP10_REPORT.md").stat().st_size > 0


def test_leakage_audit_and_historical_protection():
    audit = json.loads((OUTPUT / "leakage_audit.json").read_text(encoding="utf-8"))
    assert audit["final_holdout_frozen"] is True
    assert (ROOT / "reports/experiments/step10_4_final_fdi/FINAL_STEP10_4_REPORT.md").exists()
    assert (ROOT / "reports/experiments/step10_4_3_multiobjective_fdi/LEAKAGE_AUDIT.json").exists()
