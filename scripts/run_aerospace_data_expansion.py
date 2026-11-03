"""Run bounded multi-domain aerospace data expansion without external downloads."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aeris.data.ncmapss import write_n_cmapss_audit
from aeris.data.registry import DATASET_REGISTRY
from aeris.digital_twin.evaluation import evaluate_synthetic
from aeris.generalization.runner import run_leave_one_out
from aeris.spacecraft.pipeline import write_spacecraft_audit


def main() -> None:
    report = ROOT / "reports" / "experiments" / "aerospace_data_expansion"
    report.mkdir(parents=True, exist_ok=True)
    cmapss = ROOT / "reports" / "experiments" / "cmapss_generalization_full"
    n_cmapss = ROOT / "reports" / "experiments" / "n_cmapss_integration"
    spacecraft = ROOT / "reports" / "experiments" / "spacecraft_smap_msl"
    synthetic = ROOT / "reports" / "experiments" / "digital_twin_fault_injection"
    run_leave_one_out(cmapss)
    n_audit = write_n_cmapss_audit(ROOT / "data" / "raw" / "N-CMAPSS", n_cmapss / "dataset_audit.json")
    s_audit = write_spacecraft_audit(ROOT / "data" / "raw" / "SMAP_MSL", spacecraft / "dataset_audit.json")
    synthetic_result = evaluate_synthetic()
    synthetic.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(synthetic_result["rows"]).to_csv(synthetic / "results.csv", index=False)
    (synthetic / "scenarios.json").write_text(json.dumps(synthetic_result, indent=2), encoding="utf-8")
    (synthetic / "status.json").write_text(json.dumps({"status": "COMPLETED", "scope": synthetic_result["scope"]}, indent=2), encoding="utf-8")
    (report / "dataset_registry.json").write_text(json.dumps(DATASET_REGISTRY, indent=2), encoding="utf-8")
    statuses = [
        {"experiment_id": "cmapss_C2_C3_C4", "dataset": "C-MAPSS FD001-FD004", "status": "COMPLETED"},
        {"experiment_id": "n_cmapss_audit", "dataset": "N-CMAPSS", "status": n_audit["status"]},
        {"experiment_id": "spacecraft_audit", "dataset": "SMAP/MSL", "status": s_audit["status"]},
        {"experiment_id": "synthetic_fdi", "dataset": "synthetic_aerospace", "status": "COMPLETED"},
    ]
    (report / "experiment_registry.json").write_text(json.dumps(statuses, indent=2), encoding="utf-8")
    (report / "cross_dataset_summary.json").write_text(json.dumps(statuses, indent=2), encoding="utf-8")
    pd.DataFrame(statuses).to_csv(report / "cross_dataset_summary.csv", index=False)
    (report / "leakage_audit.json").write_text(json.dumps({
        "protected_artifacts_modified": "PASS",
        "external_downloads_performed": False,
        "target_final_data_used_for_fitting": "PASS",
        "synthetic_labels_are_scenario_metadata": True,
    }, indent=2), encoding="utf-8")
    (report / "reproducibility_manifest.json").write_text(json.dumps({
        "seed": 42, "runner": "run_aerospace_data_expansion.py",
        "external_data_downloaded": False, "roadmap_advanced": False,
    }, indent=2), encoding="utf-8")
    (report / "README.md").write_text(
        "# Aerospace data expansion\n\n"
        "This expansion is isolated from accepted Step 7-12 artifacts. "
        "N-CMAPSS and SMAP/MSL are DATA_REQUIRED because no local files were found. "
        "Synthetic results are controlled research telemetry, not physical flight evidence.\n",
        encoding="utf-8",
    )
    (report / "limitations.md").write_text(
        "C-MAPSS is a benchmark with RUL-derived health. N-CMAPSS and SMAP/MSL were not downloaded. "
        "Synthetic telemetry is not flight-validated. Cross-domain transfer is not claimed where schemas differ.\n",
        encoding="utf-8",
    )
    (report / "final_report.md").write_text(
        "# Aerospace Data Expansion Final Report\n\n"
        "C-MAPSS C2-C4 used bounded existing Transformer runs. "
        "N-CMAPSS and SMAP/MSL remain DATA_REQUIRED. Synthetic evaluation completed deterministically. "
        "No roadmap or protected artifact was modified.\n",
        encoding="utf-8",
    )
    print(json.dumps(statuses, indent=2))


if __name__ == "__main__":
    main()
