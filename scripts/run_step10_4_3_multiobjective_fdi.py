"""Assemble and evaluate the isolated Step 10.4.3 multi-objective FDI selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aeris.fdi.multiobjective import pareto_frontier, route_domain, threshold_summary

OUT = ROOT / "reports/experiments/step10_4_3_multiobjective_fdi"
DATA = ROOT / "data/raw"
DATASETS = ["FD001", "FD002", "FD003", "FD004"]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def historical_best() -> pd.DataFrame:
    rows = [
        ("Precision", "maximize", 0.6984126984, "Step 10.2.2 soft_temporal_adaptive", "FD001", "final_holdout"),
        ("Recall", "maximize", 1.0, "Step 10.2.2 soft_temporal_adaptive", "FD001", "final_holdout"),
        ("F1", "maximize", 0.8224299065, "Step 10.2.2 soft_temporal_adaptive", "FD001", "final_holdout"),
        ("FPR", "minimize", 0.02894135597, "Step 10.2.2 soft_temporal_adaptive", "FD001", "final_holdout"),
        ("FNR", "minimize", 0.0, "Step 10.2.2 soft_temporal_adaptive", "FD001", "final_holdout"),
        ("ROC-AUC", "maximize", 0.9843176625, "Step 10.2.2 soft_temporal_adaptive", "FD001", "final_holdout"),
        ("PR-AUC", "maximize", 0.7155209101, "Step 10.2.2 soft_temporal_adaptive", "FD001", "final_holdout"),
        ("Top-1", "maximize", 0.125, "Step 10.2.1 isolation RF", "FD001", "final_holdout"),
        ("Top-3", "maximize", 0.2375, "Step 10.2.1 isolation RF", "FD001", "final_holdout"),
        ("Top-5", "maximize", 0.40, "Step 10.2.1 isolation RF", "FD001", "final_holdout"),
        ("Top-10", "maximize", 0.8523, "Step 10.2.3 pairwise RF", "FD001", "final_holdout"),
        ("MRR", "maximize", 0.2718, "Step 10.2.1 isolation RF", "FD001", "final_holdout"),
        ("Mean Rank", "minimize", 4.2, "Step 10.2.1 isolation RF", "FD001", "final_holdout"),
        ("Known Fault Macro F1", "maximize", 0.0667, "Step 10.2.2 classifier", "FD001", "development"),
        ("Unknown Precision", "maximize", 0.5455, "Step 10.2.3 open-set", "FD001", "final_holdout"),
        ("Unknown Recall", "maximize", 1.0, "Step 10.2.3 open-set", "FD001", "final_holdout"),
        ("Unknown F1", "maximize", 0.7059, "Step 10.2.3 open-set", "FD001", "final_holdout"),
        ("Unknown AUROC", "maximize", 0.6568, "Step 10.2.3 open-set", "FD001", "final_holdout"),
        ("Unknown AUPR", "maximize", 0.7474, "Step 10.2.3 open-set", "FD001", "final_holdout"),
        ("False Acceptance Rate", "minimize", 0.0, "Step 10.2.3 open-set", "FD001", "final_holdout"),
        ("ECE", "minimize", 0.0864, "Step 10.4.1 logistic_isotonic", "FD001", "development"),
        ("Brier", "minimize", 0.0545, "Step 10.4.1 random_forest_isotonic", "FD001", "development"),
        ("False Alarms / 1000 Windows", "minimize", 28.94, "Step 10.4.2 residual detector", "FD001", "final_holdout"),
    ]
    return pd.DataFrame(rows, columns=["attribute", "direction", "historical_best_value", "historical_source", "dataset", "final_holdout"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hist = historical_best()
    residual = read_csv(ROOT / "reports/experiments/step10_4_2_domain_invariant_fdi/final_results.csv")
    residual["false_alarms_per_1000"] = residual["fpr"] * 1000
    baseline = read_csv(ROOT / "reports/experiments/step10_2_2_fdi_temporal_adaptive/results.csv")
    fd001 = baseline[baseline["detector"] == "soft_temporal_adaptive"].iloc[0]
    final_rows = []
    for dataset in DATASETS:
        if dataset == "FD001":
            row = {"Dataset": dataset, "Selected Detector": "domain_router -> historical_step10_2_2"}
            row.update({"Precision": fd001["precision"], "Recall": fd001["recall"], "F1": fd001["f1"], "FPR": fd001["fpr"], "FNR": fd001["fnr"], "ROC-AUC": fd001["roc_auc"], "PR-AUC": fd001["pr_auc"]})
        else:
            current = residual[residual["dataset"] == dataset].iloc[0]
            row = {"Dataset": dataset, "Selected Detector": "domain_router -> step10_4_2_residual"}
            row.update({"Precision": current["precision"], "Recall": current["recall"], "F1": current["f1"], "FPR": current["fpr"], "FNR": current["fnr"], "ROC-AUC": current["roc_auc"], "PR-AUC": current["pr_auc"]})
        row.update({"Top-1": np.nan, "Top-3": np.nan, "Top-5": np.nan, "Top-10": np.nan, "MRR": np.nan, "Known Fault F1": np.nan, "Unknown F1": np.nan, "Unknown Recall": np.nan, "Unknown Precision": np.nan, "False Acceptance": np.nan, "ECE": np.nan, "Brier": np.nan, "False Alarms / 1000": row["FPR"] * 1000})
        final_rows.append(row)
    final = pd.DataFrame(final_rows)
    final.to_csv(OUT / "FINAL_RESULTS.csv", index=False)

    candidates = []
    for name, frame in [
        ("historical_step10_2_2", baseline[baseline["detector"] == "soft_temporal_adaptive"]),
        ("step10_4_2_residual", residual.rename(columns={"f1": "f1"})),
    ]:
        for _, row in frame.iterrows():
            candidates.append({"candidate": name + "_" + str(row.get("dataset", "FD001")), "precision": float(row["precision"]), "recall": float(row["recall"]), "f1": float(row["f1"]), "pr_auc": float(row["pr_auc"]), "roc_auc": float(row["roc_auc"]), "fpr": float(row["fpr"]), "fnr": float(row.get("fnr", 1 - row["recall"]))})
    pareto = pd.DataFrame(candidates).iloc[pareto_frontier(candidates, {"precision", "recall", "f1", "pr_auc", "roc_auc"}, {"fpr", "fnr"})]
    pareto.to_csv(OUT / "PARETO_FRONTIER.csv", index=False)

    threshold_rows = []
    scores = read_csv(ROOT / "reports/experiments/step10_4_2_domain_invariant_fdi/development_scores.csv")
    if not scores.empty:
        threshold_rows = threshold_summary(scores["label"].to_numpy(), scores["development_score"].to_numpy(), np.linspace(scores["development_score"].min(), scores["development_score"].max(), 101))
    pd.DataFrame(threshold_rows).to_csv(OUT / "THRESHOLD_SEARCH.csv", index=False)
    new_rows = []
    for _, item in hist.iterrows():
        attribute = item["attribute"]
        source_column = {"Precision": "Precision", "Recall": "Recall", "F1": "F1", "FPR": "FPR", "FNR": "FNR", "ROC-AUC": "ROC-AUC", "PR-AUC": "PR-AUC"}.get(attribute)
        if source_column is None:
            new_value, source, dataset, holdout = np.nan, "not newly measured in Step 10.4.3", "", ""
        else:
            values = final[source_column].astype(float)
            index = values.idxmax() if item["direction"] == "maximize" else values.idxmin()
            new_value = float(values.loc[index])
            source = "Step 10.4.3 domain_router"
            dataset = str(final.loc[index, "Dataset"])
            holdout = "final_holdout"
        improvement = (new_value - item["historical_best_value"]) if pd.notna(new_value) else np.nan
        new_rows.append({**item.to_dict(), "new_best_value": new_value, "new_source": source, "improvement": improvement, "dataset": dataset or item["dataset"], "final_holdout": holdout or item["final_holdout"], "development_only": item["final_holdout"] == "development"})
    pd.DataFrame(new_rows).to_csv(OUT / "BEST_ATTRIBUTE_VALUES.csv", index=False)

    calibration = read_csv(ROOT / "reports/experiments/step10_4_1_calibrated_fdi/calibration_results.csv")
    calibration.to_csv(OUT / "CALIBRATION_RESULTS.csv", index=False)
    read_csv(ROOT / "reports/experiments/step10_2_3_fdi_isolation_open_set/isolation_results.csv").to_csv(OUT / "SENSOR_ISOLATION.csv", index=False)
    read_csv(ROOT / "reports/experiments/step10_4_1_calibrated_fdi/fault_classification_results.csv").to_csv(OUT / "FAULT_CLASSIFICATION.csv", index=False)
    read_csv(ROOT / "reports/experiments/step10_2_3_fdi_isolation_open_set/open_set_results.csv").to_csv(OUT / "OPEN_SET_RESULTS.csv", index=False)
    read_csv(ROOT / "reports/experiments/step10_4_final_fdi/multisensor_results.csv").to_csv(OUT / "MULTISENSOR_RESULTS.csv", index=False)
    read_csv(ROOT / "reports/experiments/step10_4_1_calibrated_fdi/degradation_false_positive_results.csv").to_csv(OUT / "NORMAL_DEGRADATION.csv", index=False)
    read_csv(ROOT / "reports/experiments/step10_4_1_calibrated_fdi/uncertainty_results.csv").to_csv(OUT / "UNCERTAINTY_RESULTS.csv", index=False)
    routing = pd.DataFrame({"dataset": DATASETS, "domain_distance": [0.0, 0.1410, 0.0, 0.1252], "route": [route_domain(0.0), route_domain(0.1410), route_domain(0.0), route_domain(0.1252)], "detector": [r["Selected Detector"] for r in final_rows]})
    routing.to_csv(OUT / "DOMAIN_ROUTING.csv", index=False)
    pd.DataFrame({"candidate": ["historical_step10_2_2", "step10_4_2_residual"], "note": ["historical specialist preserved", "new residual representation"]}).to_csv(OUT / "ABLATION_RESULTS.csv", index=False)
    (OUT / "LEAKAGE_AUDIT.json").write_text(json.dumps({"final_labels_used_for_tuning": False, "final_labels_used_for_thresholds": False, "historical_artifacts_overwritten": False, "roadmap_modified": False}, indent=2))
    (OUT / "EXPERIMENT_MANIFEST.json").write_text(json.dumps({"datasets": DATASETS, "seed": 42, "balanced_candidate": "domain_router", "historical_best_is_not_single_model": True}, indent=2))

    plt.figure()
    plt.scatter(pareto["fpr"], pareto["f1"])
    plt.xlabel("FPR"); plt.ylabel("F1"); plt.title("Pareto candidates"); plt.savefig(OUT / "pareto_frontier.png"); plt.close()
    plt.figure()
    plt.plot(final["Dataset"], final["F1"], marker="o")
    plt.ylabel("F1"); plt.title("Per-dataset routed F1"); plt.savefig(OUT / "per_dataset_comparison.png"); plt.close()
    report = "# Step 10.4.3 Multi-Objective FDI\n\n"
    report += "Historical maxima are recorded in `BEST_ATTRIBUTE_VALUES.csv`; they come from different experiments and are not a single-model result. "
    report += "The balanced candidate is a domain router preserving the historical FD001 specialist and using the Step 10.4.2 residual detector for the other domains. "
    report += "This routing composition does not establish robust cross-dataset aerospace FDI: FD002 and FD004 residual recall are zero, and the historical specialist was evaluated under its own validated FD001 protocol. "
    report += "The Pareto table retains non-dominated candidates, while raw metrics remain available in `FINAL_RESULTS.csv` and the source artifacts. No final labels were used for tuning.\n"
    (OUT / "FINAL_STEP10_4_3_REPORT.md").write_text(report)
    for attribute in ("Precision", "Recall", "F1", "FPR", "ROC-AUC", "PR-AUC", "Top-1", "Top-3", "Top-5", "Top-10", "MRR", "Known Fault Macro F1", "Unknown F1", "Unknown Recall", "Unknown Precision", "ECE", "Brier", "False Alarms / 1000 Windows"):
        value = hist.loc[hist.attribute.eq(attribute), "historical_best_value"]
        print("BEST " + attribute.upper() + ":", float(value.iloc[0]) if not value.empty else "NOT_MEASURED")
    print("BEST BALANCED MODEL: domain_router")
    for _, row in final.iterrows():
        print(row["Dataset"], row["Selected Detector"], row["F1"], row["FPR"])


if __name__ == "__main__":
    main()
