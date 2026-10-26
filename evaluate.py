"""Combine baseline and GNN results into the tables Chapter Four needs.

Implements Section 3.7.4. Produces one comparison table across all six models,
the paired statistical comparison against XGBoost, and detection rate at a
fixed false-positive rate - which is the number an operations team actually
cares about, since nobody deploys at equal error rate.
"""
from __future__ import annotations

import pandas as pd

from common import OUT, fold_summary, paired_tests, save_table


def main() -> None:
    b = pd.read_csv(OUT / "04_05_baseline_raw.csv")
    g = pd.read_csv(OUT / "04_05_gnn_raw.csv")
    allr = pd.concat([b, g], ignore_index=True)
    save_table(allr, "04_05_all_models_raw")

    cv = allr[allr.stage == "cv"].to_dict("records")
    summary = fold_summary(cv)
    order = ["LogisticRegression", "RandomForest", "XGBoost",
             "GCN", "GAT", "GraphSAGE"]
    summary["__o"] = summary["model"].map({m: i for i, m in enumerate(order)})
    summary = summary.sort_values("__o").drop(columns="__o")
    save_table(summary, "04_05_TABLE_detection_performance")

    save_table(paired_tests(cv, "auprc", "XGBoost"), "04_05_TABLE_paired_auprc")
    save_table(paired_tests(cv, "recall", "XGBoost"), "04_05_TABLE_paired_recall")

    test = allr[allr.stage == "test"].copy()
    test["__o"] = test["model"].map({m: i for i, m in enumerate(order)})
    save_table(test.sort_values("__o").drop(columns="__o"),
               "04_05_TABLE_heldout_test")

    print("\nChapter Four tables written to outputs/ with the 04_* prefix.")
    print("Each filename maps to the section that consumes it.")


if __name__ == "__main__":
    main()
