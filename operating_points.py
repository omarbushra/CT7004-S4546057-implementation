"""Matched-operating-point comparison, alert-budget analysis and SHAP.

Reconstructed for this package; see PROVENANCE.md, Section 6. The behaviour
here matches what logs/08_operating_points.log records.

Why this exists as a separate script. Tables 5 and 6 compare models at a 0.5
decision threshold, which is not how a fraud system is run: each model's scores
are differently calibrated, so a fixed threshold puts the six models at six
different false-positive rates and the recall column compares nothing. This
script re-evaluates all six at matched false-positive rates and under fixed
alert budgets, which is the comparison Section 4.5 relies on.

The baseline models were not serialised during the `baselines` stage, so they
are refitted here from the recorded hyperparameters. The difference between
refitted and recorded AUPRC is written to 04_05_reproducibility_check.csv and
discussed in PROVENANCE.md, Section 4.

    PYTHONUNBUFFERED=1 python colab/operating_points.py

Memory note: the 739-column design matrix at 413,378 rows is 1.2 GB in float32
and the host has 12.7 GB. Building it with np.concatenate holds three copies at
once and is killed. It is therefore preallocated and filled column by column,
XGBoost is fitted first while headroom is widest, and the GNN probabilities are
cached to .npy so a rerun need not reload the graph.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch

from common import SEED, set_seed, score, threshold_at_fpr, save_table
import baselines as B
import gnns

FPR_TARGETS = (0.001, 0.005, 0.010, 0.050)
ALERT_BUDGETS = (0.005, 0.010)          # share of transactions a team can review
SHAP_SAMPLE = 2000
TOP_K = 15

RECORDED = {                            # from outputs/04_05_*_test.csv
    "GCN": 0.395745, "GAT": 0.414050, "GraphSAGE": 0.423057,
    "XGBoost": 0.563114, "RandomForest": 0.494528, "LogisticRegression": 0.169762,
}
BEST = {                                # from logs/03_baselines.log
    "XGBoost": dict(n_estimators=1000, max_depth=8, learning_rate=0.05,
                    subsample=0.7, scale_pos_weight=27.434310083918007),
    "RandomForest": dict(n_estimators=200, max_depth=32, min_samples_leaf=5),
    "LogisticRegression": dict(C=1.0),
}


def rss_gb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def gnn_probabilities(y_test):
    """Restore the three graph models from saved weights and score the test set."""
    probs = {}
    cache = Path("outputs/probs_gnn.npz")
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}

    data = torch.load("outputs/graph.pt", weights_only=False)
    print(f"GNN probabilities (RSS {rss_gb():.2f} GB)")
    for name in ("GCN", "GAT", "GraphSAGE"):
        state = torch.load(f"outputs/model_{name}.pt", map_location=gnns.DEVICE)
        model = gnns.GNN.from_state(name, state, data.num_features).to(gnns.DEVICE)
        p = gnns.predict(model, data, data.edge_index)
        probs[name] = p[data.test_mask].cpu().numpy()
        print(f"  {name}: {len(probs[name]):,} probabilities")
        del model
        torch.cuda.empty_cache()
    del data
    np.savez(cache, **probs)
    return probs


def design_matrix(df, feat, rel, mask) -> np.ndarray:
    """Preallocate and fill column by column. np.concatenate would hold three
    copies of a 1.2 GB array simultaneously and the process is killed."""
    cols = feat + list(rel.columns)
    X = np.empty((int(mask.sum()), len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        src = df[c] if c in df.columns else rel[c]
        X[:, j] = src.to_numpy(np.float32)[mask]
    return X


def main() -> None:
    set_seed(SEED)
    df = pd.read_parquet("outputs/prepared.parquet")
    feat = pd.read_csv("outputs/feature_columns.csv", header=None)[0].tolist()
    y = df["isFraud"].to_numpy(np.int8)
    test = (df["split"] == "test").to_numpy()
    train = (df["split"] == "train").to_numpy()

    probs = gnn_probabilities(y[test])

    print(f"\nbaselines (RSS {rss_gb():.2f} GB)")
    rel = B.relational_features(df)
    Xtr = design_matrix(df, feat, rel, train)
    Xte = design_matrix(df, feat, rel, test)
    print(f"  train {Xtr.shape}, test {Xte.shape}, RSS {rss_gb():.2f} GB")
    del rel

    attribution, pos_weight = [], (y[train] == 0).sum() / max((y[train] == 1).sum(), 1)
    # XGBoost first: it is the largest fit, and headroom is widest now.
    for name in ("XGBoost", "RandomForest", "LogisticRegression"):
        print(f"  fitting {name} {BEST[name]}")
        model = B._make(name, BEST[name], pos_weight)
        model.fit(Xtr, y[train])
        probs[name] = model.predict_proba(Xte)[:, 1]
        print(f"    AUPRC {score(y[test], probs[name]).auprc:.4f}, RSS {rss_gb():.2f} GB")

        if name == "XGBoost":
            print(f"    SHAP over {SHAP_SAMPLE} transactions")
            import shap
            idx = np.random.default_rng(SEED).choice(len(Xte), SHAP_SAMPLE, replace=False)
            values = shap.TreeExplainer(model).shap_values(Xte[idx])
            rank = np.abs(values).mean(0)
        elif name == "RandomForest":
            rank = model.feature_importances_          # impurity decrease
        else:
            rank = np.abs(model.coef_[0])              # standardised inputs

        cols = feat + [c for c in B.relational_features(df.head(1)).columns]
        order = np.argsort(rank)[::-1][:TOP_K]
        attribution += [dict(model=name, rank=r + 1, feature=cols[i],
                             importance=float(rank[i])) for r, i in enumerate(order)]
        del model

    save_table(pd.DataFrame(attribution), "04_07_baseline_attribution")

    save_table(pd.DataFrame([
        dict(model=m, auprc_stored=RECORDED[m],
             auprc_recomputed=score(y[test], probs[m]).auprc,
             abs_difference=abs(RECORDED[m] - score(y[test], probs[m]).auprc))
        for m in RECORDED]), "04_05_reproducibility_check")

    order = ["LogisticRegression", "RandomForest", "XGBoost", "GCN", "GAT", "GraphSAGE"]
    yt, n_fraud = y[test], int(y[test].sum())

    matched = []
    for m in order:
        for target in FPR_TARGETS:
            thr = threshold_at_fpr(yt, probs[m], target)
            pred = probs[m] >= thr
            tp, fp = int((pred & (yt == 1)).sum()), int((pred & (yt == 0)).sum())
            matched.append(dict(
                model=m, target_fpr=target, threshold=thr,
                achieved_fpr=fp / (yt == 0).sum(), recall=tp / n_fraud,
                precision=tp / max(tp + fp, 1), fraud_caught=tp, fraud_total=n_fraud))
    save_table(pd.DataFrame(matched), "04_05_TABLE_matched_fpr")

    budget_rows = []
    for m in order:
        for b in ALERT_BUDGETS:
            k = int(round(b * len(yt)))
            top = np.argpartition(probs[m], -k)[-k:]
            tp = int(yt[top].sum())
            budget_rows.append(dict(
                model=m, alert_budget=b, alerts=k, fraud_caught=tp,
                recall_at_budget=tp / n_fraud, precision_at_budget=tp / k))
    save_table(pd.DataFrame(budget_rows), "04_05_TABLE_alert_budget")
    print("done")


if __name__ == "__main__":
    main()
