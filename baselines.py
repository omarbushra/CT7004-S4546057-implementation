"""Tabular baselines: logistic regression, random forest, XGBoost.

Implements Section 3.6. The baselines receive engineered aggregate features
approximating relational context (counts on the same card and device), so that
any remaining GNN advantage is attributable to learned relational structure
rather than to information the baselines were denied.

XGBoost is tuned with the same search budget as every other model. Benchmarking
a new architecture against a default-configured ensemble inflates the apparent
gain, which Section 2.3 argues is a live problem in this literature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (DATA, FAST, N_FOLDS, SEED, fold_summary, paired_tests,
                    save_table, score, set_seed)

SEARCH_BUDGET = 5 if FAST else 20   # configurations per model - equalised across all six


def relational_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hand-engineered approximations of what the graph gives the GNNs."""
    out = pd.DataFrame(index=df.index)
    g = df.groupby("card_id")
    out["card_txn_count"] = g["TransactionID"].transform("size")
    out["card_amt_mean"] = g["TransactionAmt"].transform("mean")
    out["card_amt_std"] = g["TransactionAmt"].transform("std").fillna(0)
    out["card_txn_rank"] = g["TransactionDT"].rank(method="first")
    out["card_seconds_since_prev"] = (
        df["TransactionDT"] - g["TransactionDT"].shift(1)).fillna(-1)

    d = df.groupby("device_id")
    out["device_txn_count"] = d["TransactionID"].transform("size").fillna(0)
    out["device_card_nunique"] = d["card_id"].transform("nunique").fillna(0)
    out["has_device"] = df["has_device"]
    return out.fillna(0)


def _grids(pos_weight: float) -> dict:
    rng = np.random.default_rng(SEED)
    return {
        "LogisticRegression": [
            {"C": float(c)} for c in rng.choice([0.01, 0.1, 1.0, 10.0], SEARCH_BUDGET)
        ],
        "RandomForest": [
            {"n_estimators": int(n), "max_depth": int(d), "min_samples_leaf": int(l)}
            for n, d, l in zip(rng.choice([200, 400, 800], SEARCH_BUDGET),
                               rng.choice([8, 16, 24, 32], SEARCH_BUDGET),
                               rng.choice([1, 5, 20], SEARCH_BUDGET))
        ],
        "XGBoost": [
            {"n_estimators": int(n), "max_depth": int(d),
             "learning_rate": float(lr), "subsample": float(s),
             "scale_pos_weight": pos_weight}
            for n, d, lr, s in zip(rng.choice([300, 600, 1000], SEARCH_BUDGET),
                                   rng.choice([4, 6, 8, 10], SEARCH_BUDGET),
                                   rng.choice([0.03, 0.05, 0.1], SEARCH_BUDGET),
                                   rng.choice([0.7, 0.85, 1.0], SEARCH_BUDGET))
        ],
    }


def _make(name: str, params: dict, pos_weight: float):
    if name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000, class_weight="balanced",
                                  random_state=SEED, **params)
    if name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=SEED, **params)
    from xgboost import XGBClassifier
    return XGBClassifier(tree_method="hist", eval_metric="aucpr",
                         n_jobs=-1, random_state=SEED, **params)


def tune(name: str, Xtr, ytr, Xva, yva, pos_weight: float) -> tuple[dict, float]:
    """Select by validation AUPRC. Equal budget for every model."""
    best, best_auprc = None, -1.0
    for params in _grids(pos_weight)[name]:
        m = _make(name, params, pos_weight).fit(Xtr, ytr)
        a = score(yva, m.predict_proba(Xva)[:, 1]).auprc
        if a > best_auprc:
            best, best_auprc = params, a
    print(f"  {name}: best val AUPRC={best_auprc:.4f}  {best}")
    return best, best_auprc


def run(df: pd.DataFrame, feat: list[str], use_smote: bool = False) -> pd.DataFrame:
    set_seed(SEED)
    X = pd.concat([df[feat], relational_features(df)], axis=1).to_numpy(np.float32)
    y = df["isFraud"].to_numpy()
    tr, va, te = ((df["split"] == s).to_numpy() for s in ("train", "val", "test"))

    pos_weight = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
    rows, chosen = [], {}

    for name in ("LogisticRegression", "RandomForest", "XGBoost"):
        params, _ = tune(name, X[tr], y[tr], X[va], y[va], pos_weight)
        chosen[name] = params

        # five-fold CV inside the training period (Section 3.7.3)
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        Xtr_all, ytr_all = X[tr], y[tr]
        for k, (a, b) in enumerate(skf.split(Xtr_all, ytr_all)):
            Xa, ya = Xtr_all[a], ytr_all[a]
            if use_smote and name != "LogisticRegression":
                from imblearn.over_sampling import SMOTE
                Xa, ya = SMOTE(random_state=SEED).fit_resample(Xa, ya)
            m = _make(name, params, pos_weight).fit(Xa, ya)
            s = score(ytr_all[b], m.predict_proba(Xtr_all[b])[:, 1])
            rows.append({"model": name, "fold": k, "stage": "cv", **s.as_dict()})

        # held out once, at the end (Section 3.7.3)
        m = _make(name, params, pos_weight).fit(X[tr], y[tr])
        s = score(y[te], m.predict_proba(X[te])[:, 1])
        rows.append({"model": name, "fold": -1, "stage": "test", **s.as_dict()})

    pd.DataFrame([{"model": k, **v} for k, v in chosen.items()]).to_csv(
        DATA / "baseline_params.csv", index=False)
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(DATA / "prepared.parquet")
    feat = pd.read_csv(DATA / "feature_columns.csv", header=None)[0].tolist()

    res = run(df, feat)
    save_table(res, "04_05_baseline_raw")
    cv = res[res.stage == "cv"].to_dict("records")
    save_table(fold_summary(cv), "04_05_baseline_cv_summary")
    save_table(res[res.stage == "test"], "04_05_baseline_test")
    save_table(paired_tests(cv, "auprc", "XGBoost"), "04_05_baseline_paired_tests")


if __name__ == "__main__":
    main()
