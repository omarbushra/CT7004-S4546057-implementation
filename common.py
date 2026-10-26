"""Shared configuration, seeding, metrics and IO.

CT7004 dissertation - GNN vs tabular baselines for credit card fraud detection.
Every number this pipeline produces lands in outputs/ as CSV so it can be
pasted directly into Chapter Four.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"          # put train_transaction.csv + train_identity.csv here
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# ── FAST mode ───────────────────────────────────────────────────────────────
# FAST=1 python run_all.py
# Cuts the hyperparameter search, caps epochs and optionally subsamples the
# data, turning a multi-day run into a few hours. Section 3.8 already lists
# computational constraint as limitation three, and Section 1.5 pre-commits to
# dropping GAT first if resources bind -- so a reduced run is defensible
# PROVIDED Chapter Four states exactly what was reduced.
FAST = os.environ.get("FAST", "0") == "1"
SKIP_GAT = os.environ.get("SKIP_GAT", "0") == "1"
DATA_FRACTION = float(os.environ.get("DATA_FRACTION", "1.0"))

SEED = 42
N_FOLDS = 3 if FAST else 5
LATENCY_BUDGET_MS = 100.0          # Section 3.7.5
SPLIT = (0.70, 0.15, 0.15)         # temporal train/val/test - Section 3.7.1
SUBSAMPLE_FRACTIONS = (0.25, 0.50, 1.00) if FAST else (0.25, 0.50, 0.75, 1.00)   # Section 3.7.5 scalability


def set_seed(seed: int = SEED) -> None:
    """Fix every RNG we touch. Reliability, Section 3.8."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


# ──────────────────────────────────────────────────────────── metrics
@dataclass
class Scores:
    """The six measures of Table 4, plus the count needed to sanity-check FPR."""
    recall: float
    precision: float
    f1: float
    auprc: float
    auroc: float
    fpr: float
    n_pos: int
    n_neg: int

    def as_dict(self) -> dict:
        return asdict(self)


def score(y_true, y_prob, threshold: float = 0.5) -> Scores:
    """Compute Table 4 metrics.

    AUPRC is primary (Section 2.7); AUROC is retained only for comparability
    with prior studies, because it incorporates the overwhelmingly numerous
    true negatives and flatters models under severe imbalance.
    """
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 f1_score, precision_score, recall_score,
                                 roc_auc_score)

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return Scores(
        recall=recall_score(y_true, y_pred, zero_division=0),
        precision=precision_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        auprc=average_precision_score(y_true, y_prob),
        auroc=roc_auc_score(y_true, y_prob),
        fpr=float(fp / (fp + tn)) if (fp + tn) else 0.0,
        n_pos=int(tp + fn),
        n_neg=int(tn + fp),
    )


def threshold_at_fpr(y_true, y_prob, target_fpr: float) -> float:
    """Operating threshold at a tolerable false-positive rate.

    Section 2.7: nobody deploys at equal error rate. A bank sets the threshold
    by the customer friction it can absorb, so Chapter Four should report
    detection at a fixed FPR as well as at 0.5.
    """
    from sklearn.metrics import roc_curve
    fpr, _tpr, thr = roc_curve(np.asarray(y_true).astype(int), y_prob)
    idx = int(np.searchsorted(fpr, target_fpr, side="right") - 1)
    return float(thr[max(idx, 0)])


# ──────────────────────────────────────────────────────────── io
def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    path = OUT / f"{name}.csv"
    df.to_csv(path, index=index)
    print(f"  -> {path.relative_to(ROOT)}  ({len(df)} rows)")
    return path


def save_json(obj: dict, name: str) -> Path:
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    print(f"  -> {path.relative_to(ROOT)}")
    return path


def fold_summary(rows: list[dict], group_keys=("model",)) -> pd.DataFrame:
    """Mean and standard deviation across folds.

    Section 3.7.4: differences smaller than the variation between folds do not
    support claims of superiority, so point estimates alone are not reported.
    """
    df = pd.DataFrame(rows)
    metrics = ["recall", "precision", "f1", "auprc", "auroc", "fpr"]
    agg = df.groupby(list(group_keys))[metrics].agg(["mean", "std"]).round(4)
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def paired_tests(rows: list[dict], metric: str = "auprc",
                 reference: str = "XGBoost") -> pd.DataFrame:
    """Paired comparison of every model against the reference, across folds.

    Wilcoxon signed-rank is used rather than a paired t-test because five folds
    is too few to rely on normality. With n=5 the test is weak; that weakness is
    reported rather than hidden.
    """
    from scipy.stats import wilcoxon

    df = pd.DataFrame(rows).pivot_table(index="fold", columns="model", values=metric)
    if reference not in df.columns:
        raise KeyError(f"reference model {reference!r} not in {list(df.columns)}")

    out = []
    for model in df.columns:
        if model == reference:
            continue
        a, b = df[model].to_numpy(), df[reference].to_numpy()
        try:
            stat, p = wilcoxon(a, b)
        except ValueError:          # identical vectors
            stat, p = float("nan"), 1.0
        out.append({
            "model": model,
            "reference": reference,
            "metric": metric,
            "mean_difference": round(float(np.mean(a - b)), 4),
            "statistic": stat,
            "p_value": round(float(p), 4),
            "n_folds": len(a),
        })
    return pd.DataFrame(out)


def run_profile() -> dict:
    """What was actually run. Chapter Four must report this verbatim."""
    return {
        "fast_mode": FAST,
        "data_fraction": DATA_FRACTION,
        "gat_skipped": SKIP_GAT,
        "cv_folds": N_FOLDS,
        "scaling_fractions": list(SUBSAMPLE_FRACTIONS),
        "seed": SEED,
    }
