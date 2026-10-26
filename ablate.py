"""TransactionDT ablation: how much of the decline is drift, and how much is
extrapolation failure on a time index?

Section 4.7 reports that TransactionDT ranks first for XGBoost on SHAP and for
all three graph models on GNNExplainer, and fourth for random forest. Under
the temporal split of Section 3.5.1 every TransactionDT value in the test
partition lies outside the range seen in training. A tree ensemble cannot
extrapolate past its terminal split on such a feature and a standardised
linear extrapolation is no better justified, so a model leaning on it will
lose accuracy across the boundary whether or not the fraud patterns changed.
That is an alternative explanation for part of the decline Section 4.5
attributes to concept drift, and this script measures it.

    python ablate.py base    # three baselines, both conditions
    python ablate.py gnn     # three graph models, both conditions

WHY BOTH CONDITIONS RUN HERE. The obvious shortcut is to refit without
TransactionDT and compare against the figures already in Chapter Four. That
would be wrong: those figures come from a different fitting run, and
re-execution alone moves the baselines by up to 0.010 AUPRC and the graph
models by up to 0.007 (Section 4.7). The ablation effect and the re-execution
noise would be inseparable. So both conditions are fitted here, back to back,
on one code path with one seed. Only the feature differs, so the difference
between the two columns is the feature.

WHAT IS ABLATED. TransactionDT only. The D1-D15 timedelta features and the
card_seconds_since_prev aggregate are differences rather than absolute
positions, so their test values fall inside the training range and they carry
no extrapolation problem. card_txn_rank is bounded by card size, likewise.
TransactionDT is the only feature whose support moves wholly outside the
training range at the boundary, so it is the one the confound concerns.

HOW IT IS ABLATED. The column is zeroed in place rather than deleted, which
avoids holding a second 1.2 GB design matrix on a 12.7 GB runtime. Zeroing is
equivalent to removal for the models used here: a constant column admits no
split in a tree ensemble, and in logistic regression its contribution is a
constant absorbed by the intercept. Features are standardised, so zero is the
training mean.

WHAT TO READ. The decline reported is validation to test, the comparison
Section 4.5 adopted: both are out-of-sample and both follow the training
window, so model-selection optimism is common to them and cancels. If the
decline shrinks materially without TransactionDT, part of what Chapter Four
calls drift was extrapolation failure. If it does not, the drift reading
survives this challenge.
"""
from __future__ import annotations

import gc
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PARQUET, FEATCOLS = "prepared.parquet", "feature_columns.csv"
SEED, COLBLOCK = 42, 64
ABLATE = "TransactionDT"
OUT = "04_08_transactiondt_ablation.csv"

GNN_CFG = {
    "GCN":       dict(hidden=256, layers=2, dropout=0.5, lr=0.005),
    "GraphSAGE": dict(hidden=128, layers=3, dropout=0.5, lr=0.001),
    "GAT":       dict(hidden=64,  layers=2, dropout=0.5, lr=0.005),
}
BASE_CFG = {
    "XGBoost": dict(n_estimators=1000, max_depth=8, learning_rate=0.05,
                    subsample=0.7),
    "RandomForest": dict(n_estimators=200, max_depth=32, min_samples_leaf=5),
    "LogisticRegression": dict(C=1.0),
}
NON_FEATURE = {"TransactionID", "isFraud", "card_id", "card_id_ablation",
               "device_id", "has_device", "split"}


def seed_all():
    import random
    import torch
    random.seed(SEED); np.random.seed(SEED); os.environ["PYTHONHASHSEED"] = str(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


def feat_names():
    if os.path.exists(FEATCOLS):
        return pd.read_csv(FEATCOLS, header=None)[0].tolist()
    pf = pq.ParquetFile(PARQUET)
    cand = [f.name for f in pf.schema_arrow if f.name not in NON_FEATURE]
    sample = pq.read_table(PARQUET, columns=cand).slice(0, 50000).to_pandas()
    feat = [c for c in cand
            if pd.api.types.is_numeric_dtype(sample[c]) and not sample[c].isna().any()]
    if len(feat) != 731:
        print(f"  WARNING: expected 731 features, got {len(feat)}", flush=True)
    return feat


def stream_features(into, n_rows, feat, torch_target=False):
    """Fill `into[:n_rows, :len(feat)]` from the parquet, one block at a time."""
    import torch
    for i in range(0, len(feat), COLBLOCK):
        cols = feat[i:i + COLBLOCK]
        tbl = pq.read_table(PARQUET, columns=cols)
        for j, c in enumerate(cols):
            v = tbl.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
            into[:n_rows, i + j] = torch.from_numpy(v) if torch_target else v
        del tbl
        print(f"  features {min(i + COLBLOCK, len(feat))}/{len(feat)}",
              end="\r", flush=True)
    print()


def report(rows):
    """Print the comparison and write it, in the shape Section 4.5 reports."""
    df = pd.DataFrame(rows)
    wide = df.pivot(index="model", columns="condition",
                    values=["val_auprc", "test_auprc", "decline_pct"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["decline_change_pp"] = (wide["decline_pct_with_dt"]
                                 - wide["decline_pct_without_dt"]).round(1)
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        wide = pd.concat([prev[~prev["model"].isin(wide["model"])], wide],
                         ignore_index=True)
    wide.to_csv(OUT, index=False)
    print(f"\n  -> {OUT}\n")
    print(wide.to_string(index=False))
    print("\n  decline_change_pp > 0 means the decline SHRANK when TransactionDT")
    print("  was removed, i.e. that much of the measured degradation was")
    print("  extrapolation on the time index rather than concept drift.")


# ────────────────────────────────────────────────────────────── baselines
def run_base():
    from sklearn.metrics import average_precision_score
    seed_all()

    feat = feat_names()
    meta = pq.read_table(PARQUET,
                         columns=["card_id", "device_id", "has_device", "isFraud",
                                  "split", "TransactionID", "TransactionAmt",
                                  "TransactionDT"]).to_pandas()
    tr = (meta["split"] == "train").to_numpy()
    va = (meta["split"] == "val").to_numpy()
    te = (meta["split"] == "test").to_numpy()
    y = meta["isFraud"].to_numpy()

    # the eight relational aggregates of Section 3.4, as in o5.py
    for c in ("card_id", "device_id"):
        meta[c] = meta[c].astype("category")
    g = meta.groupby("card_id", observed=True)
    dv = meta.groupby("device_id", observed=True)
    rel = pd.DataFrame(index=meta.index)
    rel["card_txn_count"] = g["TransactionID"].transform("size")
    rel["card_amt_mean"] = g["TransactionAmt"].transform("mean")
    rel["card_amt_std"] = g["TransactionAmt"].transform("std").fillna(0)
    rel["card_txn_rank"] = g["TransactionDT"].rank(method="first")
    rel["card_seconds_since_prev"] = (
        meta["TransactionDT"] - g["TransactionDT"].shift(1)).fillna(-1)
    rel["device_txn_count"] = dv["TransactionID"].transform("size").fillna(0)
    rel["device_card_nunique"] = dv["card_id"].transform("nunique").fillna(0)
    rel["has_device"] = meta["has_device"]
    rel = rel.fillna(0)
    names = feat + list(rel.columns)

    if ABLATE not in names:
        sys.exit(f"{ABLATE} is not in the design matrix; nothing to ablate.")
    dt_col = names.index(ABLATE)
    print(f"features {len(feat)} + aggregates {len(rel.columns)} = {len(names)}; "
          f"{ABLATE} at column {dt_col}", flush=True)

    X = np.empty((len(meta), len(names)), dtype=np.float32)
    stream_features(X, len(meta), feat)
    for j, c in enumerate(rel.columns):
        X[:, len(feat) + j] = rel[c].to_numpy(np.float32)
    del meta, rel, g, dv
    gc.collect()

    Xtr, Xva, Xte = X[tr], X[va], X[te]
    ytr, yva, yte = y[tr], y[va], y[te]
    del X
    gc.collect()
    pos_w = float((ytr == 0).sum() / (ytr == 1).sum())
    print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}", flush=True)

    rows = []
    for condition in ("with_dt", "without_dt"):
        if condition == "without_dt":
            # see the module docstring: zeroing is equivalent to removal here
            for A in (Xtr, Xva, Xte):
                A[:, dt_col] = 0.0
            print(f"\n--- {ABLATE} removed ---", flush=True)
        else:
            print(f"\n--- {ABLATE} present ---", flush=True)

        for name, cfg in BASE_CFG.items():
            seed_all()
            if name == "LogisticRegression":
                from sklearn.linear_model import LogisticRegression
                m = LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=SEED, **cfg)
            elif name == "RandomForest":
                from sklearn.ensemble import RandomForestClassifier
                m = RandomForestClassifier(class_weight="balanced_subsample",
                                           n_jobs=-1, random_state=SEED, **cfg)
            else:
                from xgboost import XGBClassifier
                m = XGBClassifier(scale_pos_weight=pos_w, n_jobs=-1,
                                  tree_method="hist", eval_metric="aucpr",
                                  random_state=SEED, **cfg)
            m.fit(Xtr, ytr)
            v = average_precision_score(yva, m.predict_proba(Xva)[:, 1])
            t = average_precision_score(yte, m.predict_proba(Xte)[:, 1])
            d = round(100 * (v - t) / v, 1)
            print(f"  {name:19s} val {v:.4f}  test {t:.4f}  decline {d:5.1f}%",
                  flush=True)
            rows.append(dict(model=name, condition=condition,
                             val_auprc=round(v, 4), test_auprc=round(t, 4),
                             decline_pct=d))
            del m
            gc.collect()

    report(rows)


# ──────────────────────────────────────────────────────────── graph models
def run_gnn():
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv

    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={DEV}", flush=True)
    seed_all()

    feat = feat_names()
    if ABLATE not in feat:
        sys.exit(f"{ABLATE} is not among the node features; nothing to ablate.")
    dt_col = feat.index(ABLATE)

    meta = pq.read_table(PARQUET,
                         columns=["card_id", "device_id", "isFraud", "split"]
                         ).to_pandas()
    n_tx, d = len(meta), len(feat)
    cards = meta["card_id"].astype("category").cat.codes.to_numpy().astype(np.int64)
    n_card = int(cards.max()) + 1
    dev = meta["device_id"].astype("category").cat.codes.to_numpy().astype(np.int64)
    n_dev = int(dev.max()) + 1 if (dev >= 0).any() else 0
    co, do_ = n_tx, n_tx + n_card

    x = torch.zeros((n_tx + n_card + n_dev, d + 3), dtype=torch.float)
    stream_features(x, n_tx, feat, torch_target=True)
    x[:n_tx, d] = 1.0
    x[co:co + n_card, d + 1] = 1.0
    if n_dev:
        x[do_:do_ + n_dev, d + 2] = 1.0

    tx = np.arange(n_tx, dtype=np.int64)
    m = dev >= 0
    und = np.concatenate([np.stack([tx, cards + co]),
                          np.stack([tx[m], dev[m] + do_])], axis=1).astype(np.int64)
    ei = torch.from_numpy(np.concatenate([und, und[::-1]], axis=1).copy())

    yv_all = torch.zeros(x.size(0), dtype=torch.long)
    yv_all[:n_tx] = torch.from_numpy(meta["isFraud"].to_numpy()).long()
    split = meta["split"].to_numpy()
    def mask(which):
        mm = torch.zeros(x.size(0), dtype=torch.bool)
        mm[:n_tx] = torch.from_numpy(split == which)
        return mm
    data = Data(x=x, edge_index=ei, y=yv_all)
    data.train_mask, data.val_mask, data.test_mask = mask("train"), mask("val"), mask("test")
    print(f"graph: {x.size(0):,} nodes, {und.shape[1]:,} edges", flush=True)
    del meta
    gc.collect()

    # training-only edges: Section 3.3 leakage control
    keep = data.train_mask[ei[0]] | (ei[0] >= n_tx)
    keep &= data.train_mask[ei[1]] | (ei[1] >= n_tx)
    tei = ei[:, keep]

    class GNN(torch.nn.Module):
        def __init__(self, kind, in_dim, hidden, layers, dropout, heads=4):
            super().__init__()
            self.dropout = dropout
            Conv = {"GCN": GCNConv, "GraphSAGE": SAGEConv, "GAT": GATConv}[kind]
            self.convs = torch.nn.ModuleList()
            dims = [in_dim] + [hidden] * layers
            for a, b in zip(dims[:-1], dims[1:]):
                self.convs.append(Conv(a, b // heads, heads=heads)
                                  if kind == "GAT" else Conv(a, b))
            self.out = torch.nn.Linear(hidden, 2)

        def forward(self, xx, e):
            for c in self.convs:
                xx = F.relu(c(xx, e))
                xx = F.dropout(xx, p=self.dropout, training=self.training)
            return self.out(xx)

    data = data.to(DEV); tei = tei.to(DEV)
    n_pos = int(data.y[data.train_mask].sum())
    cw = torch.tensor([1.0, (int(data.train_mask.sum()) - n_pos) / n_pos],
                      dtype=torch.float, device=DEV)
    yva = data.y[data.val_mask].cpu().numpy()
    yte = data.y[data.test_mask].cpu().numpy()

    rows = []
    for condition in ("with_dt", "without_dt"):
        if condition == "without_dt":
            data.x[:, dt_col] = 0.0
            print(f"\n--- {ABLATE} removed ---", flush=True)
        else:
            print(f"\n--- {ABLATE} present ---", flush=True)

        for kind, cfg in GNN_CFG.items():
            seed_all()
            model = GNN(kind, data.x.size(1), cfg["hidden"], cfg["layers"],
                        cfg["dropout"]).to(DEV)
            opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                   weight_decay=5e-4)
            best, ba, bad = None, -1.0, 0
            for _ in range(80):
                model.train(); opt.zero_grad()
                ce = F.cross_entropy(model(data.x, tei)[data.train_mask],
                                     data.y[data.train_mask], weight=cw,
                                     reduction="none")
                (((1 - torch.exp(-ce)) ** 2) * ce).mean().backward()
                opt.step()
                model.eval()
                with torch.no_grad():
                    pr = F.softmax(model(data.x, data.edge_index), 1)[:, 1]
                a = average_precision_score(yva, pr[data.val_mask].cpu().numpy())
                if a > ba:
                    ba, bad = a, 0
                    best = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 10:
                        break
            model.load_state_dict(best)
            model.eval()
            with torch.no_grad():
                pr = F.softmax(model(data.x, data.edge_index), 1)[:, 1]
            v = ba
            t = average_precision_score(yte, pr[data.test_mask].cpu().numpy())
            d = round(100 * (v - t) / v, 1)
            print(f"  {kind:19s} val {v:.4f}  test {t:.4f}  decline {d:5.1f}%",
                  flush=True)
            rows.append(dict(model=kind, condition=condition,
                             val_auprc=round(v, 4), test_auprc=round(t, 4),
                             decline_pct=d))
            del model, opt
            torch.cuda.empty_cache()
            gc.collect()

    report(rows)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("base", "gnn"):
        sys.exit(__doc__.strip().splitlines()[0] +
                 "\n\nusage: python ablate.py base | gnn")
    {"base": run_base, "gnn": run_gnn}[sys.argv[1]]()
