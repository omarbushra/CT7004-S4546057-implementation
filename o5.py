"""Objective O5: explainability evidence for the graph models and the baselines.

Run in two invocations so the graph and the baseline feature matrix are never
resident together:

    python o5.py gnn     # graph, three models, GAT attention, GNNExplainer
    python o5.py shap    # baseline refit, SHAP, impurity importances

Only prepared.parquet and feature_columns.csv are required. The selected
hyperparameters are those reported in Section 4.4 and are set out below rather
than read from file, so the run does not depend on artefacts of an earlier
session.

Every output is written the moment it exists and skipped if already present,
so an interrupted run resumes rather than restarting. Memory is the binding
constraint on a 12 GB runtime: feature columns are streamed in blocks into a
preallocated float32 array and the frame is never materialised.
"""
from __future__ import annotations

import gc
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PARQUET, FEATCOLS = "prepared.parquet", "feature_columns.csv"
SEED, COLBLOCK, TOP_K = 42, 64, 15
EXPLAIN_NODES, MAX_SUBGRAPH, SHAP_SAMPLE = 30, 20000, 2000

# Section 4.4, Table of selected configurations
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
RECORDED = {"GCN": 0.3957, "GAT": 0.4141, "GraphSAGE": 0.4231}


def seed_all():
    import random
    import torch
    random.seed(SEED); np.random.seed(SEED); os.environ["PYTHONHASHSEED"] = str(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


NON_FEATURE = {"TransactionID", "isFraud", "card_id", "card_id_ablation",
               "device_id", "has_device", "split"}


def feat_names():
    """The 731 model features, in the order prepare.py produced them.

    feature_columns.csv is authoritative when present. Without it the list is
    recovered from the parquet: prepare.py leaves the high-missingness columns
    it dropped in the frame, so exclusion alone is not enough - but every
    retained feature was median-imputed and standardised and therefore has no
    nulls, while every dropped column keeps its nulls. That separates them.
    """
    if os.path.exists(FEATCOLS):
        return pd.read_csv(FEATCOLS, header=None)[0].tolist()

    print(f"  {FEATCOLS} not found - recovering the feature list from the parquet",
          flush=True)
    pf = pq.ParquetFile(PARQUET)
    cand = [f.name for f in pf.schema_arrow if f.name not in NON_FEATURE]
    sample = pq.read_table(PARQUET, columns=cand).slice(0, 50000).to_pandas()
    feat = [c for c in cand
            if pd.api.types.is_numeric_dtype(sample[c]) and not sample[c].isna().any()]
    print(f"  recovered {len(feat)} features from {len(cand)} candidate columns",
          flush=True)
    if len(feat) != 731:
        print(f"  WARNING: expected 731, got {len(feat)}. Upload "
              f"feature_columns.csv to be certain the run matches the "
              f"dissertation.", flush=True)
    return feat


def stream_features(into, n_rows, feat):
    """Fill `into[:n_rows, :len(feat)]` from the parquet, one block at a time."""
    import torch
    for i in range(0, len(feat), COLBLOCK):
        cols = feat[i:i + COLBLOCK]
        tbl = pq.read_table(PARQUET, columns=cols)
        for j, c in enumerate(cols):
            v = tbl.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
            into[:n_rows, i + j] = torch.from_numpy(v)
        del tbl
        print(f"  features {min(i + COLBLOCK, len(feat))}/{len(feat)}",
              end="\r", flush=True)
    print()


# ───────────────────────────────────────────────────────────── graph models
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
    stream_features(x, n_tx, feat)
    x[:n_tx, d] = 1.0
    x[co:co + n_card, d + 1] = 1.0
    if n_dev:
        x[do_:do_ + n_dev, d + 2] = 1.0

    tx = np.arange(n_tx, dtype=np.int64)
    m = dev >= 0
    und = np.concatenate([np.stack([tx, cards + co]),
                          np.stack([tx[m], dev[m] + do_])], axis=1).astype(np.int64)
    ei = torch.tensor(np.concatenate([und, und[::-1]], axis=1), dtype=torch.long)
    y = torch.zeros(len(x), dtype=torch.long)
    y[:n_tx] = torch.tensor(meta["isFraud"].to_numpy())
    sp = meta["split"].astype(str).to_numpy()
    mk = {}
    for nm in ("train", "val", "test"):
        b = torch.zeros(len(y), dtype=torch.bool)
        b[:n_tx] = torch.from_numpy(np.asarray(sp == nm, dtype=bool))
        mk[nm] = b
    data = Data(x=x, edge_index=ei, y=y, train_mask=mk["train"],
                val_mask=mk["val"], test_mask=mk["test"]).to(DEV)
    data.n_tx, data.n_card = n_tx, n_card
    del meta, x, ei, y
    gc.collect()
    print(f"graph: {data.num_nodes:,} nodes, {data.edge_index.size(1)//2:,} edges",
          flush=True)

    s_, t_ = data.edge_index
    keep = (torch.where(s_ < n_tx, data.train_mask[s_], torch.ones_like(data.train_mask[s_]))
            & torch.where(t_ < n_tx, data.train_mask[t_], torch.ones_like(data.train_mask[t_])))
    tei = data.edge_index[:, keep]

    class GNN(torch.nn.Module):
        def __init__(self, kind, in_dim, hidden, layers, dropout, heads=4):
            super().__init__()
            self.dropout = dropout
            self.convs = torch.nn.ModuleList()
            dims = [in_dim] + [hidden] * layers
            for i in range(layers):
                a, b = dims[i], dims[i + 1]
                self.convs.append(GCNConv(a, b) if kind == "GCN" else
                                  SAGEConv(a, b) if kind == "GraphSAGE" else
                                  GATConv(a, b // heads, heads=heads, concat=True))
            self.out = torch.nn.Linear(hidden, 2)

        def forward(self, x, e):
            for conv in self.convs:
                x = F.dropout(F.relu(conv(x, e)), p=self.dropout, training=self.training)
            return self.out(x)

    n_pos = int(data.y[data.train_mask].sum())
    cw = torch.tensor([1.0, (int(data.train_mask.sum()) - n_pos) / n_pos],
                      dtype=torch.float, device=DEV)
    yv = data.y[data.val_mask].cpu().numpy()
    yt = data.y[data.test_mask].cpu().numpy()

    models, check = {}, []
    for kind, cfg in GNN_CFG.items():
        path = f"model_{kind}.pt"
        model = GNN(kind, data.x.size(1), cfg["hidden"], cfg["layers"],
                    cfg["dropout"]).to(DEV)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=DEV))
            print(f"  {kind}: reusing saved weights", flush=True)
        else:
            seed_all()
            opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=5e-4)
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
                a = average_precision_score(yv, pr[data.val_mask].cpu().numpy())
                if a > ba:
                    ba, bad = a, 0
                    best = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 10:
                        break
            model.load_state_dict(best)
            torch.save(model.state_dict(), path)
        model.eval()
        with torch.no_grad():
            pr = F.softmax(model(data.x, data.edge_index), 1)[:, 1]
        a = average_precision_score(yt, pr[data.test_mask].cpu().numpy())
        check.append({"model": kind, "auprc_reported": RECORDED[kind],
                      "auprc_this_run": round(float(a), 4),
                      "abs_difference": round(abs(a - RECORDED[kind]), 4)})
        models[kind] = model
        print(f"  {kind}: test AUPRC {a:.4f} (reported {RECORDED[kind]})", flush=True)
        torch.cuda.empty_cache()
    pd.DataFrame(check).to_csv("04_07_retrain_check.csv", index=False)

    # ---- GAT attention by edge source (Section 5.4) ----
    if not os.path.exists("04_07_attention_by_edge_source.csv"):
        g = models["GAT"]
        with torch.no_grad():
            _o, (e2, al) = g.convs[0](data.x, data.edge_index,
                                      return_attention_weights=True)
        a = al.mean(1).cpu()
        s2, t2 = e2[0].cpu(), e2[1].cpu()
        into = t2 < n_tx
        grp = {"self-loop (own features)": into & (s2 == t2),
               "from card": into & (s2 >= n_tx) & (s2 < n_tx + n_card),
               "from device": into & (s2 >= n_tx + n_card)}
        out = pd.DataFrame([
            {"edge_source": k, "n_edges": int(v.sum()),
             "mean_attention": round(float(a[v].mean()), 6) if v.any() else None,
             "sd_attention": round(float(a[v].std()), 6) if int(v.sum()) > 1 else None}
            for k, v in grp.items()])
        out.to_csv("04_07_attention_by_edge_source.csv", index=False)
        print("\nGAT attention by edge source:\n" + out.to_string(index=False), flush=True)
        del e2, al
        torch.cuda.empty_cache()

    # ---- GNNExplainer over fraudulent test nodes ----
    if not os.path.exists("04_07_gnnexplainer_top10.csv"):
        from torch_geometric.explain import Explainer, GNNExplainer
        from torch_geometric.utils import k_hop_subgraph
        names = feat + ["is_transaction_node", "is_card_node", "is_device_node"]
        fraud = ((data.y == 1) & data.test_mask).nonzero(as_tuple=True)[0][:EXPLAIN_NODES]
        rows = []
        for kind, model in models.items():
            ex = Explainer(model=model, algorithm=GNNExplainer(epochs=100),
                           explanation_type="model", node_mask_type="attributes",
                           edge_mask_type="object",
                           model_config=dict(mode="multiclass_classification",
                                             task_level="node", return_type="raw"))
            acc, used = torch.zeros(data.x.size(1)), 0
            for node in fraud:
                sub, sub_ei, mapping, _ = k_hop_subgraph(
                    int(node), num_hops=GNN_CFG[kind]["layers"],
                    edge_index=data.edge_index, relabel_nodes=True,
                    num_nodes=data.num_nodes)
                if sub.numel() > MAX_SUBGRAPH:
                    continue
                e = ex(data.x[sub], sub_ei, index=int(mapping[0]))
                acc += e.node_mask.abs().mean(dim=0).detach().cpu()
                used += 1
                del e
            torch.cuda.empty_cache()
            if not used:
                print(f"  {kind}: no node within subgraph cap", flush=True)
                continue
            acc /= used
            print(f"  GNNExplainer {kind}: {used}/{len(fraud)} nodes", flush=True)
            for rank, i in enumerate(torch.argsort(acc, descending=True)[:TOP_K].tolist(), 1):
                rows.append({"model": kind, "rank": rank, "feature": names[i],
                             "mean_mask_weight": round(float(acc[i]), 6),
                             "nodes_explained": used})
        pd.DataFrame(rows).to_csv("04_07_gnnexplainer_top10.csv", index=False)
    print("\ngnn stage complete", flush=True)


# ─────────────────────────────────────────────────────────────── baselines
def run_shap():
    from sklearn.metrics import average_precision_score
    seed_all()
    if os.path.exists("04_07_baseline_attribution.csv"):
        print("baseline attribution already present"); return

    feat = feat_names()
    meta = pq.read_table(PARQUET,
                         columns=["card_id", "device_id", "has_device", "isFraud",
                                  "split", "TransactionID", "TransactionAmt",
                                  "TransactionDT"]).to_pandas()
    tr = (meta["split"] == "train").to_numpy()
    te = (meta["split"] == "test").to_numpy()
    y = meta["isFraud"].to_numpy()

    # the eight relational aggregates supplied to the baselines (Section 3.6)
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
    print(f"features {len(feat)} + aggregates {len(rel.columns)} = {len(names)}",
          flush=True)

    X = np.empty((len(meta), len(names)), dtype=np.float32)
    stream_features(X, len(meta), feat)
    for j, c in enumerate(rel.columns):
        X[:, len(feat) + j] = rel[c].to_numpy(np.float32)
    del meta, rel, g, dv
    gc.collect()

    Xtr, ytr, Xte = X[tr], y[tr], X[te]
    del X
    gc.collect()
    pos_w = float((ytr == 0).sum() / (ytr == 1).sum())
    print(f"train {Xtr.shape}, test {Xte.shape}", flush=True)

    rows = []
    for name, cfg in BASE_CFG.items():
        print(f"  fitting {name} {cfg}", flush=True)
        if name == "LogisticRegression":
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=SEED, **cfg)
        elif name == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier
            m = RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1,
                                       random_state=SEED, **cfg)
        else:
            from xgboost import XGBClassifier
            m = XGBClassifier(tree_method="hist", eval_metric="aucpr", n_jobs=-1,
                              random_state=SEED, scale_pos_weight=pos_w, **cfg)
        m.fit(Xtr, ytr)
        pr = m.predict_proba(Xte)[:, 1]
        print(f"    AUPRC {average_precision_score(y[te], pr):.4f}", flush=True)

        if hasattr(m, "feature_importances_"):
            imp = m.feature_importances_
            for rank, i in enumerate(np.argsort(imp)[::-1][:TOP_K], 1):
                rows.append({"model": name, "method": "impurity", "rank": rank,
                             "feature": names[i],
                             "importance": round(float(imp[i]), 6)})
        if name == "XGBoost":
            import shap
            idx = np.random.default_rng(SEED).choice(
                len(Xte), min(SHAP_SAMPLE, len(Xte)), replace=False)
            v = shap.TreeExplainer(m).shap_values(Xte[idx])
            if isinstance(v, list):
                v = v[1]
            if v.ndim == 3:
                v = v[:, :, 1]
            sv = np.abs(v).mean(axis=0)
            for rank, i in enumerate(np.argsort(sv)[::-1][:TOP_K], 1):
                rows.append({"model": name, "method": "shap", "rank": rank,
                             "feature": names[i],
                             "importance": round(float(sv[i]), 6)})
            print(f"    SHAP over {len(idx)} test transactions", flush=True)
        del m
        gc.collect()
        pd.DataFrame(rows).to_csv("04_07_baseline_attribution.csv", index=False)
    print("\nshap stage complete", flush=True)


if __name__ == "__main__":
    {"gnn": run_gnn, "shap": run_shap}[sys.argv[1]]()
