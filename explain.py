"""Explainability: SHAP for the baselines, GNNExplainer and attention for GNNs.

Implements Section 3.7.6 and Objective O5. Beyond description this is
evaluative: if the graph models outperform the baselines, checking whether the
influential relationships are the relational patterns hypothesised in Chapter
Two tells you whether any improvement arises for the expected reasons or for
accidental ones.

Note the limitation argued in Section 2.8 - an attention weight indicates which
neighbour influenced a prediction, not why that influence is legitimate. Report
the weights; do not over-claim what they establish.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from common import DATA, SEED, save_table, set_seed

TOP_K = 10
SAMPLE = 2000          # SHAP on the full test split is prohibitive


def shap_baselines(df: pd.DataFrame, feat: list[str]) -> pd.DataFrame:
    import shap
    from baselines import relational_features, _make

    X = pd.concat([df[feat], relational_features(df)], axis=1)
    y = df["isFraud"].to_numpy()
    tr = (df["split"] == "train").to_numpy()
    te = (df["split"] == "test").to_numpy()

    params = pd.read_csv(DATA / "baseline_params.csv").set_index("model")
    rows = []
    for name in ("RandomForest", "XGBoost"):
        p = params.loc[name].dropna().to_dict()
        p = {k: (int(v) if float(v).is_integer() else float(v))
             for k, v in p.items() if k != "model"}
        model = _make(name, p, 1.0).fit(X[tr], y[tr])

        sample = X[te].sample(min(SAMPLE, int(te.sum())), random_state=SEED)
        vals = shap.TreeExplainer(model).shap_values(sample)
        if isinstance(vals, list):
            vals = vals[1]
        imp = np.abs(vals).mean(axis=0)
        order = np.argsort(imp)[::-1][:TOP_K]
        for rank, i in enumerate(order, 1):
            rows.append({"model": name, "rank": rank,
                         "feature": sample.columns[i],
                         "mean_abs_shap": round(float(imp[i]), 6)})
    return pd.DataFrame(rows)


def gnn_explanations(data, kinds=("GCN", "GAT", "GraphSAGE"), n_nodes: int = 50) -> pd.DataFrame:
    """GNNExplainer over a sample of correctly-flagged fraudulent transactions."""
    from torch_geometric.explain import Explainer, GNNExplainer
    from gnns import GNN, DEVICE

    params = pd.read_csv(DATA / "gnn_params.csv").set_index("model")
    fraud = ((data.y == 1) & data.test_mask).nonzero(as_tuple=True)[0][:n_nodes]

    rows = []
    for kind in kinds:
        p = params.loc[kind]
        model = GNN(kind, data.x.size(1), int(p["hidden"]), int(p["layers"]),
                    float(p["dropout"])).to(DEVICE)
        model.load_state_dict(torch.load(DATA / f"model_{kind}.pt",
                                         map_location=DEVICE))
        model.eval()

        explainer = Explainer(
            model=model, algorithm=GNNExplainer(epochs=100),
            explanation_type="model",
            node_mask_type="attributes", edge_mask_type="object",
            model_config=dict(mode="multiclass_classification",
                              task_level="node", return_type="raw"),
        )
        acc = torch.zeros(data.x.size(1))
        for node in fraud:
            e = explainer(data.x, data.edge_index, index=int(node))
            acc += e.node_mask.abs().mean(dim=0).detach().cpu()
        acc /= max(len(fraud), 1)

        order = torch.argsort(acc, descending=True)[:TOP_K]
        for rank, i in enumerate(order.tolist(), 1):
            rows.append({"model": kind, "rank": rank, "feature_index": i,
                         "mean_mask_weight": round(float(acc[i]), 6)})
    return pd.DataFrame(rows)


def attention_weights(data, n_nodes: int = 200) -> pd.DataFrame:
    """GAT attention over card edges versus device edges.

    The question worth answering in Chapter Five: does the model place more
    weight on shared-card relationships or shared-device ones? That is a claim
    about which relational pattern carries the fraud signal, and it is directly
    comparable to the hypothesis in Section 2.4.
    """
    from torch_geometric.nn import GATConv
    from gnns import GNN, DEVICE

    params = pd.read_csv(DATA / "gnn_params.csv").set_index("model").loc["GAT"]
    model = GNN("GAT", data.x.size(1), int(params["hidden"]),
                int(params["layers"]), float(params["dropout"])).to(DEVICE)
    model.load_state_dict(torch.load(DATA / "model_GAT.pt", map_location=DEVICE))
    model.eval()

    conv: GATConv = model.convs[0]
    with torch.no_grad():
        _out, (ei, alpha) = conv(data.x, data.edge_index,
                                 return_attention_weights=True)
    a = alpha.mean(dim=1).cpu()
    dst = ei[1].cpu()
    is_card = (dst >= data.n_tx) & (dst < data.n_tx + data.n_card)
    is_dev = dst >= data.n_tx + data.n_card

    return pd.DataFrame([
        {"edge_type": "transaction-card", "n_edges": int(is_card.sum()),
         "mean_attention": round(float(a[is_card].mean()), 6)},
        {"edge_type": "transaction-device", "n_edges": int(is_dev.sum()),
         "mean_attention": round(float(a[is_dev].mean()), 6) if is_dev.any() else None},
    ])


def main() -> None:
    set_seed(SEED)
    df = pd.read_parquet(DATA / "prepared.parquet")
    feat = pd.read_csv(DATA / "feature_columns.csv", header=None)[0].tolist()
    data = torch.load(DATA / "graph.pt", weights_only=False)

    print("SHAP (baselines)")
    save_table(shap_baselines(df, feat), "04_07_shap_top10")

    print("GNNExplainer")
    save_table(gnn_explanations(data), "04_07_gnnexplainer_top10")

    print("GAT attention by edge type")
    save_table(attention_weights(data), "04_07_attention_by_edge_type")


if __name__ == "__main__":
    main()
