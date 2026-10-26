"""Build the tripartite transaction-card-device graph.

Implements Section 3.5. Produces the descriptive statistics required by
Objective O1 and a PyTorch Geometric Data object for the GNN experiments.

Structure: transaction nodes carry the feature matrix; card and device nodes
are connective entities with no attributes of their own. Two transactions
sharing a card or a device are therefore joined by a path of length two,
which is what lets message passing move information between them.

The heterogeneous graph is converted to a homogeneous one with a three-way
node-type indicator appended to the features, so that the standard GCNConv /
GATConv / SAGEConv layers apply directly without a HeteroConv wrapper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from common import DATA, save_json, save_table

TX, CARD, DEV = 0, 1, 2


def build(df: pd.DataFrame, feat: list[str], card_col: str = "card_id") -> Data:
    n_tx = len(df)

    # cat.codes picks the narrowest integer that fits the category count
    # (int8/int16), which then overflows when a node offset in the hundreds of
    # thousands is added. Force int64. np.arange is int32 on Windows for the
    # same reason, so pin that too.
    cards = df[card_col].astype("category").cat.codes.to_numpy().astype(np.int64)
    n_card = int(cards.max()) + 1

    dev_series = df["device_id"].astype("category")
    devs = dev_series.cat.codes.to_numpy().astype(np.int64)   # -1 where missing
    n_dev = int(devs.max()) + 1 if (devs >= 0).any() else 0

    card_off, dev_off = n_tx, n_tx + n_card
    n_nodes = n_tx + n_card + n_dev

    # ── features: transactions carry theirs, entity nodes get zeros ──
    d = len(feat)
    x = torch.zeros((n_nodes, d + 3), dtype=torch.float)
    x[:n_tx, :d] = torch.tensor(df[feat].to_numpy(dtype=np.float32))
    x[:n_tx, d + TX] = 1.0
    x[card_off:card_off + n_card, d + CARD] = 1.0
    if n_dev:
        x[dev_off:dev_off + n_dev, d + DEV] = 1.0

    # ── edges: transaction <-> card, transaction <-> device ──
    tx_idx = np.arange(n_tx, dtype=np.int64)
    e_card = np.stack([tx_idx, cards + card_off])

    mask = devs >= 0
    e_dev = (np.stack([tx_idx[mask], devs[mask] + dev_off]) if n_dev
             else np.zeros((2, 0), dtype=np.int64))

    und = np.concatenate([e_card, e_dev], axis=1).astype(np.int64)
    edge_index = torch.tensor(np.concatenate([und, und[::-1]], axis=1), dtype=torch.long)

    y = torch.zeros(n_nodes, dtype=torch.long)
    y[:n_tx] = torch.tensor(df["isFraud"].to_numpy(), dtype=torch.long)

    split = df["split"].astype(str).to_numpy()
    masks = {}
    for name in ("train", "val", "test"):
        m = torch.zeros(n_nodes, dtype=torch.bool)
        m[:n_tx] = torch.from_numpy(np.asarray(split == name, dtype=bool))
        masks[f"{name}_mask"] = m

    data = Data(x=x, edge_index=edge_index, y=y, **masks)
    data.n_tx, data.n_card, data.n_dev = n_tx, n_card, n_dev
    return data


def training_edge_index(data: Data) -> torch.Tensor:
    """Edges safe to use for message passing during training.

    Section 3.5: because edges connect transactions that may fall on either
    side of the split, unrestricted message passing lets test-set information
    reach the training nodes. An edge is kept only when its transaction
    endpoint is in the training set; card and device nodes are always allowed
    because they carry no label.
    """
    src, dst = data.edge_index
    is_tx = lambda n: n < data.n_tx                      # noqa: E731
    ok_src = torch.where(is_tx(src), data.train_mask[src], torch.ones_like(data.train_mask[src]))
    ok_dst = torch.where(is_tx(dst), data.train_mask[dst], torch.ones_like(data.train_mask[dst]))
    return data.edge_index[:, ok_src & ok_dst]


def descriptive_statistics(data: Data, df: pd.DataFrame) -> dict:
    """Objective O1 output."""
    from torch_geometric.utils import degree

    deg = degree(data.edge_index[0], data.num_nodes).numpy()
    tx_deg, card_deg = deg[:data.n_tx], deg[data.n_tx:data.n_tx + data.n_card]

    card_fraud = df.groupby("card_id")["isFraud"].mean()
    return {
        "nodes_total": int(data.num_nodes),
        "nodes_transaction": int(data.n_tx),
        "nodes_card": int(data.n_card),
        "nodes_device": int(data.n_dev),
        "edges_undirected": int(data.edge_index.size(1) // 2),
        # every transaction has exactly one card edge; device edges exist only
        # where identity data is present, which is the ~24% figure Section 3.5
        # flags as the binding limitation on the device relation
        "edges_card": int(data.n_tx),
        "edges_device": int(df["has_device"].sum()),
        "transaction_degree_mean": round(float(tx_deg.mean()), 3),
        "transaction_degree_max": int(tx_deg.max()),
        "card_degree_mean": round(float(card_deg.mean()), 3),
        "card_degree_max": int(card_deg.max()),
        # the honest number: device edges exist for only ~a quarter of nodes
        "device_edge_coverage": round(float(df["has_device"].mean()), 4),
        "cards_with_any_fraud": int((card_fraud > 0).sum()),
        "cards_entirely_fraud": int((card_fraud == 1).sum()),
        "fraud_rate_overall": round(float(df["isFraud"].mean()), 5),
    }


def homophily(data: Data):
    """Edge homophily among transaction nodes only.

    Section 2.4 argues fraud graphs are heterophilic. This measures it on the
    graph actually built, and the number belongs in Chapter Four: if homophily
    is low, the prediction that vanilla GCN underperforms has a mechanism
    behind it rather than an assumption.
    """
    src, dst = data.edge_index
    both_tx = (src < data.n_tx) & (dst < data.n_tx)
    if both_tx.sum() == 0:          # tripartite: no direct tx-tx edges
        return None                 # None -> null; NaN is not valid JSON
    return float((data.y[src[both_tx]] == data.y[dst[both_tx]]).float().mean())


def two_hop_homophily(df: pd.DataFrame, card_col: str = "card_id") -> float:
    """Label agreement between transactions sharing a card.

    The tripartite graph has no direct transaction-transaction edge, so the
    quantity that matters is agreement across the two-hop path. Computed by
    group rather than by edge enumeration, which would be quadratic.
    """
    g = df.groupby(card_col)["isFraud"]
    n, agree = g.size().to_numpy(), g.sum().to_numpy()
    pairs = n * (n - 1) / 2
    same = agree * (agree - 1) / 2 + (n - agree) * (n - agree - 1) / 2
    keep = pairs > 0
    return float(same[keep].sum() / pairs[keep].sum())


def main() -> None:
    df = pd.read_parquet(DATA / "prepared.parquet")
    feat = pd.read_csv(DATA / "feature_columns.csv", header=None)[0].tolist()

    print("building tripartite graph")
    data = build(df, feat)

    stats = descriptive_statistics(data, df)
    stats["edge_homophily_direct"] = homophily(data)
    stats["two_hop_homophily_shared_card"] = round(two_hop_homophily(df), 4)
    save_json(stats, "04_03_graph_statistics")

    train_ei = training_edge_index(data)
    save_json({
        "edges_all": int(data.edge_index.size(1)),
        "edges_available_in_training": int(train_ei.size(1)),
        "proportion_masked": round(1 - train_ei.size(1) / data.edge_index.size(1), 4),
    }, "04_03_leakage_control")

    torch.save(data, DATA / "graph.pt")
    print(f"\ngraph.pt written: {data.num_nodes:,} nodes, "
          f"{data.edge_index.size(1)//2:,} undirected edges")


if __name__ == "__main__":
    main()
