"""GCN, GAT and GraphSAGE on the tripartite transaction graph.

Implements Section 3.6. Depth is a tuned hyperparameter because of the
over-smoothing result of Li, Han and Wu (2018): repeated smoothing drives node
representations towards indistinguishability, which attacks the objective
directly when the task is to separate fraudulent nodes from their legitimate
neighbours.

Synthetic oversampling is deliberately NOT applied here (Section 3.7.2).
Generating a synthetic minority node requires inventing its edges, and there is
no principled basis for deciding which entities a fabricated transaction should
connect to. Weighted and focal loss are used instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from common import (DATA, FAST, N_FOLDS, SEED, SKIP_GAT, fold_summary,
                    paired_tests, save_table, score, set_seed)
from graph import training_edge_index

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEARCH_BUDGET = 5 if FAST else 20
MAX_EPOCHS, PATIENCE = (80, 10) if FAST else (200, 20)


class GNN(torch.nn.Module):
    def __init__(self, kind: str, in_dim: int, hidden: int, layers: int,
                 dropout: float, heads: int = 4):
        super().__init__()
        self.kind, self.dropout = kind, dropout
        self.convs = torch.nn.ModuleList()
        dims = [in_dim] + [hidden] * layers
        for i in range(layers):
            a, b = dims[i], dims[i + 1]
            if kind == "GCN":
                self.convs.append(GCNConv(a, b))
            elif kind == "GraphSAGE":
                self.convs.append(SAGEConv(a, b))
            else:                      # GAT: concat heads, keep width constant
                self.convs.append(GATConv(a, b // heads, heads=heads, concat=True))
        self.out = torch.nn.Linear(hidden, 2)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x)


def focal_loss(logits, target, weight, gamma: float = 2.0):
    """Focal loss: down-weights the easy negatives that dominate this problem."""
    ce = F.cross_entropy(logits, target, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def _fit(kind: str, data, params: dict, train_mask, val_mask,
         edge_index, class_weight) -> tuple[GNN, float]:
    set_seed(SEED)
    model = GNN(kind, data.x.size(1), params["hidden"], params["layers"],
                params["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"],
                           weight_decay=params["weight_decay"])

    best_state, best_auprc, bad = None, -1.0, 0
    for _epoch in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        out = model(data.x, edge_index)
        loss = focal_loss(out[train_mask], data.y[train_mask], class_weight)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            prob = F.softmax(model(data.x, data.edge_index), dim=1)[:, 1]
        a = score(data.y[val_mask].cpu(), prob[val_mask].cpu()).auprc
        if a > best_auprc:
            best_auprc, bad = a, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_auprc


def _grid() -> list[dict]:
    rng = np.random.default_rng(SEED)
    return [{"hidden": int(h), "layers": int(l), "dropout": float(d),
             "lr": float(lr), "weight_decay": 5e-4}
            for h, l, d, lr in zip(rng.choice([64, 128, 256], SEARCH_BUDGET),
                                   rng.choice([2, 3], SEARCH_BUDGET),
                                   rng.choice([0.2, 0.4, 0.5], SEARCH_BUDGET),
                                   rng.choice([1e-3, 5e-3, 1e-2], SEARCH_BUDGET))]


@torch.no_grad()
def predict(model, data, edge_index) -> torch.Tensor:
    model.eval()
    return F.softmax(model(data.x, edge_index), dim=1)[:, 1]


def run(data) -> pd.DataFrame:
    data = data.to(DEVICE)
    train_ei = training_edge_index(data).to(DEVICE)   # leakage control

    n_pos = int(data.y[data.train_mask].sum())
    n_neg = int(data.train_mask.sum()) - n_pos
    class_weight = torch.tensor([1.0, n_neg / max(n_pos, 1)],
                                dtype=torch.float, device=DEVICE)

    rows, chosen = [], {}
    tr_idx = data.train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    y_tr = data.y[data.train_mask].cpu().numpy()

    kinds = ("GCN", "GraphSAGE") if SKIP_GAT else ("GCN", "GAT", "GraphSAGE")
    for kind in kinds:
        best, best_a = None, -1.0
        for params in _grid():
            _m, a = _fit(kind, data, params, data.train_mask, data.val_mask,
                         train_ei, class_weight)
            if a > best_a:
                best, best_a = params, a
        chosen[kind] = best
        print(f"  {kind}: best val AUPRC={best_a:.4f}  {best}")

        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for k, (a_idx, b_idx) in enumerate(skf.split(tr_idx, y_tr)):
            m_a = torch.zeros_like(data.train_mask)
            m_b = torch.zeros_like(data.train_mask)
            m_a[tr_idx[a_idx]] = True
            m_b[tr_idx[b_idx]] = True
            model, _ = _fit(kind, data, best, m_a, m_b, train_ei, class_weight)
            prob = predict(model, data, data.edge_index)
            s = score(data.y[m_b].cpu(), prob[m_b].cpu())
            rows.append({"model": kind, "fold": k, "stage": "cv", **s.as_dict()})

        model, _ = _fit(kind, data, best, data.train_mask, data.val_mask,
                        train_ei, class_weight)
        prob = predict(model, data, data.edge_index)
        s = score(data.y[data.test_mask].cpu(), prob[data.test_mask].cpu())
        rows.append({"model": kind, "fold": -1, "stage": "test", **s.as_dict()})
        torch.save(model.state_dict(), DATA / f"model_{kind}.pt")

    pd.DataFrame([{"model": k, **v} for k, v in chosen.items()]).to_csv(
        DATA / "gnn_params.csv", index=False)
    return pd.DataFrame(rows)


def main() -> None:
    data = torch.load(DATA / "graph.pt", weights_only=False)
    print(f"device={DEVICE}  nodes={data.num_nodes:,}")
    res = run(data)
    save_table(res, "04_05_gnn_raw")
    cv = res[res.stage == "cv"].to_dict("records")
    save_table(fold_summary(cv), "04_05_gnn_cv_summary")
    save_table(res[res.stage == "test"], "04_05_gnn_test")


if __name__ == "__main__":
    main()
