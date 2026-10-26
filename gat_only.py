"""GAT search with the hidden dimension capped, after the OOM in `gnns`.

Reconstructed for this package; see PROVENANCE.md, Section 6. The original was
written into a Colab cell and lost with the VM. The behaviour here matches what
logs/05_gat_capped.log records: the same five-candidate budget and three-fold
protocol as every other model, with hidden restricted to {64, 128} because 256
exhausts a 15 GB T4 on this graph.

    FAST=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python colab/gat_only.py

GCN and GraphSAGE results already in outputs/ are preserved; GAT is merged in.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch

from common import FAST, SEED, N_FOLDS, set_seed, save_table, fold_summary
import gnns

# The full grid in gnns._grid() offers hidden in {64, 128, 256}. At 256 GATConv
# retains a per-edge message tensor across the backward pass, and with 1,462,800
# directed edges that allocation exceeds the device. The five candidates below
# are the same budget every other model received, drawn from the surviving
# region of the same grid.
CANDIDATES = [
    dict(hidden=64,  layers=3, dropout=0.4, lr=0.01,  weight_decay=5e-4),
    dict(hidden=128, layers=2, dropout=0.5, lr=0.005, weight_decay=5e-4),
    dict(hidden=128, layers=3, dropout=0.5, lr=0.001, weight_decay=5e-4),
    dict(hidden=64,  layers=2, dropout=0.5, lr=0.01,  weight_decay=5e-4),
    dict(hidden=64,  layers=2, dropout=0.5, lr=0.005, weight_decay=5e-4),
]


def main() -> None:
    set_seed(SEED)
    data = torch.load("outputs/graph.pt", weights_only=False)
    print(f"device={gnns.DEVICE}  nodes={data.num_nodes:,}")

    best, best_val = None, -1.0
    for params in CANDIDATES:
        _model, val = gnns._fit("GAT", data, params, data.train_mask, data.val_mask)
        print(f"  {params} -> val AUPRC {val:.4f}")
        if val > best_val:
            best, best_val = params, val

    print(f"\n  GAT: best val AUPRC={best_val:.4f}  {best}")

    # Same three-fold protocol as the other five models, then one test pass.
    rows = []
    for fold in range(N_FOLDS):
        set_seed(SEED + fold)
        _m, auprc = gnns._fit("GAT", data, best, data.train_mask, data.val_mask)
        print(f"    fold {fold}: AUPRC {auprc:.4f}")
        rows.append(dict(model="GAT", fold=fold, auprc=auprc, **best))

    model, _ = gnns._fit("GAT", data, best, data.train_mask, data.val_mask)
    prob = gnns.predict(model, data, data.edge_index)
    from common import score
    test = score(data.y[data.test_mask].cpu().numpy(),
                 prob[data.test_mask].cpu().numpy())
    print(f"    test: AUPRC {test.auprc:.4f}")
    torch.save(model.state_dict(), "outputs/model_GAT.pt")

    # Merge with the GCN and GraphSAGE rows the checkpointed run already wrote.
    raw_path = Path("outputs/04_05_gnn_raw.csv")
    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    existing = existing[existing.get("model", pd.Series(dtype=str)) != "GAT"]
    raw = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    save_table(raw, "04_05_gnn_raw")
    save_table(fold_summary(raw.to_dict("records")), "04_05_gnn_cv_summary")

    test_path = Path("outputs/04_05_gnn_test.csv")
    tdf = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()
    tdf = tdf[tdf.get("model", pd.Series(dtype=str)) != "GAT"]
    tdf = pd.concat([tdf, pd.DataFrame([dict(model="GAT", **test.__dict__)])],
                    ignore_index=True)
    save_table(tdf, "04_05_gnn_test")
    print("GAT merged with GCN and GraphSAGE in outputs/")


if __name__ == "__main__":
    main()
