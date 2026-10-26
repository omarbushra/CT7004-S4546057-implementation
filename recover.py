"""Rebuild graph.pt and the graph models from prepared.parquet.

Reconstructed for this package; see PROVENANCE.md, Section 6.

Colab reclaims an idle VM and deletes /content with it. prepared.parquet is
105 MB and can be re-uploaded in a few minutes; graph.pt is 2.4 GB and cannot.
This script reconstructs the graph and retrains the three architectures at
their selected configurations, so that a reclaimed session can be restored
from the small file rather than the large one.

    PYTHONUNBUFFERED=1 python colab/recover.py

Retraining does not reproduce the original weights bit for bit. The GCN check
in Section 4.4 of the dissertation records a rebuilt test AUPRC of 0.4060
against the 0.3957 originally obtained, a difference of 0.0103 arising from
non-deterministic CUDA reductions in the scatter operations underlying message
passing. Where a figure in Chapter Four could have come from either run, the
original is the one reported.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch

from common import SEED, set_seed, score
import graph as G
import gnns

BEST = {        # selected configurations, from logs/04_gnns.log and 05_gat_capped.log
    "GCN":       dict(hidden=256, layers=2, dropout=0.5, lr=0.005, weight_decay=5e-4),
    "GraphSAGE": dict(hidden=128, layers=3, dropout=0.5, lr=0.001, weight_decay=5e-4),
    "GAT":       dict(hidden=64,  layers=2, dropout=0.5, lr=0.005, weight_decay=5e-4),
}


def main() -> None:
    set_seed(SEED)
    out = Path("outputs")
    out.mkdir(exist_ok=True)

    if (out / "graph.pt").exists():
        data = torch.load(out / "graph.pt", weights_only=False)
        print(f"loaded existing graph: {data.num_nodes:,} nodes")
    else:
        df = pd.read_parquet("prepared.parquet")
        feat = pd.read_csv("feature_columns.csv", header=None)[0].tolist()
        print(f"building graph from {len(df):,} transactions, {len(feat)} features")
        data = G.build(df, feat)
        torch.save(data, out / "graph.pt")
        print(f"  {data.num_nodes:,} nodes, {data.edge_index.shape[1]:,} directed edges")
        del df

    # GAT last: it is the architecture that exhausted the device, so a failure
    # here leaves the other two intact rather than losing the whole run.
    for name in ("GCN", "GraphSAGE", "GAT"):
        set_seed(SEED)
        model, val = gnns._fit(name, data, BEST[name], data.train_mask, data.val_mask)
        prob = gnns.predict(model, data, data.edge_index)
        test = score(data.y[data.test_mask].cpu().numpy(),
                     prob[data.test_mask].cpu().numpy())
        torch.save(model.state_dict(), out / f"model_{name}.pt")
        print(f"  {name}: val AUPRC {val:.4f}, test AUPRC {test.auprc:.4f}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
