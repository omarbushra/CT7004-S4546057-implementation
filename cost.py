"""Inference latency, peak memory and scaling behaviour.

Implements Section 3.7.5 and Objective O4. Latency is measured offline on fixed
hardware and reported as an indicator of near-real-time feasibility, NOT as
evidence of real-time performance - that would require deployment inside a live
streaming architecture.

The figure to compare against is Lu et al. (2022), who report from eBay
production systems that graph query plus GNN inference typically runs to
hundreds of milliseconds, an order of magnitude above an authorisation budget.
Note that the measurement below excludes graph retrieval from a production
store, so it is a lower bound on real deployment cost. Say so in Chapter Four.
"""
from __future__ import annotations

import platform
import time

import numpy as np
import pandas as pd
import torch

from common import (DATA, LATENCY_BUDGET_MS, SEED, SUBSAMPLE_FRACTIONS,
                    save_json, save_table, set_seed)

REPEATS, WARMUP = 30, 5


def hardware() -> dict:
    """Chapter Four must state the machine; absolute values do not transfer."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def _peak_memory_mb(fn) -> tuple[float, float]:
    """Return (elapsed_ms, peak_mb) for one call."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        return dt, torch.cuda.max_memory_allocated() / 1024**2
    import tracemalloc
    tracemalloc.start()
    t0 = time.perf_counter()
    fn()
    dt = (time.perf_counter() - t0) * 1000
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dt, peak / 1024**2


def latency_gnn(model, data, batch_size: int = 1) -> dict:
    """Per-transaction inference latency.

    batch_size=1 is the honest number for an authorisation pipeline: the
    decision is needed for one transaction, not for a batch of 10,000.
    Throughput-style batched figures look far better and mean far less.
    """
    model.eval()
    idx = data.test_mask.nonzero(as_tuple=True)[0][:REPEATS + WARMUP]
    times, peaks = [], []
    with torch.no_grad():
        for i, node in enumerate(idx):
            from torch_geometric.utils import k_hop_subgraph
            sub, sub_ei, mapping, _ = k_hop_subgraph(
                int(node), num_hops=2, edge_index=data.edge_index,
                relabel_nodes=True, num_nodes=data.num_nodes)
            x = data.x[sub]
            dt, mb = _peak_memory_mb(lambda: model(x, sub_ei))
            if i >= WARMUP:
                times.append(dt)
                peaks.append(mb)
    a = np.array(times)
    return {
        "latency_mean_ms": round(float(a.mean()), 3),
        "latency_p95_ms": round(float(np.percentile(a, 95)), 3),
        "latency_p99_ms": round(float(np.percentile(a, 99)), 3),
        "peak_memory_mb": round(float(np.mean(peaks)), 2),
        "within_budget": bool(np.percentile(a, 95) <= LATENCY_BUDGET_MS),
        "budget_ms": LATENCY_BUDGET_MS,
        "note": "excludes production graph retrieval; lower bound",
    }


def latency_tabular(model, X) -> dict:
    times, peaks = [], []
    for i in range(REPEATS + WARMUP):
        row = X[i:i + 1]
        dt, mb = _peak_memory_mb(lambda: model.predict_proba(row))
        if i >= WARMUP:
            times.append(dt)
            peaks.append(mb)
    a = np.array(times)
    return {
        "latency_mean_ms": round(float(a.mean()), 3),
        "latency_p95_ms": round(float(np.percentile(a, 95)), 3),
        "latency_p99_ms": round(float(np.percentile(a, 99)), 3),
        "peak_memory_mb": round(float(np.mean(peaks)), 2),
        "within_budget": bool(np.percentile(a, 95) <= LATENCY_BUDGET_MS),
        "budget_ms": LATENCY_BUDGET_MS,
    }


def scaling(df: pd.DataFrame, feat: list[str]) -> pd.DataFrame:
    """Train and evaluate at 25/50/75/100 per cent of graph size.

    Temporal ordering and class ratio are preserved by taking the earliest
    fraction of transactions rather than a random sample. Fitting growth in
    time and memory against node and edge count gives the empirical scaling
    behaviour; plotting AUPRC against size shows whether any architecture
    needs a minimum scale before its advantage appears, which would explain
    part of the disagreement reviewed in Section 2.6.
    """
    from graph import build, training_edge_index
    from gnns import DEVICE, _fit, _grid, predict
    from common import score
    from prepare import temporal_split

    rows = []
    for frac in SUBSAMPLE_FRACTIONS:
        sub = df.iloc[: int(len(df) * frac)].copy()
        # The split column inherited from the full dataset marks every one of
        # these earliest rows as "train", leaving val and test empty. Re-apply
        # the 70:15:15 temporal split within the subsample so each graph size
        # is evaluated on its own held-out window.
        sub = temporal_split(sub)
        data = build(sub, feat).to(DEVICE)
        train_ei = training_edge_index(data).to(DEVICE)
        n_pos = int(data.y[data.train_mask].sum())
        n_neg = int(data.train_mask.sum()) - n_pos
        cw = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float, device=DEVICE)
        params = _grid()[0]

        for kind in ("GCN", "GAT", "GraphSAGE"):
            t0 = time.perf_counter()
            model, _ = _fit(kind, data, params, data.train_mask, data.val_mask,
                            train_ei, cw)
            train_s = time.perf_counter() - t0
            prob = predict(model, data, data.edge_index)
            s = score(data.y[data.test_mask].cpu(), prob[data.test_mask].cpu())
            rows.append({
                "model": kind, "fraction": frac,
                "nodes": int(data.num_nodes),
                "edges": int(data.edge_index.size(1) // 2),
                "train_seconds": round(train_s, 2),
                "auprc": round(s.auprc, 4),
                **latency_gnn(model, data),
            })
            print(f"  {kind} @{frac:.0%}: {data.num_nodes:,} nodes, "
                  f"train {train_s:.1f}s, AUPRC {s.auprc:.4f}")
    return pd.DataFrame(rows)


def main() -> None:
    set_seed(SEED)
    save_json(hardware(), "04_06_hardware")

    df = pd.read_parquet(DATA / "prepared.parquet")
    feat = pd.read_csv(DATA / "feature_columns.csv", header=None)[0].tolist()

    print("scaling study")
    sc = scaling(df, feat)
    save_table(sc, "04_06_scaling")
    save_table(sc[sc.fraction == 1.0][
        ["model", "latency_mean_ms", "latency_p95_ms", "peak_memory_mb",
         "within_budget"]], "04_06_latency_full_graph")


if __name__ == "__main__":
    main()
