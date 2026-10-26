# Implementation and reproducibility package

**CT7004 Dissertation — Graph Neural Networks for Credit Card Fraud Detection**
Student S4546057 · University of Gloucestershire

This package contains the complete implementation behind Chapter Four of the
dissertation: every script that was run, the console output of each run, and
the result tables those runs produced. It exists so that the reported numbers
can be checked rather than taken on trust.

**Read `PROVENANCE.md` alongside this file.** It states, for each artefact
here, whether it is output captured as a program emitted it or a figure
transcribed from such output, and what each can and cannot be used to
establish. Several files in `results/` and all three scripts in `colab/` fall
into the second category, for reasons that document sets out.

---

## 1. What is here

```
README.md                  this file
PROVENANCE.md              which artefacts are primary records and which are transcribed
requirements.txt           package versions
run_all.py                 stage runner
src/                       the pipeline, in execution order
  common.py                configuration, seeding, metrics, IO
  prepare.py               load, entity keys, preprocessing, temporal split
  graph.py                 tripartite graph construction, leakage masking
  baselines.py             logistic regression, random forest, XGBoost
  gnns.py                  GCN, GAT, GraphSAGE
  evaluate.py              combines both families into the Chapter Four tables
  cost.py                  latency, peak memory, scaling study
  explain.py               SHAP, GNNExplainer, GAT attention
  o5.py                    consolidated explainability run (Objective O5)
  ablate.py                TransactionDT ablation (Section 4.7)
colab/                     scripts written during the run, for the reasons given below
  operating_points.py      matched false-positive comparison and SHAP
  gat_only.py              GAT at a capped hidden dimension after the OOM
  recover.py               rebuild graph and models from prepared.parquet
logs/                      verbatim console output of every stage that was run
results/                   the tables and statistics reported in Chapter Four
```

## 2. Reproducing the results

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch                                    # match your CUDA version
pip install -r requirements.txt && pip install torch_geometric
```

Place `train_transaction.csv` and `train_identity.csv` from the IEEE-CIS Fraud
Detection dataset in `data/`. The dataset is not redistributed here; it is
available from Kaggle under the competition's terms.

```bash
FAST=1 python run_all.py            # prepare, graph, baselines, gnns, evaluate, cost, explain
```

`FAST=1` is the configuration actually used and reported: five hyperparameter
candidates per model instead of twenty, three cross-validation folds instead of
five, and a cap of eighty epochs with patience ten. Section 4.4 of the
dissertation records this reduction and Section 5.6 lists it as a limitation.
Omitting the flag runs the full design in Section 3.7.3, which takes
substantially longer and was not affordable within the project period.

Individual stages:

```bash
python run_all.py prepare           # ~10 min CPU
python run_all.py graph             # ~5 min
python run_all.py baselines         # 4,673 s measured
FAST=1 python run_all.py gnns       # ~25 min on a Tesla T4
python run_all.py evaluate          # seconds
FAST=1 python run_all.py cost       # 332 s measured
python src/o5.py gnn && python src/o5.py shap   # explainability
```

Every stage writes to `outputs/`. The seed is fixed at 42 in `common.py` and
applied to Python, NumPy and PyTorch before each fit.

## 3. How the methodology maps to the code

| Dissertation | Implementation | Produces |
|---|---|---|
| §3.5 entity keys | `prepare.py::add_entity_keys` | `card_id`, `card_id_ablation`, `device_id` |
| §3.5 graph construction | `graph.py::build` | `graph.pt` |
| §3.5 leakage control | `graph.py::training_edge_index` | masked message-passing edges |
| §3.7.1 preprocessing, temporal split | `prepare.py::preprocess`, `::temporal_split` | `prepared.parquet` |
| §3.7.2 imbalance treatment | `gnns.py::focal_loss`, `baselines.py::_make` | class weights, focal loss |
| §3.7.3 tuning | `gnns.py::_grid`, `baselines.py::_grids` | `gnn_params.csv`, `baseline_params.csv` |
| §3.7.4 metrics | `common.py::score`, `::fold_summary`, `::paired_tests` | Tables 5–9 |
| §2.7 matched operating points | `colab/operating_points.py` | Tables 7–8 |
| §3.7.5 cost and scaling | `cost.py` | Tables 11–12 |
| §3.7.6 explainability | `o5.py`, `explain.py` | §4.7 attribution tables |

## 4. Which script produced which table or figure

| Chapter Four | Script | Output file |
|---|---|---|
| §4.2 dataset statistics | `prepare.py` | `results/04_02_dataset_statistics.json` |
| §4.3 graph statistics, homophily | `graph.py` | `results/04_03_graph_statistics.json` |
| Table 5 cross-validation | `baselines.py`, `gnns.py`, `evaluate.py` | `results/04_05_TABLE_detection_performance.csv` |
| Table 6 held-out test | `evaluate.py` | `results/04_05_TABLE_heldout_test.csv` |
| Table 7 matched FPR | `colab/operating_points.py` | `results/04_05_TABLE_matched_fpr.csv` |
| Table 8 alert budget | `colab/operating_points.py` | `results/04_05_TABLE_alert_budget.csv` |
| Table 9, Table 10, Figure 3 | `evaluate.py` + derivation in §4.5 | `results/04_05_TABLE_decline_two_reference_points.csv` |
| Table 11 latency and memory | `cost.py::latency_gnn` | `results/04_06_latency_full_graph.csv` |
| Table 12 scaling | `cost.py::scaling` | `results/04_06_scaling.csv` |
| refit verification | `colab/operating_points.py` | `results/04_05_reproducibility_check.csv` |

## 5. Decisions in the code that the dissertation depends on

These are the places where an implementation choice determines a reported
number, and each is commented in the source at the point it is made.

**Composite card key** (`prepare.py`). `card1` resolves only 13,553 distinct
values across 590,540 transactions and cannot represent individual accounts, so
the key is `card1 + addr1 + a normalised account-age field derived from D1 and
TransactionDT`, giving 218,030 entities. Missing components take a sentinel
rather than propagating null: without it, roughly 65,000 transactions with no
`addr1` would collapse into one hub node and manufacture the spurious structure
§3.5 rejects for merchants. `card_id_ablation` retains the `card1`-only version
so the sensitivity check in §3.5 can be run.

**Missingness indicators** (`prepare.py::preprocess`). A binary indicator is
generated for every retained feature between 1% and 90% missing, before
imputation, taking the feature count from 420 to 731. This is not cosmetic:
transactions with no `addr1` carry a fraud rate of 11.78% against 2.46%, and
the graph encodes that structurally through the key sentinel. Without the
indicators the baselines would be denied information the graph models receive
and the like-for-like claim in §3.6 would not hold.

**Leakage control** (`graph.py::training_edge_index`). An edge is available for
message passing during training only when its transaction endpoint is in the
training partition; entity nodes are always permitted because they carry no
label. 28.79% of directed edges are masked. Evaluation uses the full graph,
which is correct — a deployed model sees everything accumulated up to the
moment of inference.

**No synthetic oversampling for the graph models** (`gnns.py`). Generating a
synthetic minority node requires inventing its edges, and there is no
principled basis for choosing which entities a fabricated transaction connects
to. Weighted focal loss is used instead.

**Batch size one for latency** (`cost.py::latency_gnn`). An authorisation
decision concerns a single transaction. Batched throughput figures look far
better and mean considerably less. The reported `peak_memory_mb` includes the
resident graph and is not a per-inference figure; §4.6 says so.

## 6. The TransactionDT ablation

`src/ablate.py` tests the confound Section 4.7 raises: TransactionDT ranks
first on attribution for four of the five models, and under a temporal split
its test values lie wholly outside the training range, so part of the
degradation Section 4.5 attributes to concept drift might be extrapolation
failure on a time index instead.

    python src/ablate.py base     # three baselines, both conditions
    python src/ablate.py gnn      # three graph models, both conditions

The `base` stage was run twice (`logs/11_ablation.log`). Both conditions are
fitted inside one run on one code path, so the difference between them is the
feature and not re-execution noise; comparing an ablated refit against the
figures already in Chapter Four would have confounded the two.

**The confound does not hold.** Removing TransactionDT changes the
validation-to-test decline by at most 0.7 percentage points, and for XGBoost
the decline grows slightly rather than shrinking. The two tree ensembles
reproduced exactly across both replications; logistic regression moved, as it
does across every refit in this study, because `lbfgs` terminates on its
iteration limit rather than on convergence.

A second result falls out of it. TransactionDT ranks first by SHAP, yet
removing it costs XGBoost 0.0075 AUPRC on test, because D1-D15 and the card
aggregates carry correlated time information that absorbs the loss. High
attribution is not high necessity.

The `gnn` stage was not run: retraining alone moves the graph models by up to
0.0070 AUPRC, the same order as the effect being measured, so the test would be
inconclusive. Section 6.4 records it as outstanding.

## 7. Known deviations from the design in Chapter Three

Each is recorded in the dissertation as well as here.

1. **Reduced search budget.** Five candidates and three folds rather than
   twenty and five. Applied identically to all six models, so the equal-budget
   property is preserved, but absolute performance for every model is a lower
   bound on what a full search would find, and three folds leaves the planned
   Wilcoxon test unable to return a p-value below 0.25 at any effect size.
2. **GAT hidden dimension capped at 128.** At 256 the per-edge attention tensor
   exhausted the 15 GB device (see `logs/04_gnns.log` for the failure and
   `logs/05_gat_capped.log` for the rerun). The excluded configurations were in
   the direction already performing worse, which §4.4 sets out, but the
   argument is inferential.
3. **Logistic regression did not converge.** `lbfgs` reached the 2,000-iteration
   limit; see `logs/08_operating_points.log`. Its figures are a floor.
4. **Baselines refitted for the matched-operating-point analysis.** The original
   fits were not retained. The refits reproduce the recorded AUPRC to within
   0.0006–0.0102; `results/04_05_reproducibility_check.csv` reports the
   difference per model. The graph models reproduce exactly, having been
   restored from saved weights.
5. **Explainability.** See `PROVENANCE.md`.

## 8. Environment

Graph models: NVIDIA Tesla T4, 15 GB device memory, 12.7 GB host memory,
Python 3.13, PyTorch 2.11 with CUDA 13.0, PyTorch Geometric 2.8.1.
Baselines: CPU. Absolute timings do not transfer between machines; the
comparisons between models and across graph sizes do.
