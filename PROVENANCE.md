# Provenance of the evidence in this package

This file states, for every artefact here, where it came from and what it is
evidence of. The distinction that matters is between a **primary record** —
output produced by a program and captured as it was emitted — and a
**transcription** — a figure copied by hand from such output into a table.
Both appear in this package. Treating them as interchangeable would be exactly
the kind of claim this document exists to prevent.

---

## 1. Where the computation happened

The pipeline did not run in one place, and the history matters for reading the
logs.

| Stage | Environment | Why |
|---|---|---|
| `prepare`, `graph` | Local workstation, CPU | Memory-bound rather than compute-bound |
| `baselines` | Local workstation, CPU | 4,673 s; no GPU benefit for scikit-learn or CPU XGBoost |
| `gnns`, `cost`, `operating_points` | Google Colab, NVIDIA Tesla T4 | The local machine has no CUDA device |

Working in a hosted environment had a consequence that shapes this document.
Colab reclaims a virtual machine after a period of inactivity and deletes
`/content` entirely when it does. This happened during the project, after the
analysis stages had completed and their output had been read, but before the
output files themselves were copied off the machine. The numbers survived
because they had been read from the console; several of the files that
produced them did not.

I could have claimed those files were lost to a crash and said nothing further.
Recording the actual sequence is more useful, both because the supervisor asked
for execution evidence and because it explains why some artefacts here are
console transcripts rather than program output.

## 2. `logs/` — primary records

These are the console output of the runs, reproduced verbatim: the command
line, the program's own progress reporting, the warnings it raised and the
tracebacks where it failed. Nothing has been removed. Where a comment has been
added for the reader it is prefixed with `#` and set apart, as at the foot of
`04_gnns.log`.

They are the strongest evidence in this package because they were emitted by
the programs themselves and record what happened rather than what was intended,
including:

- the CUDA out-of-memory failure that ended the first GAT search
  (`04_gnns.log`), and the capped rerun that replaced it (`05_gat_capped.log`),
  with all five candidate scores
- the measured stage durations: baselines 4,673.1 s, evaluate 1.2 s,
  cost 331.8 s
- the resident-set growth through the scaling study, from 4.69 GB to 7.54 GB
  (`07_cost.log`, `08_operating_points.log`), which is why the memory
  limitation in Section 5.6 is stated as a measurement
- the `lbfgs` convergence failure for logistic regression
  (`08_operating_points.log`), which is why Section 4.5 calls its figures a
  floor rather than a result
- the refit reproducibility check, printed in full

The one qualification: these were captured from notebook cell output rather
than written by a logging framework to a file, because the pipeline reports
progress to stdout and no file handler was configured. They are faithful to
what was displayed; they are not machine-signed artefacts, and an independent
reviewer verifies them by re-running the code rather than by trusting the
transcript.

## 3. `results/` — transcriptions

The CSV and JSON files here carry the figures reported in Chapter Four, in the
structure the chapter reports them. They were reconstructed from the console
tables in `logs/` and from the dissertation text after the original output
files were lost, except for the explainability outputs described in Section 5
below.

They are therefore **derived artefacts, not program output**. Their value is
that they make the reported numbers machine-readable and let a reader check
internal consistency — that the declines in
`04_05_TABLE_decline_two_reference_points.csv` follow arithmetically from the
cross-validation, validation and test columns beside them, for instance. Their
value is not that they independently confirm those numbers, because they do
not: they restate them.

Running the pipeline regenerates all of them as genuine program output in
`outputs/`.

## 4. The refits, and what they establish

The baseline models fitted during the `baselines` stage were not serialised,
so the matched-operating-point analysis in Section 4.5 required refitting them
from the recorded hyperparameters. `results/04_05_reproducibility_check.csv`
reports what that refitting recovered:

| Model | Stored AUPRC | Refitted AUPRC | Difference |
|---|---|---|---|
| GCN | 0.395745 | 0.395745 | 0.000000 |
| GAT | 0.414050 | 0.414050 | 0.000000 |
| GraphSAGE | 0.423057 | 0.423057 | 0.000000 |
| XGBoost | 0.563114 | 0.557784 | 0.005330 |
| RandomForest | 0.494528 | 0.495170 | 0.000642 |
| LogisticRegression | 0.169762 | 0.179975 | 0.010213 |

The three graph models are identical because they were restored from saved
weights rather than retrained; that row is a checksum on the loading path, not
a reproducibility result. The three baselines were genuinely refitted under the
same seed and differ in the fourth decimal place, consistent with thread
scheduling in XGBoost's histogram construction and in scikit-learn's parallel
tree building. Logistic regression differs most, which is expected of a model
that terminated on an iteration limit rather than on convergence.

The baselines were later refitted a second time, independently, during the
explainability run (`logs/10_o5_shap.log`). That refit reached 0.5640 for
XGBoost, 0.4943 for RandomForest and 0.1799 for LogisticRegression.
`results/04_07_baseline_refit_check.csv` sets the two refits against the
reported figures.

Two independent refits make the pattern legible rather than anecdotal. The two
tree ensembles land within 0.001 of the reported value in at least one refit
each and within 0.006 in the worst case. Logistic regression is off by 0.010 in
both, in the same direction, and in both runs `lbfgs` terminated on the
2,000-iteration limit rather than on convergence. Its instability is a property
of an unconverged fit, not of the harness.

What this establishes is that the baseline figures are stable to within about
0.01 AUPRC on re-execution, and the tree ensembles to within about 0.005. It
does not establish that the pipeline is bit-reproducible, and the dissertation
does not claim that it is.

## 5. Explainability — status

Objective O5 called for SHAP attribution on the tabular models and
GNNExplainer together with attention analysis on the graph models. These
completed to different degrees, and the difference is set out here rather than
left for a reader to infer.

**Completed.** SHAP attribution over a 2,000-transaction sample of the test
partition ran within the `operating_points` stage; `08_operating_points.log`
records both the sampling step and the writing of
`outputs/04_07_baseline_attribution.csv` with 45 rows, being the fifteen
highest-attribution features for each of the three baseline models. The
execution is evidenced by the log. The output file was on the Colab VM when it
was reclaimed and the table itself was not captured, so the attribution values
are being regenerated by `src/o5.py shap`.

**Failed in the original run, completed on re-execution.** The `explain` stage,
which carries GNNExplainer and the GAT attention analysis, was terminated by
the operating system with signal 9 on host memory exhaustion and produced no
output. The cause is that it materialised the full dense feature matrix
alongside the resident graph. `src/o5.py` reorganises the same work into two
lighter passes and reads features in 64-column blocks into a preallocated
array; `logs/09_o5_explainability.log` records that run, which completed the
GAT attention analysis and GNNExplainer over 30 sampled nodes for each of the
three architectures.

The two result files from that run,
`results/04_07_attention_by_edge_source.csv` and
`results/04_07_retrain_check.csv`, are transcribed from the console output in
that log on the same basis as the rest of `results/`.

One qualification applies to everything in that run. It reconstructs the graph
from `prepared.parquet` and **retrains** the three architectures rather than
restoring the original weights, which were lost with the VM. Retraining does
not reproduce weights bit for bit, because the scatter reductions underlying
message passing are non-deterministic on CUDA. The retrained models reach
within 0.0070 AUPRC of the reported figures (GCN 0.4027 against 0.3957,
GraphSAGE 0.4187 against 0.4231, GAT 0.4145 against 0.4141). The attribution
results therefore describe models equivalent to, but not identical with, those
whose performance Chapter Four reports. Chapter Four reports the original
figures throughout; this run is evidence that they reproduce to within 0.007,
and it is the genuine reproducibility result for the graph models, superseding
the exact match in Section 4 above, which only reflects the weight-loading
path.

## 6. `colab/` — reconstructions

The three scripts in `colab/` were written directly into notebook cells during
the hosted sessions, using `%%writefile`, and were deleted with the VM. They
are not recoverable. What is in `colab/` was rewritten for this package from
the behaviour its logs record: the same inputs, the same candidate
configurations, the same false-positive targets and alert budgets, the same
output files with the same column names and row counts.

They are therefore **functional reconstructions, not the original source**, and
are labelled as such in their own docstrings. They will reproduce the analysis;
they are not evidence of how it was originally written. Anything in this
package that turns on exactly what code ran should rest on `logs/`, which was
captured as it was emitted, rather than on these files.

The three stages in `src/` that ran unmodified — `prepare`, `graph`,
`baselines` — are the original source and are not affected by this.

## 7. What a reader should not conclude from this package

- That the figures in `results/` were independently verified. They were
  transcribed from the runs in `logs/`.
- That the pipeline is deterministic to the last decimal. Section 4 shows it is
  stable to roughly 0.01 AUPRC on the baselines.
- That the reported performance is the best these models can achieve. The
  reduced search budget in the README, Section 6, makes every figure a lower
  bound.
- That the latency measurements describe a deployed system. They describe a
  forward pass on one T4 with the graph already resident and no retrieval step.
  Section 4.6 of the dissertation states this, and it is the reason the
  scalability claims there are bounded rather than general.

---

S4546057 · 21 October 2026
