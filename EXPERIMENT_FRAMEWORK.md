# Experiment Runner Design

Status: initial framework implemented and adopted by `liusida/demo`.

## Goal

Every long experiment should present the same compact terminal interface, keep a complete log, and safely resume when the same command is repeated.

The runner owns terminal rendering. Experiment code reports structured events instead of creating its own `tqdm` bars.

## Terminal interface

```text
╭─ Activation Space Explorer · liusida/demo ───────────────────────────────╮
│ Qwen/Qwen3-0.6B · layer 16 · OpenWebText · run 8fa21c3b                  │
╰──────────────────────────────────────────────────────────────────────────╯

Overall     ███████████████░░░░░  3/5 stages     00:18:42  ETA 00:07:31
Stage       Fit k-means             742/1,230 batches      ETA 00:03:16
Substage    epoch 7/10 · batch 4/123
Status      running

Recent output
  Reused capture: 1,000,000 × 1,024 float32
  Initialized 1,000 centroids
  Epoch 6 completed in 31.8s
  Current inertia: 2.418e+08

Full log    experiments/liusida/demo/output/logs/20261012-201530.log
State       experiments/liusida/demo/output/run.json
```

The display refreshes in place and contains only:

1. experiment identity and resolved configuration;
2. overall stage progress, elapsed time, and ETA;
3. current stage and substage progress;
4. a bounded recent-output window;
5. full-log and state paths;
6. a visible warning when the source tree is dirty.

On completion or failure, the live display stops and prints a short permanent summary.

## Experiment contract

Each experiment keeps a small `run.py` which declares a plan and supplies stage functions:

```python
from core.runs import ExperimentPlan, Stage, run_experiment


plan = ExperimentPlan(
    name="liusida/demo",
    output_dir=HERE / "output",
    config=config,
    stages=[
        Stage("capture", "Capture activations", capture, durable=True),
        Stage("cluster", "Fit k-means", fit_clusters, durable=True),
        Stage("local-pca", "Fit target-cluster PCA", fit_local_pca, durable=True),
        Stage("render", "Render", render, durable=False),
    ],
)

run_experiment(plan)
```

A stage receives one `RunContext`:

```python
def fit_clusters(ctx: RunContext) -> None:
    with ctx.progress(total=1_230, unit="batch") as progress:
        for epoch in range(10):
            progress.substage(f"epoch {epoch + 1}/10")
            for batch in batches:
                model.partial_fit(batch)
                progress.advance()
    ctx.save_npz_atomic("clusters/centroids.npz", centroids=centroids)
    ctx.complete({"clusters": 1_000})
```

The context exposes only a small common API:

```python
ctx.log(message, level="info")
ctx.progress(total, unit)
ctx.save_json_atomic(path, value)
ctx.save_npz_atomic(path, **arrays)
ctx.complete(summary)
```

Experiments may use ordinary functions internally. They should not write terminal progress directly.

## Stage model

`Stage` contains:

```python
Stage(
    id="cluster",
    title="Fit k-means",
    function=fit_clusters,
    durable=True,
    dependencies=("capture",),
    validator=validate_cluster_artifacts,
)
```

- `id` is stable and machine-readable.
- `title` is user-facing.
- `durable=True` means a validated completion is reused on the next run.
- `dependencies` document upstream ordering; the initial runner executes the declared list linearly.
- `validator` checks and reloads a completed stage's durable artifacts on resume.

The initial implementation needs only a linear stage order. The dependency fields preserve a path to a small DAG later without requiring one now.

## Progress and ETA

The runner tracks two levels:

- **Overall:** completed stages out of total stages.
- **Current stage:** completed work units out of the stage total.

The current-stage ETA uses a smoothed observed rate from this invocation. Reused work does not affect the rate.

An overall ETA is shown only when it is defensible:

1. current-stage ETA is measured from current progress;
2. remaining-stage estimates come from durations recorded by an earlier compatible run;
3. if no history exists, overall ETA is shown as `—` rather than inventing one.

Stage code may update a substage label freely; this does not change progress accounting.

## Output and logging

`ctx.log()` sends each message to both:

- an in-memory ring buffer shown as the most recent 6–10 lines;
- an append-only UTF-8 log file containing every lifecycle and experiment message.

Python logging is connected to the same sink. A lightweight line-buffered stream adapter can capture third-party stdout/stderr when needed. Carriage-return progress output is normalized before entering the log.

Logs live under:

```text
<output>/logs/<timestamp>-<run-id>.log
```

The UI never truncates the full log.

## State and resume

`<output>/run.json` is the source of durable run state:

```json
{
  "schema_version": 1,
  "experiment": "liusida/demo",
  "config_hash": "sha256:...",
  "core_version": "0.1.0",
  "git_commit": "...",
  "git_dirty": false,
  "stages": {
    "capture": {
      "status": "complete",
      "duration_seconds": 493.2,
      "summary": {"tokens": 1000000, "dimension": 1024}
    }
  }
}
```

Repeating a command:

1. loads and validates `run.json`;
2. compares the resolved configuration and displays readable differences;
3. calls each completed stage's validator to verify and reload its artifacts;
4. marks compatible stages as reused;
5. resumes at the first incomplete stage.

A different result-affecting configuration must use another output directory. The runner never mixes configurations or silently deletes results.

## Durable writes

All state and small artifacts use write–flush–rename:

```text
run.json.tmp → validate → run.json
```

Large captures write to a `.partial` path. A stage promotes the partial file only after its size and metadata validate. Interrupted partial files are retained only when the stage explicitly knows how to resume them; otherwise the next run reports them clearly.

`Ctrl-C` changes the active stage to `interrupted`, flushes the log and state, restores the terminal, and exits with code 130.

## Implemented package

```text
core/runs/
├── __init__.py   # public runner API
├── model.py      # ExperimentPlan, Stage, RunContext, progress events
├── runner.py     # lifecycle, validation, resume, interruption
├── display.py    # Rich live terminal view
├── state.py      # versioned run.json and atomic persistence
└── logging.py    # stdout/stderr line capture
```

`rich` supplies the live layout, progress bars, colors, and bounded panels. It should be the only terminal-rendering dependency. Core numerical operations receive progress callbacks from the runner instead of importing terminal UI code.

## Demo migration

The current demo maps naturally to five stages:

| Stage | Durable output | Work unit |
| --- | --- | --- |
| Capture | activation and token memmaps plus metadata | token |
| Cluster | centroids and assignments | mini-batch |
| Target-cluster PCA | one fitted 3D projector | cluster |
| Render | interactive Plotly HTML and summary | plotted point |

The initial implementation migrates only `liusida/demo`. Once its interruption, resume, logging, and terminal behavior are verified in real long runs, new experiments can adopt the same contract and older experiments can migrate when touched.

## Intentionally deferred

- distributed workers or remote monitoring;
- arbitrary concurrent stage DAGs;
- a browser dashboard;
- per-batch durable checkpoints for scikit-learn models;
- automatic artifact upload;
- cross-machine resume.
- content fingerprints that automatically invalidate downstream stages;
- history-calibrated overall ETA across stages with very different costs.

These can be added without changing the experiment-facing stage/context API.
