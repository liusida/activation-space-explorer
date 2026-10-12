# Activation Space Explorer

An open research project for exploring activation spaces in language and other models.

The initial approach clusters activations with k-means, assigns new activations to a cluster, and uses PCA to visualize cluster geometry and activation locations.

## Structure

```text
core/                         Shared capture, clustering, and visualization code
experiments/<member>/<name>/  Reproducible member experiments and notable results
```

Start experiments in your own named folder. Keep reusable methods in `core/` and document the model, data, settings, and findings with each experiment.

## Running experiments

Each experiment is its own uv project and manages its own virtual environment. You do not need a root virtual environment to run one. For example:

```bash
uv run --project experiments/liusida/demo python experiments/liusida/demo/run.py
```

On the first run, uv creates `experiments/liusida/demo/.venv`, installs its locked dependencies, and installs the root core library into that environment in editable mode.

## Developing the core

The root environment is only needed when changing or testing `core/`:

```bash
uv sync
uv run pytest
uv run ruff check .
```

See [API.md](API.md) for the proposed core library interface.

See [EXPERIMENT_FRAMEWORK.md](EXPERIMENT_FRAMEWORK.md) for the shared long-run experiment runner design.

The first demo targets `Qwen/Qwen3-0.6B` layer 16; see `experiments/liusida/demo/`.
