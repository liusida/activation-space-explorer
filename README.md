# Activation Space Explorer

An open research project for exploring activation spaces in language and other models.

The initial approach clusters activations with k-means, assigns new activations to a cluster, and uses PCA to visualize cluster geometry and activation locations.

## Structure

```text
core/                         Shared capture, clustering, and visualization code
experiments/<member>/<name>/  Reproducible member experiments and notable results
```

Start experiments in your own named folder. Keep reusable methods in `core/` and document the model, data, settings, and findings with each experiment.

## Setup

```bash
uv sync
```

Run project commands with `uv run`, for example `uv run pytest`.

See [API.md](API.md) for the proposed core library interface.

The first demo targets `Qwen/Qwen3-0.6B` layer 16; see `experiments/liusida/demo/`.
