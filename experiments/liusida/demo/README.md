# Qwen3-0.6B layer 16 OpenWebText demo

This demo streams exactly one million OpenWebText tokens, captures every token's activation after zero-based transformer layer 16, and fits 1,000 mini-batch k-means clusters. It separately captures the configured token ` Friday` in `Today is Friday.`, assigns it to its nearest cluster, fits PCA only on that cluster's OpenWebText members, and projects the target into the resulting interactive Plotly 3D view. It does not fit a global PCA. Its settings and exact required core version live in `config.toml`.

The demo is an independent uv project. From the repository root, run it directly:

```bash
uv run --project experiments/liusida/demo python experiments/liusida/demo/run.py
```

No separate sync step is required. On the first run, uv creates this experiment's `.venv`, installs the dependencies from its lockfile, and installs the repository's core library into that environment in editable mode. The root `.venv` is not needed.

The view highlights decoded token text containing `fri` after trimming and lowercasing. Override it without rerunning the expensive stages:

```bash
uv run --project experiments/liusida/demo python experiments/liusida/demo/run.py --keyword day
```

Generated files are written to `output/` and are ignored by Git. The float32 activation file alone is approximately 4 GiB for Qwen3-0.6B, so ensure sufficient disk space before running.

The shared runner displays overall and per-stage progress, ETA, substages, and recent output. Repeating the command reuses validated completed stages; full logs and resume state live under `output/logs/` and `output/run.json`.

The target prompt and exact target token live under `[target]` in `config.toml`; the token must occur once in the prompt and map to exactly one tokenizer token. The interactive output is `output/target-friday-cluster-<id>-pca-3d.html`. It shows the target token and its surrounding text, and clicking any cluster member reveals that token's context. `--keyword` changes only rendering, so capture, clustering, and PCA artifacts remain reusable. Other static plots in the project continue to use Matplotlib.
