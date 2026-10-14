# Qwen3.5 superposition experiment

Test whether the `square` activation in `The artist painted a red square.` is better described by one k-means cluster or simultaneous color and shape ICA features at layer 5.

```bash
# Fit ICA for all layers.
uv run --project experiments/liusida/superposition python experiments/liusida/superposition/fit_ica.py

# Build all component profiles.
uv run --project experiments/liusida/superposition python experiments/liusida/superposition/build_profiles.py

# Inspect layer 5 interactively.
uv run --project experiments/liusida/superposition jupyter lab experiments/liusida/superposition/analyze_layer5.ipynb

# Fit layer-5 k-means, assign the target, and render cluster-local PCA-3D.
uv run --project experiments/liusida/superposition python experiments/liusida/superposition/fit_kmeans.py
```

Configuration lives in `config.toml`. Large fitted models, activations, clusters, logs, and HTML views live in ignored `output/`.
