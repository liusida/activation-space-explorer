# Qwen3-0.6B layer 16 demo

This demo captures the final prompt-token activation after zero-based transformer layer 16, clusters the activations, and creates global and cluster-local PCA views. Its model, analysis settings, and exact required core version live in `config.toml`.

On GB10:

```bash
uv sync --extra demo
uv run --extra demo python experiments/liusida/demo/run.py
```

Edit `prompts.txt` to change the small demonstration dataset. Generated files are written to `output/` and are ignored by Git.
