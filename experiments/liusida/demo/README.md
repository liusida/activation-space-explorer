# Qwen3-0.6B layer 16 OpenWebText demo

This demo streams exactly one million OpenWebText tokens, captures every token's activation after zero-based transformer layer 16, and fits 1,000 mini-batch k-means clusters. It soft-assigns a target activation to all centroids with `softmax(-distance / τ)`, samples neighbors from the top-K clusters, fits a **joint** 3D PCA on that multi-cluster sample, and writes an interactive Plotly view where **each cluster has its own color** plus a sidebar of **graded membership weights**.

This is a **neighborhood / soft-cluster decomposition**, not a causal SAE-style feature dictionary.

Settings live in `config.toml` (`[visualization]` controls top-K, sample size, τ, attention-sink prefix, and PCA L2 normalization).

## Quick start

```bash
# From the repository root. First run creates experiments/liusida/demo/.venv via uv.
# One-off target (auto writes to a NEW folder under output/runs/ — no config conflict)
uv run --project experiments/liusida/demo python experiments/liusida/demo/run.py \
  --prompt "Today is Monday." --token " Monday"

# Or edit [target] in config.toml, then:
uv run --project experiments/liusida/demo python experiments/liusida/demo/run.py
```

Optional flags: `--output-dir PATH`, `--library PATH`, `--reuse-output` (legacy single-folder behavior), `--keyword`.

## Output layout

| Path | Contents |
| --- | --- |
| `output/library/<hash>/` | Shared capture + clusters (~4 GiB). Built once per model/data/k-means config. |
| `output/runs/<timestamp>_<token>/` | Per-run `run.json`, target artifacts, HTML (auto-created). |
| `output/serve_cache/` | HTML pages built by the interactive server. |

## Interactive UI

```bash
# Must run on a GPU compute node (needs a forward pass for new prompts), e.g. after gpush:
cd /path/to/activation-space-explorer
uv run --project experiments/liusida/demo \
  python experiments/liusida/demo/serve.py --host 0.0.0.0 --port 7860
```

Use `--host 0.0.0.0` so the login node can reach the compute node. Then in the UI: type a prompt → **Analyze** → click a token chip → soft multi-cluster view.

### SSH tunnel from your laptop (Windows / macOS / Linux)

The server listens on the **GPU node** (e.g. `cml20`), not on your laptop. `http://127.0.0.1:7860` on your PC only works after a tunnel.

1. In the `gpush` terminal, note the hostname:

   ```bash
   hostname   # e.g. cml20
   ```

2. On your **local** machine (PowerShell / Terminal), forward a free local port to that node’s 7860 through the login host:

   ```bash
   # Prefer a high local port — Windows often blocks binding 7860 ("Permission denied")
   ssh -N -L 17860:cml20:7860 feijhan@nexuscml.umiacs.umd.edu
   ```

   Replace `cml20` with your `hostname`. Duo / password prompts are normal. The command then sits idle (that means forwarding is active).

3. Open in your local browser:

   ```text
   http://127.0.0.1:17860
   ```

If local port 7860 fails with `bind … Permission denied`, use `17860`, `8765`, or `18888` on the left side of `-L` only; the right side stays `<compute-node>:7860`.

```text
Laptop:17860  --ssh-->  nexuscml  --forward-->  cml20:7860 (serve.py)
```

## What we changed (feijianghan exploration)

Relative to the original single-cluster Friday demo:

### Visualization & interpretation
- **Multi-cluster soft decomposition**: top-K k-means soft weights + colored 3D joint PCA (not nearest-cluster-only).
- **Interactive `serve.py`**: prompt in → per-token soft assignment → click token → decomposition HTML.
- **Attention-sink prefix** (`visualization.attention_sink_prefix = "auto"`): prepend tokenizer pad (`<|endoftext|>` on Qwen3) so the first *user* token is not absolute model position 0 (avoids attention-sink outliers looking like “huge PC1”).
- **L2-normalize before joint PCA** (`visualization.l2_normalize_pca = true`): PC1 is not dominated by activation magnitude.

### Experiment ergonomics
- **Shared library + auto run dirs** (`library.py`): expensive capture/clusters live under `output/library/`; each run gets `output/runs/...` so changing `[target]` no longer hits “different configuration” on the same `run.json`.
- **CLI overrides**: `--prompt` / `--token` / `--output-dir` / `--library` / `--reuse-output` without editing toml for every trial.
- Shared analysis helpers in `analyze.py` used by both `run.py` and `serve.py`.

### Files touched / added
| File | Role |
| --- | --- |
| `analyze.py` | Soft weights, sink prefix, L2 PCA, prompt activation extract, HTML render helper |
| `library.py` | Library/run directory layout, migrate & symlink shared artifacts |
| `serve.py` | Local HTTP UI + `/api/analyze` + `/api/token_view` |
| `render.py` | Multi-cluster Plotly HTML + decomposition sidebar |
| `run.py` | Wire library/run dirs, CLI overrides, use shared analyze path |
| `config.toml` | `[visualization]` options |
| `tests/test_demo_render.py` | Cover decomposition HTML |

## Notes

- Large artifacts under `output/` are gitignored; do not commit activation matrices.
- uv creates this experiment’s `.venv` on first run; the root `.venv` is not required to run the demo.
