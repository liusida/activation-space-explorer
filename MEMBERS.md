# Member workflow

Always develop your experiment on its own branch. Do not add unfinished work directly to `main`.

```bash
git switch main
git pull
git switch -c experiment/<member>-<experiment>
```

Use the shared library in `core/` and the runner in `core/runs/`. Keep the experiment reproducible with a `config.toml`, pin the required core version, and begin with a small smoke test before starting a long run.

Give the experiment its own `.python-version`, `pyproject.toml`, and `uv.lock`. Declare the root library as an editable local dependency so experiment-specific packages never need to be added to the root project:

```toml
[project]
dependencies = ["activation-space-explorer"]

[tool.uv.sources]
activation-space-explorer = { path = "../../..", editable = true }
```

Run it from the repository root with `uv run --project experiments/<member>/<experiment> python experiments/<member>/<experiment>/run.py`.

Keep large generated files under `output/`; do not commit activation matrices, model weights, datasets, or other large intermediates. When a result is promising, include a concise explanation, the exact configuration and command, and small result artifacts needed to evaluate it.

Open a pull request to `main` when the result is ready. After review and reproducibility audit, the accepted experiment will live at:

```text
experiments/<member>/<experiment>/
├── README.md
├── .python-version
├── config.toml
├── pyproject.toml
├── run.py
├── uv.lock
└── results/        # optional, curated and small
```

The `core/` library is currently maintained by **liusida**. If you find a core issue or need a new capability, open an issue first. You may instead submit a core pull request when you are confident that the solution is general, tested, documented, and ready for other experiments to depend on.
