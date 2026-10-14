"""Fit the all-layer ICA Lens used by the Qwen superposition experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved ICA Lens command without running it",
    )
    return parser.parse_args()


def fit_command() -> list[str]:
    with (HERE / "config.toml").open("rb") as file:
        config = tomllib.load(file)
    model = config["model"]
    ica = config["ica"]
    output = HERE / ica["output"]
    return [
        sys.executable,
        "-m",
        "icalens.cli",
        "fit",
        "text",
        "--model",
        model["id"],
        "--dataset",
        ica["dataset"],
        "--split",
        ica["split"],
        "--text-field",
        ica["text_field"],
        "--document-framing",
        ica["document_framing"],
        "--layers",
        ica["layers"],
        "--context-length",
        str(ica["context_length"]),
        "--candidate-tokens",
        str(ica["candidate_tokens"]),
        "--token-budget",
        str(ica["token_budget"]),
        "--capture-layers-at-once",
        str(ica["capture_layers_at_once"]),
        "--fit-batch-size",
        str(ica["fit_batch_size"]),
        "--max-iter",
        str(ica["max_iter"]),
        "--objective-every",
        str(ica["objective_every"]),
        "--icalens-preprocessing",
        ica["preprocessing"],
        "--seed",
        str(config["analysis"]["random_state"]),
        "--output",
        str(output),
    ]


def main() -> None:
    arguments = parse_args()
    command = fit_command()
    if arguments.dry_run:
        print(" ".join(command))
        return
    subprocess.run(command, cwd=HERE, check=True)


if __name__ == "__main__":
    main()
