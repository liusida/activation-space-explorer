"""Build component profiles for every fitted ICA Lens layer."""

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
        "--layers",
        help="layer selection passed to ICA Lens (default: config.toml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved ICA Lens command without running it",
    )
    return parser.parse_args()


def profile_command(layers: str | None = None) -> list[str]:
    with (HERE / "config.toml").open("rb") as file:
        config = tomllib.load(file)
    profile = config["profile"]
    lens_path = HERE / config["ica"]["output"]
    output_path = HERE / profile["output"]
    return [
        sys.executable,
        "-m",
        "icalens.cli",
        "profile",
        "--lens",
        str(lens_path),
        "--layers",
        layers or profile["layers"],
        "--output",
        str(output_path),
        "--dataset",
        profile["dataset"],
        "--split",
        profile["split"],
        "--text-field",
        profile["text_field"],
        "--input-type",
        profile["input_type"],
        "--max-tokens",
        str(profile["max_tokens"]),
        "--context-length",
        str(profile["context_length"]),
        "--top-k-examples",
        str(profile["top_k_examples"]),
        "--logit-lens-top-k",
        str(profile["logit_lens_top_k"]),
        "--logit-lens-batch-size",
        str(profile["logit_lens_batch_size"]),
        "--sample-seed",
        str(profile["sample_seed"]),
        "--device",
        profile["device"],
    ]


def main() -> None:
    arguments = parse_args()
    command = profile_command(arguments.layers)
    if arguments.dry_run:
        print(" ".join(command))
        return
    subprocess.run(command, cwd=HERE, check=True)


if __name__ == "__main__":
    main()
