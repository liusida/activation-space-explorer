"""Capture and explore Qwen3-0.6B layer-16 activations on GB10."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from gb10_load_llm import from_pretrained_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import (
    ActivationSet,
    ActivationSource,
    ActivationSpaceExplorer,
    KMeansClusterer,
    PCAProjector,
    plot_cluster,
    plot_global,
    require_version,
)

HERE = Path(__file__).resolve().parent


def load_config() -> dict[str, Any]:
    with (HERE / "config.toml").open("rb") as file:
        return tomllib.load(file)


def load_prompts() -> list[str]:
    prompts = [line.strip() for line in (HERE / "prompts.txt").read_text().splitlines()]
    return [prompt for prompt in prompts if prompt]


@torch.inference_mode()
def capture(config: dict[str, Any], prompts: list[str]) -> ActivationSet:
    model_config = config["model"]
    capture_config = config["capture"]
    if capture_config["reduction"] != "last_token":
        raise ValueError("the demo currently supports only reduction = 'last_token'")
    model_id = model_config["id"]
    device = model_config["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = from_pretrained_to_cuda(
        AutoModelForCausalLM,
        model_id,
        device=device,
        dtype=getattr(torch, model_config["dtype"]),
        low_cpu_mem_usage=False,
    )
    model.eval().requires_grad_(False)
    layer_index = int(model_config["layer_index"])
    expected_location = f"model.model.layers.{layer_index}"
    if model_config["location"] != expected_location:
        raise ValueError(f"configured location must be {expected_location!r}")
    layer = model.model.layers[layer_index]
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: object, _inputs: object, output: object) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("layer output did not contain a tensor")
        captured["hidden"] = hidden.detach()

    rows: list[np.ndarray] = []
    handle = layer.register_forward_hook(hook)
    batch_size = int(capture_config["batch_size"])
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            tokens = tokenizer(batch, return_tensors="pt", padding=True)
            tokens = {name: value.to(device) for name, value in tokens.items()}
            captured.clear()
            model(**tokens, use_cache=False)
            hidden = captured.get("hidden")
            if hidden is None:
                raise RuntimeError("layer hook did not run")
            last_token = tokens["attention_mask"].sum(dim=1) - 1
            batch_index = torch.arange(len(batch), device=hidden.device)
            rows.append(hidden[batch_index, last_token].float().cpu().numpy())
    finally:
        handle.remove()

    values = np.concatenate(rows, axis=0)
    return ActivationSet(
        values=values,
        sample_ids=[f"prompt-{index:03d}" for index in range(len(prompts))],
        metadata=[{"prompt": prompt} for prompt in prompts],
        source=ActivationSource(
            model_id=model_id,
            location=model_config["location"],
            reduction=capture_config["reduction"],
        ),
    )


def main() -> None:
    config = load_config()
    require_version(config["experiment"]["core_version"])
    prompts = load_prompts()
    activations = capture(config, prompts)
    analysis = config["analysis"]
    explorer = ActivationSpaceExplorer(
        KMeansClusterer(
            n_clusters=int(analysis["n_clusters"]),
            normalize=bool(analysis["normalize"]),
            random_state=int(analysis["random_state"]),
        ),
        PCAProjector(
            n_components=int(analysis["pca_components"]),
            random_state=int(analysis["random_state"]),
        ),
        local_components=int(analysis["pca_components"]),
    ).fit(activations)
    result = explorer.analyze(activations)

    output = HERE / "output"
    output.mkdir(exist_ok=True)
    activations.save(output / "activations.npz")
    explorer.save(output / "explorer")

    prompt_zero_cluster = int(result.assignments.cluster_ids[0])
    if not result.local_projections:
        raise RuntimeError("no cluster contained enough points for a local PCA view")
    local_cluster = (
        prompt_zero_cluster
        if prompt_zero_cluster in result.local_projections
        else next(iter(result.local_projections))
    )
    highlighted_index = int(np.flatnonzero(result.assignments.cluster_ids == local_cluster)[0])
    highlighted_id = activations.sample_ids[highlighted_index]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_global(result, ax=axes[0], highlight=[highlighted_id])
    plot_cluster(result, local_cluster, ax=axes[1], highlight=[highlighted_id])
    figure.tight_layout()
    figure.savefig(output / "projection.png", dpi=160)
    plt.close(figure)
    print(f"captured {len(activations.values)} activations with shape {activations.values.shape}")
    print(f"prompt-000 belongs to cluster {prompt_zero_cluster}")
    print(f"saved results to {output}")


if __name__ == "__main__":
    main()
