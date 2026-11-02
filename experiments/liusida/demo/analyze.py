"""Shared analysis helpers for batch runs and the interactive server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from render import write_activation_decomposition_html
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import ActivationSet, ActivationSource, ClusterAssignments, KMeansClusterer, PCAProjector
from gb10_load_llm import from_pretrained_to_cuda


def soft_cluster_weights(
    activation: np.ndarray,
    centroids: np.ndarray,
    *,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    distances = np.linalg.norm(centroids - activation[None, :], axis=1)
    used = float(temperature)
    if used <= 0.0:
        used = float(np.median(distances)) + 1e-6
    shifted = distances - float(distances.min())
    unnormalized = np.exp(-shifted / used)
    weights = unnormalized / max(float(unnormalized.sum()), 1e-12)
    return distances.astype(np.float64), weights.astype(np.float64), used


def l2_normalize_rows(values: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Row-wise L2 normalize so PCA is not dominated by activation magnitude."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        denom = float(np.linalg.norm(array))
        return array / max(denom, eps)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, eps)


def resolve_attention_sink_prefix(tokenizer: Any, config: dict[str, Any]) -> str:
    """Prefix that absorbs absolute position-0 / attention-sink effects.

    Qwen3 has no bos_token; without a prefix the first user token sits at
    position 0 and becomes an activation-space outlier relative to mid-context
    OpenWebText references.
    """
    viz = config.get("visualization", {})
    configured = viz.get("attention_sink_prefix", "auto")
    if configured is None or configured is False or configured == "":
        return ""
    if configured == "auto" or configured is True:
        if getattr(tokenizer, "pad_token", None):
            return str(tokenizer.pad_token)
        if getattr(tokenizer, "eos_token", None):
            return str(tokenizer.eos_token)
        return "\n"
    return str(configured)


def load_shared_artifacts(
    library: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    analysis = config["analysis"]
    capture = config["capture"]
    n_tokens = int(capture["tokens"])
    activation_path = library / "activations.f32"
    bytes_total = activation_path.stat().st_size
    dimension = bytes_total // (n_tokens * 4)
    if dimension * n_tokens * 4 != bytes_total:
        raise ValueError(f"unexpected activations.f32 size under {library}")
    values = np.memmap(
        activation_path, mode="r", dtype=np.float32, shape=(n_tokens, dimension)
    )
    token_ids = np.memmap(
        library / "token_ids.i32", mode="r", dtype=np.int32, shape=(n_tokens,)
    )
    with np.load(library / "clusters.npz", allow_pickle=False) as data:
        centroids = np.asarray(data["centroids"])
        cluster_ids = np.asarray(data["cluster_ids"])
        nearest = np.asarray(data["nearest_distances"])
    clusterer = KMeansClusterer(
        n_clusters=int(analysis["n_clusters"]),
        normalize=bool(analysis["normalize"]),
        random_state=int(analysis["random_state"]),
        n_init=1,
        minibatch=True,
        batch_size=int(analysis["minibatch_size"]),
        max_iter=int(analysis["kmeans_max_iter"]),
    )
    clusterer._centroids = centroids
    activations = ActivationSet(
        values=values,
        sample_ids=None,
        metadata=None,
        source=ActivationSource(
            model_id=config["model"]["id"],
            location=config["model"]["location"],
            reduction="all_tokens",
        ),
    )
    return {
        "activations": activations,
        "token_ids": token_ids,
        "clusterer": clusterer,
        "assignments": ClusterAssignments(cluster_ids, None, nearest),
        "dimension": dimension,
    }


def load_model(config: dict[str, Any]) -> tuple[Any, Any]:
    model_config = config["model"]
    device = model_config["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(model_config["id"])
    model = from_pretrained_to_cuda(
        AutoModelForCausalLM,
        model_config["id"],
        device=device,
        dtype=getattr(torch, model_config["dtype"]),
        low_cpu_mem_usage=False,
    )
    model.eval().requires_grad_(False)
    return tokenizer, model


@torch.inference_mode()
def extract_prompt_token_activations(
    *,
    tokenizer: Any,
    model: Any,
    prompt: str,
    layer_index: int,
    device: str,
    dimension: int,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Forward once; return one record per tokenizer piece with its activation.

    When ``visualization.attention_sink_prefix`` is enabled (default), a sink
    token is prepended so the first *user* token is not absolute position 0.
    Returned ``position`` / character offsets are relative to ``prompt`` only.
    """
    prefix = resolve_attention_sink_prefix(tokenizer, config or {})
    prefix_ids = (
        tokenizer.encode(prefix, add_special_tokens=False) if prefix else []
    )
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    prompt_ids = encoded["input_ids"][0].tolist()
    full_ids = prefix_ids + prompt_ids
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: object, _inputs: object, output: object) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("layer output did not contain a tensor")
        captured["hidden"] = hidden.detach()

    handle = model.model.layers[layer_index].register_forward_hook(hook)
    try:
        input_ids = torch.tensor([full_ids], device=device)
        attention_mask = torch.ones_like(input_ids)
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    hidden = captured.get("hidden")
    if hidden is None or hidden.shape[-1] != dimension:
        actual = None if hidden is None else tuple(hidden.shape)
        raise RuntimeError(f"unexpected activation shape: {actual}")
    matrix = hidden[0].float().cpu().numpy()
    offset = len(prefix_ids)
    if matrix.shape[0] != offset + len(prompt_ids):
        raise RuntimeError(
            f"activation length {matrix.shape[0]} != prefix+prompt {offset + len(prompt_ids)}"
        )
    records: list[dict[str, Any]] = []
    for position, (token_id, (start, stop)) in enumerate(zip(prompt_ids, offsets, strict=True)):
        piece = tokenizer.decode([token_id])
        records.append(
            {
                "position": position,
                "model_position": offset + position,
                "token_id": int(token_id),
                "token": piece,
                "character_start": int(start),
                "character_stop": int(stop),
                "activation": matrix[offset + position],
                "attention_sink_prefix": prefix,
            }
        )
    return records


def summarize_token(
    activation: np.ndarray,
    *,
    clusterer: KMeansClusterer,
    top_k: int,
    temperature: float,
) -> dict[str, Any]:
    distances, weights, used_temperature = soft_cluster_weights(
        np.asarray(activation, dtype=np.float32),
        clusterer.centroids,
        temperature=temperature,
    )
    order = np.argsort(-weights)[: max(1, min(top_k, len(weights)))]
    hard = int(np.argmin(distances))
    return {
        "hard_cluster": hard,
        "temperature": used_temperature,
        "top": [
            {
                "cluster_id": int(cluster_id),
                "weight": float(weights[cluster_id]),
                "distance": float(distances[cluster_id]),
                "rank": rank,
            }
            for rank, cluster_id in enumerate(order, start=1)
        ],
    }


def render_token_decomposition_html(
    path: Path,
    *,
    config: dict[str, Any],
    shared: dict[str, Any],
    prompt: str,
    token_record: dict[str, Any],
    keyword: str | None = None,
) -> dict[str, Any]:
    """Build the multi-cluster Plotly HTML for one token activation."""
    viz = config.get("visualization", {})
    analysis = config["analysis"]
    top_k = int(viz.get("top_k_clusters", 8))
    points_per_cluster = int(viz.get("points_per_cluster", 120))
    temperature = float(viz.get("softmax_temperature", 0.0))
    components = int(analysis["pca_components"])
    highlight = (
        keyword
        if keyword is not None
        else str(token_record["token"]).strip().lower()
    )

    activations: ActivationSet = shared["activations"]
    assignments: ClusterAssignments = shared["assignments"]
    clusterer: KMeansClusterer = shared["clusterer"]
    token_ids = shared["token_ids"]

    summary = summarize_token(
        token_record["activation"],
        clusterer=clusterer,
        top_k=top_k,
        temperature=temperature,
    )
    distances, weights, used_temperature = soft_cluster_weights(
        np.asarray(token_record["activation"], dtype=np.float32),
        clusterer.centroids,
        temperature=temperature,
    )
    top = [item["cluster_id"] for item in summary["top"]]
    hard_cluster = summary["hard_cluster"]

    rng = np.random.default_rng(int(analysis["random_state"]) + int(token_record["position"]))
    sampled_rows: list[int] = []
    sampled_cluster_ids: list[int] = []
    decomposition: list[dict[str, Any]] = []
    for rank, cluster_id in enumerate(top, start=1):
        members = np.flatnonzero(assignments.cluster_ids == cluster_id)
        if not len(members):
            continue
        chosen = (
            rng.choice(members, size=points_per_cluster, replace=False)
            if len(members) > points_per_cluster
            else members
        )
        chosen = np.asarray(chosen, dtype=np.int64)
        sampled_rows.extend(int(row) for row in chosen)
        sampled_cluster_ids.extend([int(cluster_id)] * len(chosen))
        decomposition.append(
            {
                "cluster_id": int(cluster_id),
                "weight": float(weights[cluster_id]),
                "distance": float(distances[cluster_id]),
                "rank": rank,
                "n_shown": int(len(chosen)),
                "n_total": int(len(members)),
                "sample_tokens": [],
            }
        )
    if not sampled_rows:
        raise ValueError("no reference members in top soft-assigned clusters")

    # Joint PCA on raw activations is often dominated by L2-norm / sink outliers
    # (PC1 ≈ 99%). Default: row-wise L2 normalize so geometry reflects direction.
    use_l2 = bool(viz.get("l2_normalize_pca", True))
    sample_values = np.asarray(activations.select(sampled_rows).values, dtype=np.float32)
    target_values = np.asarray(token_record["activation"], dtype=np.float32)[None, :]
    if use_l2:
        sample_values = l2_normalize_rows(sample_values)
        target_values = l2_normalize_rows(target_values)
    projector = PCAProjector(
        n_components=components,
        random_state=int(analysis["random_state"]),
    ).fit(sample_values)
    sample_projection = projector.transform(sample_values)
    target_coordinates = projector.transform(target_values).coordinates
    coordinates = np.vstack((sample_projection.coordinates, target_coordinates))

    context_length = int(config["capture"]["context_length"])
    radius = 32
    before_ids: list[list[int]] = []
    current_ids: list[list[int]] = []
    after_ids: list[list[int]] = []
    for row in sampled_rows:
        row = int(row)
        sequence_start = (row // context_length) * context_length
        sequence_stop = min(sequence_start + context_length, len(token_ids))
        before_ids.append(token_ids[max(sequence_start, row - radius) : row].tolist())
        current_ids.append([int(token_ids[row])])
        after_ids.append(token_ids[row + 1 : min(sequence_stop, row + radius + 1)].tolist())

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["id"])
    decode_options = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    before_text = tokenizer.batch_decode(before_ids, **decode_options)
    token_text = tokenizer.batch_decode(current_ids, **decode_options)
    after_text = tokenizer.batch_decode(after_ids, **decode_options)
    details = [
        {
            "row": int(row),
            "token_id": int(token_ids[row]),
            "sequence": int(row) // context_length,
            "position": int(row) % context_length,
            "before": before,
            "token": token,
            "after": after,
        }
        for row, before, token, after in zip(
            sampled_rows, before_text, token_text, after_text, strict=True
        )
    ]
    tokens_by_cluster: dict[int, list[str]] = {}
    for cluster_id, token in zip(sampled_cluster_ids, token_text, strict=True):
        bucket = tokens_by_cluster.setdefault(int(cluster_id), [])
        if token not in bucket and len(bucket) < 12:
            bucket.append(token)
    for item in decomposition:
        item["sample_tokens"] = tokens_by_cluster.get(int(item["cluster_id"]), [])

    details.append(
        {
            "row": "external",
            "token_id": int(token_record["token_id"]),
            "sequence": "prompt",
            "position": int(token_record["position"]),
            "before": prompt[: int(token_record["character_start"])],
            "token": token_record["token"],
            "after": prompt[int(token_record["character_stop"]) :],
        }
    )
    point_cluster_ids = list(sampled_cluster_ids) + [hard_cluster]
    write_activation_decomposition_html(
        path,
        coordinates=coordinates,
        explained_variance_ratio=sample_projection.explained_variance_ratio,
        details=details,
        cluster_ids=point_cluster_ids,
        target_position=len(details) - 1,
        decomposition=decomposition,
        model_id=config["model"]["id"],
        layer_index=int(config["model"]["layer_index"]),
        keyword=highlight,
        temperature=used_temperature,
    )
    return {
        "html": path.name,
        "hard_cluster": hard_cluster,
        "temperature": used_temperature,
        "soft_top_clusters": summary["top"],
    }
