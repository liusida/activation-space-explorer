"""Fit layer-5 k-means and assign the colored-shape target activation."""

from __future__ import annotations

import json
import math
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from gb10_load_llm import from_pretrained_to_cuda
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from core import ActivationSet, ActivationSource, ClusterAssignments, KMeansClusterer, PCAProjector
from core.runs import ExperimentPlan, RunContext, Stage, run_experiment
from experiments.liusida.demo.render import write_cluster_pca_html

HERE = Path(__file__).resolve().parent


def load_config() -> dict[str, Any]:
    with (HERE / "config.toml").open("rb") as file:
        return tomllib.load(file)


class TokenStream:
    def __init__(self, tokenizer: Any, config: dict[str, Any]) -> None:
        rows = load_dataset(config["dataset"], split=config["split"], streaming=True)
        self.rows = iter(rows)
        self.tokenizer = tokenizer
        self.config = config
        self.pending: list[int] = []

    def take(self, count: int) -> list[int]:
        while len(self.pending) < count:
            text = next(self.rows).get(self.config["text_field"], "")
            if not text:
                continue
            if self.config["document_framing"] == "prepend-eos":
                self.pending.append(self.tokenizer.eos_token_id)
            self.pending.extend(self.tokenizer.encode(text, add_special_tokens=False))
        values = self.pending[:count]
        del self.pending[:count]
        return values


def activation_set(values: np.ndarray, model_id: str, layer: int) -> ActivationSet:
    return ActivationSet(
        values=values,
        sample_ids=None,
        metadata=None,
        source=ActivationSource(
            model_id=model_id,
            location=f"model.model.layers.{layer}",
            reduction="all_tokens",
        ),
    )


def load_model(config: dict[str, Any]) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(config["id"])
    model = from_pretrained_to_cuda(
        AutoModelForCausalLM,
        config["id"],
        device=config["device"],
        dtype=getattr(torch, config["dtype"]),
        low_cpu_mem_usage=False,
    )
    model.eval().requires_grad_(False)
    return tokenizer, model


def main() -> None:
    config = load_config()
    model_config = config["model"]
    kmeans = config["kmeans"]
    layer = int(kmeans["layer"])
    architecture = AutoConfig.from_pretrained(model_config["id"])
    text_config = getattr(architecture, "text_config", architecture)
    dimension = int(text_config.hidden_size)
    output = HERE / kmeans["output"]
    activation_path = output / "activations.f32"
    token_path = output / "token_ids.i32"
    capture_path = output / "capture.json"
    runtime: dict[str, Any] = {}

    def open_capture() -> ActivationSet:
        values = np.memmap(
            activation_path,
            mode="r",
            dtype=np.float32,
            shape=(int(kmeans["tokens"]), dimension),
        )
        return activation_set(values, model_config["id"], layer)

    def validate_capture(_ctx: RunContext) -> bool:
        if not activation_path.exists() or not capture_path.exists():
            return False
        metadata = json.loads(capture_path.read_text())
        expected = {
            "model_id": model_config["id"],
            "layer": layer,
            "tokens": int(kmeans["tokens"]),
            "dimension": dimension,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False
        if activation_path.stat().st_size != expected["tokens"] * dimension * 4:
            return False
        runtime["activations"] = open_capture()
        return True

    @torch.inference_mode()
    def capture(ctx: RunContext) -> None:
        tokenizer, model = load_model(model_config)
        stream = TokenStream(tokenizer, kmeans)
        total = int(kmeans["tokens"])
        context = int(kmeans["context_length"])
        batch_limit = int(kmeans["model_batch_size"])
        partial = activation_path.with_suffix(".f32.partial")
        values = np.memmap(partial, mode="w+", dtype=np.float32, shape=(total, dimension))
        captured: dict[str, torch.Tensor] = {}

        def hook(_module: object, _inputs: object, result: object) -> None:
            hidden = result[0] if isinstance(result, tuple) else result
            if not isinstance(hidden, torch.Tensor):
                raise TypeError("layer output did not contain a tensor")
            captured["hidden"] = hidden.detach()

        handle = model.model.layers[layer].register_forward_hook(hook)
        written = 0
        try:
            with ctx.progress(total, "tokens") as progress:
                while written < total:
                    length = min(context, total - written)
                    batch = min(batch_limit, max(1, (total - written) // length))
                    ids = torch.tensor(
                        [stream.take(length) for _ in range(batch)],
                        device=model_config["device"],
                    )
                    captured.clear()
                    model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                    hidden = captured["hidden"]
                    count = batch * length
                    values[written : written + count] = (
                        hidden.reshape(-1, dimension).float().cpu().numpy()
                    )
                    written += count
                    progress.update(written)
        finally:
            handle.remove()
            values.flush()
            del values, model
            torch.cuda.empty_cache()
        partial.replace(activation_path)
        ctx.save_json_atomic(
            "capture.json",
            {
                "model_id": model_config["id"],
                "layer": layer,
                "tokens": total,
                "dimension": dimension,
                "dataset": kmeans["dataset"],
                "activation_dtype": "float32",
            },
        )
        runtime["activations"] = open_capture()
        ctx.complete({"tokens": total, "dimension": dimension})

    def new_clusterer() -> KMeansClusterer:
        return KMeansClusterer(
            int(kmeans["n_clusters"]),
            random_state=int(config["analysis"]["random_state"]),
            n_init=1,
            minibatch=True,
            batch_size=int(kmeans["minibatch_size"]),
            max_iter=int(kmeans["max_iter"]),
        )

    def validate_clusters(ctx: RunContext) -> bool:
        path = ctx.output_dir / "clusters.npz"
        if not path.exists():
            return False
        with np.load(path, allow_pickle=False) as data:
            clusterer = new_clusterer()
            clusterer._centroids = data["centroids"]
            assignments = ClusterAssignments(data["cluster_ids"], None, data["nearest_distances"])
        runtime.update(clusterer=clusterer, assignments=assignments)
        return True

    def fit_clusters(ctx: RunContext) -> None:
        activations = runtime["activations"]
        clusterer = new_clusterer()
        batch_size = int(kmeans["minibatch_size"])
        fit_steps = math.ceil(len(activations.values) / batch_size) * int(kmeans["max_iter"])
        assign_steps = math.ceil(len(activations.values) / batch_size)
        with ctx.progress(fit_steps + assign_steps, "batches") as progress:
            progress.substage("fitting centroids")
            clusterer.fit(activations, progress=lambda done, _total: progress.update(done))
            progress.substage("assigning corpus activations")
            assignments = clusterer.assign(
                activations,
                return_distances=False,
                progress=lambda done, _total: progress.update(fit_steps + done),
            )
        ctx.save_npz_atomic(
            "clusters.npz",
            centroids=clusterer.centroids,
            cluster_ids=assignments.cluster_ids,
            nearest_distances=assignments.nearest_distances,
        )
        runtime.update(clusterer=clusterer, assignments=assignments)
        ctx.complete({"clusters": int(kmeans["n_clusters"])})

    @torch.inference_mode()
    def assign_target(ctx: RunContext) -> None:
        tokenizer, model = load_model(model_config)
        prompt = kmeans["target_prompt"]
        target = kmeans["target_token"]
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
        offsets = encoded.pop("offset_mapping")[0].tolist()
        positions = [
            i
            for i, (start, stop) in enumerate(offsets)
            if prompt[start:stop].strip() == target
        ]
        if len(positions) != 1:
            raise ValueError(f"target {target!r} is not exactly one token: {offsets}")
        position = positions[0]
        captured: dict[str, torch.Tensor] = {}

        def hook(_module: object, _inputs: object, result: object) -> None:
            hidden = result[0] if isinstance(result, tuple) else result
            captured["hidden"] = hidden.detach()

        handle = model.model.layers[layer].register_forward_hook(hook)
        try:
            inputs = {key: value.to(model_config["device"]) for key, value in encoded.items()}
            model(**inputs, use_cache=False)
            activation = captured["hidden"][0, position].float().cpu().numpy()
        finally:
            handle.remove()
            del model
            torch.cuda.empty_cache()
        assignment = runtime["clusterer"].assign(activation[None, :], return_distances=False)
        cluster = int(assignment.cluster_ids[0])
        distance = float(assignment.nearest_distances[0])
        size = int(np.count_nonzero(runtime["assignments"].cluster_ids == cluster))
        result = {
            "model_id": model_config["id"],
            "layer": layer,
            "prompt": prompt,
            "target_token": target,
            "token_position": position,
            "character_start": offsets[position][0],
            "character_stop": offsets[position][1],
            "cluster": cluster,
            "distance": distance,
            "cluster_size": size,
        }
        ctx.save_npz_atomic("target.npz", activation=activation)
        ctx.save_json_atomic("target_cluster.json", result)
        runtime["target"] = {**result, "activation": activation}
        ctx.complete(result)
        print(json.dumps(result, indent=2))

    def validate_tokens(_ctx: RunContext) -> bool:
        return token_path.exists() and token_path.stat().st_size == int(kmeans["tokens"]) * 4

    def reconstruct_tokens(ctx: RunContext) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_config["id"])
        stream = TokenStream(tokenizer, kmeans)
        total = int(kmeans["tokens"])
        partial = token_path.with_suffix(".i32.partial")
        token_ids = np.memmap(partial, mode="w+", dtype=np.int32, shape=(total,))
        chunk = 100_000
        with ctx.progress(total, "tokens") as progress:
            for start in range(0, total, chunk):
                stop = min(start + chunk, total)
                token_ids[start:stop] = stream.take(stop - start)
                progress.update(stop)
        token_ids.flush()
        del token_ids
        partial.replace(token_path)
        ctx.complete({"tokens": total})

    def validate_pca(ctx: RunContext) -> bool:
        path = ctx.output_dir / "cluster-864-pca.npz"
        if not path.exists():
            return False
        with np.load(path, allow_pickle=False) as data:
            projector = PCAProjector(n_components=3)
            projector._components = data["components"]
            projector._mean = data["mean"]
            projector._explained_variance = data["explained_variance"]
            projector._explained_variance_ratio = data["explained_variance_ratio"]
        runtime["projector"] = projector
        return True

    def fit_local_pca(ctx: RunContext) -> None:
        cluster = 864
        indices = np.flatnonzero(runtime["assignments"].cluster_ids == cluster)
        projector = PCAProjector(
            n_components=3,
            random_state=int(config["analysis"]["random_state"]),
        ).fit(runtime["activations"].select(indices))
        ctx.save_npz_atomic(
            "cluster-864-pca.npz",
            components=projector.components,
            mean=projector.mean,
            explained_variance=projector.explained_variance,
            explained_variance_ratio=projector.explained_variance_ratio,
        )
        runtime["projector"] = projector
        ctx.complete({"cluster": cluster, "points": len(indices), "components": 3})

    def render(ctx: RunContext) -> None:
        cluster = 864
        target = runtime["target"]
        if target["cluster"] != cluster:
            raise RuntimeError(f"target moved from cluster {cluster} to {target['cluster']}")
        indices = np.flatnonzero(runtime["assignments"].cluster_ids == cluster)
        projector: PCAProjector = runtime["projector"]
        reference = projector.transform(runtime["activations"].select(indices)).coordinates
        target_coordinates = projector.transform(target["activation"][None, :]).coordinates
        coordinates = np.vstack((reference, target_coordinates))
        token_ids = np.memmap(token_path, mode="r", dtype=np.int32, shape=(int(kmeans["tokens"]),))
        tokenizer = AutoTokenizer.from_pretrained(model_config["id"])
        context_length = int(kmeans["context_length"])
        radius = 32
        details: list[dict[str, Any]] = []
        for row in indices:
            row = int(row)
            sequence_start = (row // context_length) * context_length
            sequence_stop = min(sequence_start + context_length, len(token_ids))
            options = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
            details.append(
                {
                    "row": row,
                    "token_id": int(token_ids[row]),
                    "sequence": row // context_length,
                    "position": row % context_length,
                    "before": tokenizer.decode(token_ids[max(sequence_start, row - radius) : row].tolist(), **options),
                    "token": tokenizer.decode([int(token_ids[row])], **options),
                    "after": tokenizer.decode(token_ids[row + 1 : min(sequence_stop, row + radius + 1)].tolist(), **options),
                }
            )
        details.append(
            {
                "row": "external",
                "token_id": tokenizer.encode(target["prompt"], add_special_tokens=False)[target["token_position"]],
                "sequence": "prompt",
                "position": target["token_position"],
                "before": target["prompt"][: target["character_start"]],
                "token": target["prompt"][target["character_start"] : target["character_stop"]],
                "after": target["prompt"][target["character_stop"] :],
            }
        )
        html_path = ctx.output_dir / "cluster-864-pca-3d.html"
        write_cluster_pca_html(
            html_path,
            coordinates=coordinates,
            explained_variance_ratio=projector.explained_variance_ratio,
            details=details,
            target_position=len(details) - 1,
            model_id=model_config["id"],
            layer_index=layer,
            cluster_id=cluster,
            keyword="square",
        )
        ctx.complete({"html": html_path.name, "cluster": cluster, "points": len(indices)})

    plan = ExperimentPlan(
        name="liusida/superposition/kmeans-layer5",
        output_dir=output,
        config={"model": model_config, "kmeans": kmeans, "random_state": config["analysis"]["random_state"]},
        description=f"{model_config['id']} · layer {layer} · {int(kmeans['tokens']):,} tokens · {int(kmeans['n_clusters']):,} clusters",
        stages=[
            Stage("capture", "Capture layer-5 activations", capture, validator=validate_capture),
            Stage("cluster", "Fit layer-5 k-means", fit_clusters, dependencies=("capture",), validator=validate_clusters),
            Stage("target", "Assign target activation", assign_target, dependencies=("cluster",), durable=False),
            Stage("tokens", "Reconstruct token contexts", reconstruct_tokens, dependencies=("capture",), validator=validate_tokens),
            Stage("pca", "Fit cluster-864 PCA-3D", fit_local_pca, dependencies=("cluster",), validator=validate_pca),
            Stage("render", "Render cluster-864 HTML", render, dependencies=("target", "tokens", "pca"), durable=False),
        ],
    )
    run_experiment(plan)


if __name__ == "__main__":
    main()
