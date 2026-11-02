"""Explore one million Qwen3-0.6B layer-16 activations from OpenWebText."""

from __future__ import annotations

import argparse
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

from core import (
    ActivationSet,
    ActivationSource,
    ActivationSpaceExplorer,
    ClusterAssignments,
    KMeansClusterer,
    PCAProjector,
    require_version,
)
from core.runs import ExperimentPlan, RunContext, Stage, run_experiment

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"


def load_config() -> dict[str, Any]:
    with (HERE / "config.toml").open("rb") as file:
        return tomllib.load(file)


class OpenWebTextTokens:
    def __init__(self, tokenizer: Any, config: dict[str, Any]) -> None:
        rows = load_dataset(config["dataset"], split=config["split"], streaming=True).shuffle(
            seed=config["shuffle_seed"], buffer_size=config["shuffle_buffer"]
        )
        self.rows = iter(rows)
        self.tokenizer = tokenizer
        self.pending: list[int] = []

    def take(self, count: int) -> list[int]:
        while len(self.pending) < count:
            text = next(self.rows).get("text", "")
            if not text:
                continue
            self.pending.extend(self.tokenizer.encode(text, add_special_tokens=False))
            self.pending.append(self.tokenizer.eos_token_id)
        result = self.pending[:count]
        del self.pending[:count]
        return result


def activation_paths() -> tuple[Path, Path, Path]:
    return (
        OUTPUT / "activations.f32",
        OUTPUT / "token_ids.i32",
        OUTPUT / "capture.json",
    )


def existing_capture(config: dict[str, Any], dimension: int) -> ActivationSet | None:
    activation_path, token_path, metadata_path = activation_paths()
    if not all(path.exists() for path in (activation_path, token_path, metadata_path)):
        return None
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "model_id": config["model"]["id"],
        "layer_index": config["model"]["layer_index"],
        "dataset": config["capture"]["dataset"],
        "tokens": config["capture"]["tokens"],
        "dimension": dimension,
        "activation_dtype": "float32",
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("existing capture does not match config.toml; use a fresh output folder")
    expected_bytes = int(config["capture"]["tokens"]) * dimension * np.dtype(np.float32).itemsize
    if activation_path.stat().st_size != expected_bytes:
        raise ValueError("existing activation file has an unexpected size")
    values = np.memmap(
        activation_path,
        mode="r",
        dtype=np.float32,
        shape=(int(config["capture"]["tokens"]), dimension),
    )
    return make_activation_set(values, config)


def make_activation_set(values: np.ndarray, config: dict[str, Any]) -> ActivationSet:
    return ActivationSet(
        values=values,
        sample_ids=None,
        metadata=None,
        source=ActivationSource(
            model_id=config["model"]["id"],
            location=config["model"]["location"],
            reduction="all_tokens",
        ),
    )


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
def capture(config: dict[str, Any], dimension: int, ctx: RunContext) -> ActivationSet:
    cached = existing_capture(config, dimension)
    if cached is not None:
        ctx.log("reusing existing one-million-token capture")
        return cached

    model_config = config["model"]
    capture_config = config["capture"]
    if capture_config["activation_dtype"] != "float32":
        raise ValueError("the demo currently requires activation_dtype = 'float32'")

    tokenizer, model = load_model(config)
    device = model_config["device"]
    layer_index = int(model_config["layer_index"])
    expected_location = f"model.model.layers.{layer_index}"
    if model_config["location"] != expected_location:
        raise ValueError(f"configured location must be {expected_location!r}")

    OUTPUT.mkdir(exist_ok=True)
    activation_path, token_path, metadata_path = activation_paths()
    activation_partial = activation_path.with_suffix(".f32.partial")
    token_partial = token_path.with_suffix(".i32.partial")
    if activation_partial.exists() or token_partial.exists():
        raise FileExistsError("partial capture found; inspect or remove it before retrying")

    token_target = int(capture_config["tokens"])
    activations = np.memmap(
        activation_partial,
        mode="w+",
        dtype=np.float32,
        shape=(token_target, dimension),
    )
    token_ids = np.memmap(token_partial, mode="w+", dtype=np.int32, shape=(token_target,))
    stream = OpenWebTextTokens(tokenizer, capture_config)
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: object, _inputs: object, output: object) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("layer output did not contain a tensor")
        captured["hidden"] = hidden.detach()

    handle = model.model.layers[layer_index].register_forward_hook(hook)
    written = 0
    context = int(capture_config["context_length"])
    model_batch_size = int(capture_config["model_batch_size"])
    try:
        with ctx.progress(token_target, "tokens") as progress:
            while written < token_target:
                remaining = token_target - written
                sequence_length = min(context, remaining)
                batch_size = min(model_batch_size, max(1, remaining // sequence_length))
                count = batch_size * sequence_length
                ids = torch.tensor(
                    [stream.take(sequence_length) for _ in range(batch_size)],
                    dtype=torch.long,
                    device=device,
                )
                captured.clear()
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                hidden = captured.get("hidden")
                if hidden is None or tuple(hidden.shape) != (
                    batch_size,
                    sequence_length,
                    dimension,
                ):
                    actual = None if hidden is None else tuple(hidden.shape)
                    raise RuntimeError(f"unexpected hooked activation shape: {actual}")
                stop = written + count
                activations[written:stop] = hidden.reshape(-1, dimension).float().cpu().numpy()
                token_ids[written:stop] = ids.reshape(-1).cpu().numpy()
                written = stop
                progress.update(written)
    finally:
        handle.remove()

    activations.flush()
    token_ids.flush()
    del activations, token_ids
    activation_partial.replace(activation_path)
    token_partial.replace(token_path)
    metadata = {
        "model_id": model_config["id"],
        "layer_index": layer_index,
        "location": model_config["location"],
        "dataset": capture_config["dataset"],
        "dataset_split": capture_config["split"],
        "tokens": token_target,
        "dimension": dimension,
        "activation_dtype": "float32",
        "token_dtype": "int32",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    values = np.memmap(
        activation_path,
        mode="r",
        dtype=np.float32,
        shape=(token_target, dimension),
    )
    return make_activation_set(values, config)


@torch.inference_mode()
def capture_target(config: dict[str, Any], dimension: int) -> dict[str, Any]:
    from analyze import extract_prompt_token_activations

    prompt = config["target"]["prompt"]
    target_text = config["target"]["token"]
    if prompt.count(target_text) != 1:
        raise ValueError("target.token must occur exactly once in target.prompt")
    character_start = prompt.index(target_text)
    character_stop = character_start + len(target_text)

    tokenizer, model = load_model(config)
    records = extract_prompt_token_activations(
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
        layer_index=int(config["model"]["layer_index"]),
        device=config["model"]["device"],
        dimension=dimension,
        config=config,
    )
    matches = [
        record
        for record in records
        if record["character_start"] == character_start
        and record["character_stop"] == character_stop
    ]
    if len(matches) != 1:
        pieces = [record["token"] for record in records]
        raise ValueError(
            f"target token {target_text!r} is not exactly one model token; pieces={pieces!r}"
        )
    record = matches[0]
    return {
        "activation": np.asarray(record["activation"], dtype=np.float32),
        "prompt": prompt,
        "token": target_text,
        "token_id": int(record["token_id"]),
        "token_position": int(record["position"]),
        "model_position": int(record.get("model_position", record["position"])),
        "character_start": character_start,
        "character_stop": character_stop,
        "attention_sink_prefix": record.get("attention_sink_prefix", ""),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keyword",
        default=None,
        help="highlight tokens whose stripped, lowercased text contains this value "
        "(default: stripped target.token)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="override config.toml [target].prompt for a one-off run",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="override config.toml [target].token (must occur once in --prompt/prompt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="run directory (default: auto output/runs/<timestamp>_<token>/)",
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=None,
        help="shared capture/cluster library (default: output/library/<base-hash>/)",
    )
    parser.add_argument(
        "--reuse-output",
        action="store_true",
        help="write into --output-dir or the legacy output/ folder without auto-creating a new run",
    )
    return parser.parse_args()


def main(
    *,
    keyword: str | None = None,
    prompt: str | None = None,
    token: str | None = None,
    output_dir: Path | None = None,
    library: Path | None = None,
    reuse_output: bool = False,
) -> None:
    global OUTPUT
    config = load_config()
    if prompt is not None:
        config.setdefault("target", {})["prompt"] = prompt
    if token is not None:
        config.setdefault("target", {})["token"] = token
    require_version(config["experiment"]["core_version"])

    from library import (
        OUTPUT_ROOT,
        default_run_dir,
        ensure_library_links,
        library_dir_for,
        migrate_flat_output_into_library,
        promote_shared_into_library,
        write_run_pointer,
    )

    migrated = migrate_flat_output_into_library(config)
    library_path = Path(library) if library is not None else (migrated or library_dir_for(config))
    if reuse_output:
        run_dir = Path(output_dir) if output_dir is not None else OUTPUT_ROOT
    else:
        run_dir = Path(output_dir) if output_dir is not None else default_run_dir(config)
    run_dir = run_dir.resolve()
    library_path = library_path.resolve()
    # Shared capture/cluster live in the library; each run gets its own run.json/html.
    if all((library_path / name).exists() for name in (
        "activations.f32", "token_ids.i32", "capture.json", "clusters.npz"
    )):
        ensure_library_links(library_path, run_dir)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[demo] library incomplete at {library_path}; "
            "capture/cluster will write into this run and should be promoted later."
        )
    write_run_pointer(run_dir, library=library_path, config=config)
    OUTPUT = run_dir
    print(f"[demo] library = {library_path}")
    print(f"[demo] run dir = {run_dir}")

    dimension = int(AutoConfig.from_pretrained(config["model"]["id"]).hidden_size)
    analysis = config["analysis"]
    components = int(analysis["pca_components"])
    batch_size = int(analysis["minibatch_size"])
    runtime: dict[str, Any] = {}

    def new_clusterer() -> KMeansClusterer:
        return KMeansClusterer(
            n_clusters=int(analysis["n_clusters"]),
            normalize=bool(analysis["normalize"]),
            random_state=int(analysis["random_state"]),
            n_init=1,
            minibatch=True,
            batch_size=batch_size,
            max_iter=int(analysis["kmeans_max_iter"]),
        )

    def target_cluster() -> int:
        target = runtime["target"]
        assignment = runtime["clusterer"].assign(
            target["activation"][None, :], return_distances=False
        )
        return int(assignment.cluster_ids[0])

    def validate_capture(_ctx: RunContext) -> bool:
        activations = existing_capture(config, dimension)
        if activations is not None:
            runtime["activations"] = activations
            return True
        return False

    def stage_capture(ctx: RunContext) -> None:
        activations = capture(config, dimension, ctx)
        runtime["activations"] = activations
        promote_shared_into_library(OUTPUT, library_path, config)
        ctx.complete({"tokens": len(activations.values), "dimension": dimension})

    def validate_clusters(ctx: RunContext) -> bool:
        path = ctx.output_dir / "clusters.npz"
        if not path.exists():
            return False
        with np.load(path, allow_pickle=False) as data:
            clusterer = new_clusterer()
            clusterer._centroids = data["centroids"]
            assignments = ClusterAssignments(data["cluster_ids"], None, data["nearest_distances"])
        runtime["clusterer"] = clusterer
        runtime["assignments"] = assignments
        return True

    def stage_clusters(ctx: RunContext) -> None:
        activations: ActivationSet = runtime["activations"]
        clusterer = new_clusterer()
        fit_batches = math.ceil(len(activations.values) / batch_size) * clusterer.max_iter
        assign_batches = math.ceil(len(activations.values) / batch_size)
        with ctx.progress(fit_batches + assign_batches, "batches") as progress:
            progress.substage("fitting centroids")
            clusterer.fit(activations, progress=lambda done, _total: progress.update(done))
            progress.substage("assigning reference activations")
            assignments = clusterer.assign(
                activations,
                return_distances=False,
                progress=lambda done, _total: progress.update(fit_batches + done),
            )
        ctx.save_npz_atomic(
            "clusters.npz",
            centroids=clusterer.centroids,
            cluster_ids=assignments.cluster_ids,
            nearest_distances=assignments.assigned_distances,
        )
        runtime["clusterer"] = clusterer
        runtime["assignments"] = assignments
        promote_shared_into_library(OUTPUT, library_path, config)
        ctx.complete({"clusters": clusterer.n_clusters})

    def validate_target(ctx: RunContext) -> bool:
        array_path = ctx.output_dir / "target.npz"
        metadata_path = ctx.output_dir / "target.json"
        if not array_path.exists() or not metadata_path.exists():
            return False
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("prompt") != config["target"]["prompt"]
            or metadata.get("token") != config["target"]["token"]
        ):
            return False
        with np.load(array_path, allow_pickle=False) as data:
            activation = data["activation"]
        if activation.shape != (dimension,):
            return False
        runtime["target"] = {**metadata, "activation": activation}
        return True

    def stage_target(ctx: RunContext) -> None:
        with ctx.progress(1, "prompt") as progress:
            progress.substage(f"capturing {config['target']['token']!r}")
            target = capture_target(config, dimension)
            progress.advance()
        activation = target.pop("activation")
        ctx.save_npz_atomic("target.npz", activation=activation)
        ctx.save_json_atomic("target.json", target)
        runtime["target"] = {**target, "activation": activation}
        cluster_id = target_cluster()
        ctx.complete(
            {
                "prompt": target["prompt"],
                "token": target["token"],
                "token_id": target["token_id"],
                "cluster": cluster_id,
            }
        )

    def validate_local_pca(ctx: RunContext) -> bool:
        path = ctx.output_dir / "explorer" / "manifest.json"
        if not path.exists():
            return False
        explorer = ActivationSpaceExplorer.load(path.parent)
        if target_cluster() not in explorer.local_projectors:
            return False
        runtime["explorer"] = explorer
        return True

    def stage_local_pca(ctx: RunContext) -> None:
        activations: ActivationSet = runtime["activations"]
        assignments: ClusterAssignments = runtime["assignments"]
        explorer = ActivationSpaceExplorer(
            runtime["clusterer"],
            None,
            local_components=components,
        )
        cluster_id = target_cluster()
        indices = np.flatnonzero(assignments.cluster_ids == cluster_id)
        if min(len(indices), activations.values.shape[1]) < components:
            raise ValueError(
                f"target cluster {cluster_id} has too few points for {components}D PCA"
            )
        with ctx.progress(1, "cluster") as progress:
            progress.substage(
                f"{config['target']['token']!r} → cluster {cluster_id:,} · {len(indices):,} points"
            )
            explorer.local_projectors[cluster_id] = PCAProjector(
                n_components=components,
                random_state=explorer.clusterer.random_state,
            ).fit(activations.select(indices))
            progress.advance()
        explorer.save(ctx.output_dir / "explorer")
        runtime["explorer"] = explorer
        ctx.complete(
            {
                "target_prompt": config["target"]["prompt"],
                "target_token": config["target"]["token"],
                "target_cluster": cluster_id,
                "points": len(indices),
            }
        )

    def stage_render(ctx: RunContext) -> None:
        from analyze import render_token_decomposition_html
        from library import slugify

        target = runtime["target"]
        highlight = (
            keyword
            if keyword is not None
            else str(config["target"]["token"]).strip().lower()
        )
        token_record = {
            "position": target["token_position"],
            "token_id": target["token_id"],
            "token": target["token"],
            "character_start": target["character_start"],
            "character_stop": target["character_stop"],
            "activation": target["activation"],
        }
        shared = {
            "activations": runtime["activations"],
            "token_ids": np.memmap(
                activation_paths()[1],
                mode="r",
                dtype=np.int32,
                shape=(len(runtime["activations"].values),),
            ),
            "clusterer": runtime["clusterer"],
            "assignments": runtime["assignments"],
            "dimension": dimension,
        }
        with ctx.progress(2, "steps") as progress:
            progress.substage("building soft multi-cluster view")
            token_slug = slugify(str(target["token"]), fallback="token")
            html_path = ctx.output_dir / f"target-{token_slug}-decomp-pca-3d.html"
            render_summary = render_token_decomposition_html(
                html_path,
                config=config,
                shared=shared,
                prompt=target["prompt"],
                token_record=token_record,
                keyword=highlight,
            )
            progress.advance()
            progress.substage("writing summary")
            summary = {
                "tokens": len(runtime["activations"].values),
                "activation_dimension": dimension,
                "clusters": int(analysis["n_clusters"]),
                "pca_components": components,
                "target_prompt": target["prompt"],
                "target_token": target["token"],
                "target_token_id": target["token_id"],
                "target_cluster": render_summary["hard_cluster"],
                "soft_top_clusters": render_summary["soft_top_clusters"],
                "softmax_temperature": render_summary["temperature"],
                "keyword": highlight.strip().lower(),
                "html": html_path.name,
                "run_dir": str(OUTPUT),
                "library": str(library_path),
                "estimated_activation_gib": math.prod(runtime["activations"].values.shape)
                * 4
                / 1024**3,
            }
            ctx.save_json_atomic("summary.json", summary)
            progress.advance()
        ctx.complete(summary)

    plan = ExperimentPlan(
        name="liusida/demo",
        output_dir=OUTPUT,
        config=config,
        description=(
            f"{config['model']['id']} · layer {config['model']['layer_index']} · "
            f"OpenWebText · {config['capture']['tokens']:,} tokens"
        ),
        stages=[
            Stage("capture", "Capture activations", stage_capture, validator=validate_capture),
            Stage(
                "cluster",
                "Fit k-means",
                stage_clusters,
                dependencies=("capture",),
                validator=validate_clusters,
            ),
            Stage(
                "target",
                "Capture target prompt",
                stage_target,
                dependencies=("cluster",),
                validator=validate_target,
            ),
            Stage(
                "local-pca",
                "Fit target-cluster PCA",
                stage_local_pca,
                dependencies=("target",),
                validator=validate_local_pca,
            ),
            Stage(
                "render",
                "Render visualization",
                stage_render,
                durable=False,
                dependencies=("local-pca",),
            ),
        ],
    )
    run_experiment(plan)


if __name__ == "__main__":
    arguments = parse_args()
    main(
        keyword=arguments.keyword,
        prompt=arguments.prompt,
        token=arguments.token,
        output_dir=arguments.output_dir,
        library=arguments.library,
        reuse_output=arguments.reuse_output,
    )
