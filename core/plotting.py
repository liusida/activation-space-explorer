"""Matplotlib helpers for activation-space projections."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from core.explorer import AnalysisResult
from core.projection import Projection


def plot_global(
    result: AnalysisResult,
    *,
    ax: Axes | None = None,
    highlight: Sequence[str] = (),
) -> Axes:
    ax = ax or plt.subplots()[1]
    coordinates = result.global_projection.coordinates
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=result.assignments.cluster_ids,
        cmap="tab20",
        alpha=0.75,
    )
    _highlight(ax, result.global_projection, highlight)
    ax.set(xlabel="PC1", ylabel="PC2", title="Global activation space")
    return ax


def plot_cluster(
    result: AnalysisResult,
    cluster_id: int,
    *,
    ax: Axes | None = None,
    highlight: Sequence[str] = (),
) -> Axes:
    try:
        projection = result.local_projections[cluster_id]
    except KeyError as error:
        raise ValueError(f"no local projection for cluster {cluster_id}") from error
    ax = ax or plt.subplots()[1]
    coordinates = projection.coordinates
    ax.scatter(coordinates[:, 0], coordinates[:, 1], alpha=0.75)
    _highlight(ax, projection, highlight)
    ax.set(xlabel="Local PC1", ylabel="Local PC2", title=f"Cluster {cluster_id}")
    return ax


def _highlight(ax: Axes, projection: Projection, highlight: Sequence[str]) -> None:
    if not highlight or projection.sample_ids is None:
        return
    wanted = set(highlight)
    for sample_id, point in zip(projection.sample_ids, projection.coordinates, strict=True):
        if sample_id in wanted:
            ax.scatter(point[0], point[1], marker="*", s=140, color="black")
            ax.annotate(sample_id, point)
