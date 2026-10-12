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
    if result.global_projection is None:
        raise ValueError("the analysis has no global projection")
    coordinates = result.global_projection.coordinates
    ax = ax or _new_axes(coordinates.shape[1])
    _validate_axes(ax, coordinates.shape[1])
    if coordinates.shape[1] == 3:
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            c=result.assignments.cluster_ids,
            cmap="gist_ncar",
            alpha=0.5,
            s=3,
        )
        ax.set_zlabel("PC3")
    else:
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=result.assignments.cluster_ids,
            cmap="gist_ncar",
            alpha=0.6,
            s=5,
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
    coordinates = projection.coordinates
    ax = ax or _new_axes(coordinates.shape[1])
    _validate_axes(ax, coordinates.shape[1])
    if coordinates.shape[1] == 3:
        ax.scatter(
            coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], alpha=0.6, s=5
        )
        ax.set_zlabel("Local PC3")
    else:
        ax.scatter(coordinates[:, 0], coordinates[:, 1], alpha=0.75, s=5)
    _highlight(ax, projection, highlight)
    ax.set(xlabel="Local PC1", ylabel="Local PC2", title=f"Cluster {cluster_id}")
    return ax


def _highlight(ax: Axes, projection: Projection, highlight: Sequence[str]) -> None:
    if not highlight or projection.sample_ids is None:
        return
    wanted = set(highlight)
    for sample_id, point in zip(projection.sample_ids, projection.coordinates, strict=True):
        if sample_id in wanted:
            if len(point) == 3:
                ax.scatter(point[0], point[1], point[2], marker="*", s=140, color="black")
                ax.text(point[0], point[1], point[2], sample_id)
            else:
                ax.scatter(point[0], point[1], marker="*", s=140, color="black")
                ax.annotate(sample_id, point)


def _new_axes(dimensions: int) -> Axes:
    if dimensions == 3:
        return plt.figure().add_subplot(111, projection="3d")
    if dimensions == 2:
        return plt.subplots()[1]
    raise ValueError("plotting requires a two- or three-component projection")


def _validate_axes(ax: Axes, dimensions: int) -> None:
    if dimensions not in (2, 3):
        raise ValueError("plotting requires a two- or three-component projection")
    if dimensions == 3 and not hasattr(ax, "zaxis"):
        raise ValueError("a three-component projection requires 3D Matplotlib axes")
