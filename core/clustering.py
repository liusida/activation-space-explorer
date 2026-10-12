"""K-means clustering and assignment."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize

from core.activations import ActivationSet, FloatArray, as_activation_values


@dataclass(frozen=True)
class ClusterAssignments:
    cluster_ids: npt.NDArray[np.integer]
    distances: FloatArray | None
    nearest_distances: FloatArray

    @property
    def assigned_distances(self) -> FloatArray:
        return self.nearest_distances


class KMeansClusterer:
    def __init__(
        self,
        n_clusters: int,
        *,
        random_state: int = 0,
        n_init: int | str = "auto",
        normalize: bool = False,
        minibatch: bool = False,
        batch_size: int = 8192,
        max_iter: int = 100,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be positive")
        if minibatch and batch_size < n_clusters:
            raise ValueError("mini-batch size must be at least n_clusters")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.normalize = normalize
        self.minibatch = minibatch
        self.batch_size = batch_size
        self.max_iter = max_iter
        self._centroids: FloatArray | None = None

    def _prepare(self, activations: ActivationSet | npt.ArrayLike) -> FloatArray:
        values = as_activation_values(activations)
        return normalize(values) if self.normalize else values

    def fit(
        self,
        activations: ActivationSet | npt.ArrayLike,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> KMeansClusterer:
        values = self._prepare(activations)
        if self.minibatch:
            model = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=self.n_init,
                batch_size=self.batch_size,
            )
            batches = range(0, len(values), self.batch_size)
            steps = self.max_iter * len(batches)
            completed = 0
            for _epoch in range(self.max_iter):
                for start in batches:
                    model.partial_fit(values[start : start + self.batch_size])
                    completed += 1
                    if progress is not None:
                        progress(completed, steps)
        else:
            model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=self.n_init,
                max_iter=self.max_iter,
            ).fit(values)
        self._centroids = np.asarray(model.cluster_centers_)
        return self

    def assign(
        self,
        activations: ActivationSet | npt.ArrayLike,
        *,
        return_distances: bool = True,
        progress: Callable[[int, int], None] | None = None,
    ) -> ClusterAssignments:
        values = self._prepare(activations)
        centroids = self.centroids
        if values.shape[1] != centroids.shape[1]:
            raise ValueError(
                f"expected {centroids.shape[1]} activation dimensions, got {values.shape[1]}"
            )
        cluster_ids = np.empty(len(values), dtype=np.int32)
        nearest = np.empty(len(values), dtype=values.dtype)
        all_distances = (
            np.empty((len(values), self.n_clusters), dtype=values.dtype)
            if return_distances
            else None
        )
        batches = range(0, len(values), self.batch_size)
        for batch_index, start in enumerate(batches, start=1):
            stop = min(start + self.batch_size, len(values))
            distances = euclidean_distances(values[start:stop], centroids)
            labels = np.argmin(distances, axis=1)
            cluster_ids[start:stop] = labels
            nearest[start:stop] = distances[np.arange(stop - start), labels]
            if all_distances is not None:
                all_distances[start:stop] = distances
            if progress is not None:
                progress(batch_index, len(batches))
        return ClusterAssignments(cluster_ids, all_distances, nearest)

    def fit_assign(
        self,
        activations: ActivationSet | npt.ArrayLike,
        *,
        return_distances: bool = True,
        progress: Callable[[int, int], None] | None = None,
    ) -> ClusterAssignments:
        return self.fit(activations, progress=progress).assign(
            activations,
            return_distances=return_distances,
            progress=progress,
        )

    @property
    def centroids(self) -> FloatArray:
        if self._centroids is None:
            raise RuntimeError("clusterer has not been fitted")
        return self._centroids
