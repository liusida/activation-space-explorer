"""K-means clustering and assignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from core.activations import ActivationSet, FloatArray, as_activation_values


@dataclass(frozen=True)
class ClusterAssignments:
    cluster_ids: npt.NDArray[np.integer]
    distances: FloatArray

    @property
    def assigned_distances(self) -> FloatArray:
        return self.distances[np.arange(len(self.cluster_ids)), self.cluster_ids]


class KMeansClusterer:
    def __init__(
        self,
        n_clusters: int,
        *,
        random_state: int = 0,
        n_init: int | str = "auto",
        normalize: bool = False,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be positive")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.normalize = normalize
        self._centroids: FloatArray | None = None

    def _prepare(self, activations: ActivationSet | npt.ArrayLike) -> FloatArray:
        values = as_activation_values(activations)
        return normalize(values) if self.normalize else values

    def fit(self, activations: ActivationSet | npt.ArrayLike) -> KMeansClusterer:
        values = self._prepare(activations)
        model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        ).fit(values)
        self._centroids = np.asarray(model.cluster_centers_, dtype=np.float64)
        return self

    def assign(self, activations: ActivationSet | npt.ArrayLike) -> ClusterAssignments:
        values = self._prepare(activations)
        centroids = self.centroids
        if values.shape[1] != centroids.shape[1]:
            raise ValueError(
                f"expected {centroids.shape[1]} activation dimensions, got {values.shape[1]}"
            )
        distances = np.linalg.norm(values[:, None, :] - centroids[None, :, :], axis=2)
        return ClusterAssignments(cluster_ids=np.argmin(distances, axis=1), distances=distances)

    def fit_assign(self, activations: ActivationSet | npt.ArrayLike) -> ClusterAssignments:
        return self.fit(activations).assign(activations)

    @property
    def centroids(self) -> FloatArray:
        if self._centroids is None:
            raise RuntimeError("clusterer has not been fitted")
        return self._centroids
