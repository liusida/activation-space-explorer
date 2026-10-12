"""End-to-end activation-space analysis."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from numpy.lib.npyio import NpzFile

from core.activations import ActivationSet, FloatArray
from core.clustering import ClusterAssignments, KMeansClusterer
from core.projection import PCAProjector, Projection
from core.version import __version__, repository_state


@dataclass(frozen=True)
class AnalysisResult:
    assignments: ClusterAssignments
    global_projection: Projection
    local_projections: Mapping[int, Projection]


@dataclass(frozen=True)
class PointAnalysis:
    cluster_id: int
    centroid_distances: FloatArray
    global_coordinates: FloatArray
    local_coordinates: FloatArray | None


class ActivationSpaceExplorer:
    """Fit and apply a k-means model with global and cluster-local PCA views."""

    artifact_version = 1

    def __init__(
        self,
        clusterer: KMeansClusterer,
        global_projector: PCAProjector | None = None,
        *,
        local_components: int = 2,
    ) -> None:
        self.clusterer = clusterer
        self.global_projector = global_projector or PCAProjector()
        self.local_components = local_components
        self.local_projectors: dict[int, PCAProjector] = {}
        self.fit_warnings: tuple[str, ...] = ()

    def fit(self, reference: ActivationSet) -> ActivationSpaceExplorer:
        assignments = self.clusterer.fit_assign(reference)
        self.global_projector.fit(reference)
        self.local_projectors = {}
        warnings: list[str] = []
        for cluster_id in range(self.clusterer.n_clusters):
            indices = np.flatnonzero(assignments.cluster_ids == cluster_id)
            max_components = min(len(indices), reference.values.shape[1])
            if max_components < self.local_components:
                warnings.append(
                    f"cluster {cluster_id} has {len(indices)} points; local PCA was omitted"
                )
                continue
            projector = PCAProjector(
                n_components=self.local_components,
                random_state=self.clusterer.random_state,
            ).fit(reference.select(indices))
            self.local_projectors[cluster_id] = projector
        self.fit_warnings = tuple(warnings)
        return self

    def analyze(self, activations: ActivationSet) -> AnalysisResult:
        assignments = self.clusterer.assign(activations)
        local_projections: dict[int, Projection] = {}
        for cluster_id, projector in self.local_projectors.items():
            indices = np.flatnonzero(assignments.cluster_ids == cluster_id)
            if len(indices):
                local_projections[cluster_id] = projector.transform(activations.select(indices))
        return AnalysisResult(
            assignments=assignments,
            global_projection=self.global_projector.transform(activations),
            local_projections=local_projections,
        )

    def analyze_one(self, activation: npt.ArrayLike) -> PointAnalysis:
        values = np.asarray(activation, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[0] != 1:
            raise ValueError("analyze_one expects exactly one activation")
        assignments = self.clusterer.assign(values)
        cluster_id = int(assignments.cluster_ids[0])
        projector = self.local_projectors.get(cluster_id)
        local = projector.transform(values).coordinates[0] if projector is not None else None
        return PointAnalysis(
            cluster_id=cluster_id,
            centroid_distances=assignments.distances[0],
            global_coordinates=self.global_projector.transform(values).coordinates[0],
            local_coordinates=local,
        )

    def save(self, path: str | Path) -> None:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, FloatArray] = {"centroids": self.clusterer.centroids}
        self._store_projector(arrays, "global", self.global_projector)
        for cluster_id, projector in self.local_projectors.items():
            self._store_projector(arrays, f"local_{cluster_id}", projector)
        np.savez_compressed(directory / "arrays.npz", **arrays)
        manifest = {
            "artifact_version": self.artifact_version,
            "core": {"version": __version__, **repository_state()},
            "clusterer": {
                "n_clusters": self.clusterer.n_clusters,
                "random_state": self.clusterer.random_state,
                "n_init": self.clusterer.n_init,
                "normalize": self.clusterer.normalize,
            },
            "global_projector": self._projector_config(self.global_projector),
            "local_components": self.local_components,
            "local_cluster_ids": sorted(self.local_projectors),
            "fit_warnings": list(self.fit_warnings),
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> ActivationSpaceExplorer:
        directory = Path(path)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("artifact_version") != cls.artifact_version:
            raise ValueError(f"unsupported artifact version: {manifest.get('artifact_version')}")
        clusterer = KMeansClusterer(**manifest["clusterer"])
        global_projector = PCAProjector(**manifest["global_projector"])
        explorer = cls(clusterer, global_projector, local_components=manifest["local_components"])
        with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
            clusterer._centroids = arrays["centroids"]
            cls._restore_projector(arrays, "global", global_projector)
            for cluster_id in manifest["local_cluster_ids"]:
                projector = PCAProjector(n_components=manifest["local_components"])
                cls._restore_projector(arrays, f"local_{cluster_id}", projector)
                explorer.local_projectors[int(cluster_id)] = projector
        explorer.fit_warnings = tuple(manifest["fit_warnings"])
        return explorer

    @staticmethod
    def _projector_config(projector: PCAProjector) -> dict[str, int | bool]:
        return {
            "n_components": projector.n_components,
            "whiten": projector.whiten,
            "random_state": projector.random_state,
        }

    @staticmethod
    def _store_projector(
        arrays: dict[str, FloatArray], prefix: str, projector: PCAProjector
    ) -> None:
        arrays[f"{prefix}_components"] = projector.components
        arrays[f"{prefix}_mean"] = projector.mean
        arrays[f"{prefix}_explained_variance"] = projector.explained_variance
        arrays[f"{prefix}_explained_variance_ratio"] = projector.explained_variance_ratio

    @staticmethod
    def _restore_projector(arrays: NpzFile, prefix: str, projector: PCAProjector) -> None:
        projector._components = arrays[f"{prefix}_components"]
        projector._mean = arrays[f"{prefix}_mean"]
        projector._explained_variance = arrays[f"{prefix}_explained_variance"]
        projector._explained_variance_ratio = arrays[f"{prefix}_explained_variance_ratio"]
