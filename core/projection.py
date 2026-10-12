"""PCA projection for global and cluster-local views."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA, IncrementalPCA

from core.activations import ActivationSet, FloatArray, as_activation_values


@dataclass(frozen=True)
class Projection:
    coordinates: FloatArray
    explained_variance_ratio: FloatArray
    sample_ids: Sequence[str] | None = None


class PCAProjector:
    def __init__(
        self,
        n_components: int = 2,
        *,
        whiten: bool = False,
        random_state: int = 0,
        incremental: bool = False,
        batch_size: int = 8192,
    ) -> None:
        if n_components < 1:
            raise ValueError("n_components must be positive")
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        self.incremental = incremental
        self.batch_size = batch_size
        self._components: FloatArray | None = None
        self._mean: FloatArray | None = None
        self._explained_variance: FloatArray | None = None
        self._explained_variance_ratio: FloatArray | None = None

    def fit(
        self,
        activations: ActivationSet | npt.ArrayLike,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> PCAProjector:
        values = as_activation_values(activations)
        if self.incremental:
            model = IncrementalPCA(
                n_components=self.n_components,
                whiten=self.whiten,
                batch_size=self.batch_size,
            )
            starts = list(range(0, len(values), self.batch_size))
            if len(starts) > 1 and len(values) - starts[-1] < self.n_components:
                starts.pop()
            for index, start in enumerate(starts):
                stop = starts[index + 1] if index + 1 < len(starts) else len(values)
                model.partial_fit(values[start:stop])
                if progress is not None:
                    progress(index + 1, len(starts))
        else:
            model = PCA(
                n_components=self.n_components,
                whiten=self.whiten,
                random_state=self.random_state,
            ).fit(values)
        self._components = np.asarray(model.components_)
        self._mean = np.asarray(model.mean_)
        self._explained_variance = np.asarray(model.explained_variance_)
        self._explained_variance_ratio = np.asarray(
            model.explained_variance_ratio_
        )
        return self

    def transform(self, activations: ActivationSet | npt.ArrayLike) -> Projection:
        values = as_activation_values(activations)
        components, mean = self._fitted_parameters()
        if values.shape[1] != components.shape[1]:
            raise ValueError(f"expected {components.shape[1]} dimensions, got {values.shape[1]}")
        coordinates = np.empty((len(values), self.n_components), dtype=values.dtype)
        for start in range(0, len(values), self.batch_size):
            stop = min(start + self.batch_size, len(values))
            batch = (values[start:stop] - mean) @ components.T
            if self.whiten:
                batch = batch / np.sqrt(
                    np.maximum(self.explained_variance, np.finfo(float).eps)
                )
            coordinates[start:stop] = batch
        sample_ids = activations.sample_ids if isinstance(activations, ActivationSet) else None
        return Projection(coordinates, self.explained_variance_ratio, sample_ids)

    def fit_transform(
        self,
        activations: ActivationSet | npt.ArrayLike,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> Projection:
        return self.fit(activations, progress=progress).transform(activations)

    def _fitted_parameters(self) -> tuple[FloatArray, FloatArray]:
        if self._components is None or self._mean is None:
            raise RuntimeError("projector has not been fitted")
        return self._components, self._mean

    @property
    def components(self) -> FloatArray:
        return self._fitted_parameters()[0]

    @property
    def mean(self) -> FloatArray:
        return self._fitted_parameters()[1]

    @property
    def explained_variance(self) -> FloatArray:
        if self._explained_variance is None:
            raise RuntimeError("projector has not been fitted")
        return self._explained_variance

    @property
    def explained_variance_ratio(self) -> FloatArray:
        if self._explained_variance_ratio is None:
            raise RuntimeError("projector has not been fitted")
        return self._explained_variance_ratio
