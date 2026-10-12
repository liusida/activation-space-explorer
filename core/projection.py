"""PCA projection for global and cluster-local views."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA

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
    ) -> None:
        if n_components < 1:
            raise ValueError("n_components must be positive")
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        self._components: FloatArray | None = None
        self._mean: FloatArray | None = None
        self._explained_variance: FloatArray | None = None
        self._explained_variance_ratio: FloatArray | None = None

    def fit(self, activations: ActivationSet | npt.ArrayLike) -> PCAProjector:
        values = as_activation_values(activations)
        model = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            random_state=self.random_state,
        ).fit(values)
        self._components = np.asarray(model.components_, dtype=np.float64)
        self._mean = np.asarray(model.mean_, dtype=np.float64)
        self._explained_variance = np.asarray(model.explained_variance_, dtype=np.float64)
        self._explained_variance_ratio = np.asarray(
            model.explained_variance_ratio_, dtype=np.float64
        )
        return self

    def transform(self, activations: ActivationSet | npt.ArrayLike) -> Projection:
        values = as_activation_values(activations)
        components, mean = self._fitted_parameters()
        if values.shape[1] != components.shape[1]:
            raise ValueError(f"expected {components.shape[1]} dimensions, got {values.shape[1]}")
        coordinates = (values - mean) @ components.T
        if self.whiten:
            coordinates = coordinates / np.sqrt(
                np.maximum(self.explained_variance, np.finfo(float).eps)
            )
        sample_ids = activations.sample_ids if isinstance(activations, ActivationSet) else None
        return Projection(coordinates, self.explained_variance_ratio, sample_ids)

    def fit_transform(self, activations: ActivationSet | npt.ArrayLike) -> Projection:
        return self.fit(activations).transform(activations)

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
