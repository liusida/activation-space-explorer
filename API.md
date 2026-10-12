# Core API Design

Status: proposed v0.1 API. This document defines the intended interface before implementation.

## Concepts

- **Activation point**: one fixed-length vector from a model location, such as a layer and token.
- **Activation set**: activation points plus sample IDs and metadata.
- **Cluster assignment**: the nearest learned cluster for a point. A cluster is not automatically a semantic feature or category.
- **Global view**: one PCA basis fitted across all points, so clusters share comparable coordinates.
- **Local view**: a PCA basis fitted within one cluster, revealing its internal geometry. Coordinates from different local views are not comparable.

The core operates on NumPy arrays and does not require a particular model framework. Framework-specific extraction can be supplied through adapters.

## Package layout

```text
core/
├── activations.py   # ActivationSet and validation
├── extraction.py    # Extractor protocol and extraction specifications
├── clustering.py    # KMeansClusterer and assignments
├── projection.py    # PCAProjector and projected views
├── explorer.py      # End-to-end orchestration
├── plotting.py      # Matplotlib helpers
└── artifacts.py     # Save/load fitted research artifacts
```

The public objects below should also be importable directly from `core`.

## Activation data

```python
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]


@dataclass(frozen=True)
class ActivationSet:
    values: FloatArray                    # shape: (n_points, n_dimensions)
    sample_ids: Sequence[str] | None       # optional; length: n_points
    metadata: Sequence[Mapping[str, Any]] | None
    source: "ActivationSource"

    def select(self, indices: npt.ArrayLike) -> "ActivationSet": ...
    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> "ActivationSet": ...


@dataclass(frozen=True)
class ActivationSource:
    model_id: str
    location: str       # e.g. "model.layers.12"
    reduction: str      # e.g. "last_token", "mean_tokens", or "none"
    model_revision: str | None = None
```

`values` must be finite, two-dimensional, and use rows as independently clustered points. Float32 arrays and memory maps are retained without conversion to float64. If token-level activations are retained, each token is a point. Row metadata may be omitted for very large captures and stored separately in a compact representation.

## Extraction

```python
from typing import Iterable, Protocol


@dataclass(frozen=True)
class ExtractionSpec:
    location: str
    reduction: str = "last_token"
    batch_size: int = 8
    dtype: str = "float32"


class ActivationExtractor(Protocol):
    def extract(
        self,
        inputs: Iterable[Any],
        spec: ExtractionSpec,
    ) -> ActivationSet: ...
```

Adapters implement this protocol for PyTorch, Transformers, vision models, or experiment-specific models. Extraction and analysis remain separate so saved activations can be reused without loading the original model.

## K-means clustering

```python
@dataclass(frozen=True)
class ClusterAssignments:
    cluster_ids: npt.NDArray[np.integer]  # shape: (n_points,)
    distances: FloatArray | None          # optional shape: (n_points, n_clusters)

    @property
    def assigned_distances(self) -> FloatArray: ...


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
    ) -> None: ...

    def fit(self, activations: ActivationSet | FloatArray) -> "KMeansClusterer": ...
    def assign(
        self,
        activations: ActivationSet | FloatArray,
        *,
        return_distances: bool = True,
    ) -> ClusterAssignments: ...
    def fit_assign(
        self,
        activations: ActivationSet | FloatArray,
    ) -> ClusterAssignments: ...

    @property
    def centroids(self) -> FloatArray: ...
```

By default, `assign` returns every centroid distance, not only the winning cluster. This supports ambiguity measures, outlier detection, and comparisons between alternative interpretations. At large scale, `return_distances=False` retains only the assigned distance and avoids materializing an `n_points × n_clusters` matrix. Cluster IDs are arbitrary within a fit and must not be treated as stable labels across independently fitted models.

## PCA projection

```python
@dataclass(frozen=True)
class Projection:
    coordinates: FloatArray               # shape: (n_points, n_components)
    explained_variance_ratio: FloatArray  # shape: (n_components,)
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
    ) -> None: ...

    def fit(self, activations: ActivationSet | FloatArray) -> "PCAProjector": ...
    def transform(self, activations: ActivationSet | FloatArray) -> Projection: ...
    def fit_transform(self, activations: ActivationSet | FloatArray) -> Projection: ...
```

A global projector is optional and, when supplied, is fitted on the full reference set. A local projector is fitted only on points assigned to one cluster. Set `incremental=True` for activation sets too large for in-memory PCA. Two- and three-component projections are supported by the Matplotlib plotting API. New points must use the already-fitted projector to remain comparable to the reference visualization.

## Explorer

`ActivationSpaceExplorer` is the convenient end-to-end interface. The lower-level objects remain public for experiments that need more control.

```python
@dataclass(frozen=True)
class AnalysisResult:
    assignments: ClusterAssignments
    global_projection: Projection | None
    local_projections: Mapping[int, Projection]


class ActivationSpaceExplorer:
    def __init__(
        self,
        clusterer: KMeansClusterer,
        global_projector: PCAProjector | None = None,
        *,
        local_components: int = 2,
    ) -> None: ...

    def fit(
        self,
        reference: ActivationSet,
        *,
        progress: bool = False,
    ) -> "ActivationSpaceExplorer": ...
    def analyze(
        self,
        activations: ActivationSet,
        *,
        return_distances: bool = True,
    ) -> AnalysisResult: ...
    def analyze_one(self, activation: npt.ArrayLike) -> "PointAnalysis": ...
    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> "ActivationSpaceExplorer": ...


@dataclass(frozen=True)
class PointAnalysis:
    cluster_id: int
    centroid_distances: FloatArray
    global_coordinates: FloatArray
    local_coordinates: FloatArray | None
```

During `fit`, the explorer fits k-means, one global PCA projector, and one local PCA projector per sufficiently large cluster. `analyze` never refits these objects. If a cluster has too few points for the requested PCA dimension, its local projection is omitted with a warning recorded in artifact metadata.

## Plotting

Plotting functions return Matplotlib objects and never call `show()` or write files implicitly.

```python
def plot_global(
    result: AnalysisResult,
    *,
    ax: "Axes | None" = None,
    highlight: Sequence[str] = (),
) -> "Axes": ...


def plot_cluster(
    result: AnalysisResult,
    cluster_id: int,
    *,
    ax: "Axes | None" = None,
    highlight: Sequence[str] = (),
) -> "Axes": ...
```

## Typical workflow

```python
from core import ActivationSpaceExplorer, KMeansClusterer, PCAProjector

reference = extractor.extract(reference_inputs, spec)

explorer = ActivationSpaceExplorer(
    clusterer=KMeansClusterer(n_clusters=32, random_state=7),
    global_projector=PCAProjector(n_components=2),
)
explorer.fit(reference)
explorer.save("artifacts/layer_12_k32")

new_activations = extractor.extract(new_inputs, spec)
result = explorer.analyze(new_activations)
```

## Artifact and reproducibility contract

A saved explorer should contain:

- cluster centroids and all preprocessing parameters;
- global and per-cluster PCA parameters;
- model/source metadata and activation dimensionality;
- library, NumPy, and scikit-learn versions;
- random seeds and fit-time warnings;
- a manifest with a format version.

Experiments declare their required core version in configuration and validate it before running:

```toml
[experiment]
core_version = "==0.1.0"
```

```python
from core import require_version

require_version(config["experiment"]["core_version"])
```

Saved manifests record `core.version`, `core.git_commit`, and `core.git_dirty`. A dirty repository is reproducible only after its changes are committed or separately archived.

Loading must validate the artifact format and activation dimensionality. Artifacts should use explicit arrays plus JSON metadata rather than pickle as the long-term interchange format.

## Deferred extensions

The v0.1 API intentionally leaves these for later research:

- streaming or mini-batch k-means;
- soft, hierarchical, or density-based clustering;
- cluster matching across layers, models, or independent fits;
- automatic semantic naming of clusters;
- nonlinear projections such as UMAP or t-SNE;
- causal interventions based on discovered directions.
