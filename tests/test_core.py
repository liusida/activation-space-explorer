import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from core import (
    ActivationSet,
    ActivationSource,
    ActivationSpaceExplorer,
    KMeansClusterer,
    PCAProjector,
    __version__,
    plot_global,
    require_version,
)


def activation_set() -> ActivationSet:
    values = np.asarray([[-2.0, 0.0], [-1.8, 0.2], [2.0, 0.0], [1.8, -0.2]])
    return ActivationSet(
        values=values,
        sample_ids=[f"sample-{index}" for index in range(len(values))],
        metadata=[{} for _ in values],
        source=ActivationSource("test-model", "layer", "none"),
    )


def test_activation_set_round_trip(tmp_path) -> None:
    original = activation_set()
    path = tmp_path / "activations.npz"
    original.save(path)
    loaded = ActivationSet.load(path)
    np.testing.assert_allclose(loaded.values, original.values)
    assert loaded.sample_ids == original.sample_ids
    assert loaded.source == original.source


def test_explorer_analyzes_reference() -> None:
    activations = activation_set()
    explorer = ActivationSpaceExplorer(
        KMeansClusterer(2, random_state=0),
        PCAProjector(2),
    ).fit(activations)
    result = explorer.analyze(activations)
    assert result.global_projection.coordinates.shape == (4, 2)
    assert len(np.unique(result.assignments.cluster_ids)) == 2
    assert set(result.local_projections) == {0, 1}


def test_explorer_artifact_round_trip(tmp_path) -> None:
    activations = activation_set()
    explorer = ActivationSpaceExplorer(KMeansClusterer(2), PCAProjector(2)).fit(activations)
    before = explorer.analyze_one(np.asarray([-2.1, 0.1]))
    artifact = tmp_path / "explorer"
    explorer.save(artifact)
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert manifest["core"]["version"] == __version__
    assert "git_commit" in manifest["core"]
    assert "git_dirty" in manifest["core"]
    after = ActivationSpaceExplorer.load(artifact).analyze_one(
        np.asarray([-2.1, 0.1])
    )
    assert after.cluster_id == before.cluster_id
    np.testing.assert_allclose(after.global_coordinates, before.global_coordinates)
    np.testing.assert_allclose(after.local_coordinates, before.local_coordinates)


def test_core_version_requirement() -> None:
    require_version("==0.1.0")
    with pytest.raises(RuntimeError, match="requires activation-space-explorer"):
        require_version(">=99")


def test_large_data_modes_and_3d_plotting() -> None:
    rng = np.random.default_rng(3)
    values = np.vstack(
        [rng.normal(-2, 0.2, size=(20, 5)), rng.normal(2, 0.2, size=(20, 5))]
    ).astype(np.float32)
    activations = ActivationSet(
        values=values,
        sample_ids=None,
        metadata=None,
        source=ActivationSource("test-model", "layer", "all_tokens"),
    )
    explorer = ActivationSpaceExplorer(
        KMeansClusterer(2, minibatch=True, batch_size=8, n_init=1),
        PCAProjector(3, incremental=True, batch_size=8),
        local_components=3,
    ).fit(activations)
    result = explorer.analyze(activations, return_distances=False)
    assert result.assignments.distances is None
    assert result.assignments.assigned_distances.shape == (40,)
    assert result.global_projection.coordinates.shape == (40, 3)
    axes = plot_global(result)
    assert hasattr(axes, "zaxis")
    plt.close(axes.figure)


def test_explorer_can_omit_global_pca(tmp_path) -> None:
    activations = activation_set()
    explorer = ActivationSpaceExplorer(KMeansClusterer(2), None).fit(activations)
    result = explorer.analyze(activations, return_distances=False)
    assert result.global_projection is None
    assert explorer.analyze_one(activations.values[0]).global_coordinates is None
    explorer.save(tmp_path / "local-only")
    loaded = ActivationSpaceExplorer.load(tmp_path / "local-only")
    assert loaded.global_projector is None
