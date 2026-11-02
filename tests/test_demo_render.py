import importlib.util
from pathlib import Path

import numpy as np

RENDER_PATH = Path(__file__).parents[1] / "experiments" / "liusida" / "demo" / "render.py"
SPEC = importlib.util.spec_from_file_location("demo_render", RENDER_PATH)
assert SPEC is not None and SPEC.loader is not None
RENDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDER)


def test_cluster_pca_html_contains_target_and_context(tmp_path) -> None:
    path = tmp_path / "view.html"
    RENDER.write_cluster_pca_html(
        path,
        coordinates=np.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]),
        explained_variance_ratio=np.asarray([0.5, 0.3, 0.1]),
        details=[
            {
                "row": 4,
                "token_id": 10,
                "sequence": 0,
                "position": 4,
                "before": "before ",
                "token": "target",
                "after": " after",
            },
            {
                "row": 8,
                "token_id": 11,
                "sequence": 0,
                "position": 8,
                "before": "another ",
                "token": "point",
                "after": " context",
            },
        ],
        target_position=0,
        model_id="model/test",
        layer_index=16,
        cluster_id=23,
        keyword="  TAR  ",
    )

    html = path.read_text()
    assert "Target activation" in html
    assert "Graded cluster decomposition" in html
    assert '"before":"before "' in html
    assert '"token":"target"' in html
    assert '"keyword_match":true' in html
    assert '"keyword":"tar"' in html
    assert "model/test" in html
    assert '"cluster_id":23' in html


def test_decomposition_html_colors_multiple_clusters(tmp_path) -> None:
    path = tmp_path / "decomp.html"
    RENDER.write_activation_decomposition_html(
        path,
        coordinates=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.5]],
            dtype=np.float64,
        ),
        explained_variance_ratio=np.asarray([0.4, 0.3, 0.2]),
        details=[
            {
                "row": 1,
                "token_id": 1,
                "sequence": 0,
                "position": 1,
                "before": "a ",
                "token": " Paris",
                "after": " .",
            },
            {
                "row": 2,
                "token_id": 2,
                "sequence": 0,
                "position": 2,
                "before": "b ",
                "token": " London",
                "after": " .",
            },
            {
                "row": 3,
                "token_id": 3,
                "sequence": 0,
                "position": 3,
                "before": "c ",
                "token": " not",
                "after": " .",
            },
            {
                "row": "external",
                "token_id": 9,
                "sequence": "prompt",
                "position": 5,
                "before": "is",
                "token": " Paris",
                "after": ".",
            },
        ],
        cluster_ids=[10, 10, 20, 10],
        target_position=3,
        decomposition=[
            {
                "cluster_id": 10,
                "weight": 0.7,
                "distance": 1.2,
                "rank": 1,
                "n_shown": 2,
                "n_total": 50,
                "sample_tokens": [" Paris", " London"],
            },
            {
                "cluster_id": 20,
                "weight": 0.3,
                "distance": 3.4,
                "rank": 2,
                "n_shown": 1,
                "n_total": 40,
                "sample_tokens": [" not"],
            },
        ],
        model_id="model/test",
        layer_index=16,
        keyword="paris",
        temperature=2.5,
    )
    html = path.read_text()
    assert "soft decomposition" in html
    assert '"weight_pct":70.0' in html
    assert '"cluster_id":10' in html
    assert '"cluster_id":20' in html
    assert "softmax" in html
