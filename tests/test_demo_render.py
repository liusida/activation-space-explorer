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
    assert "Click a point to inspect its token context" in html
    assert '"before":"before "' in html
    assert '"token":"target"' in html
    assert '"keyword_match":true' in html
    assert '"keyword":"tar"' in html
    assert "model/test" in html
    assert "cluster 23" not in html  # Built safely from embedded data at runtime.
