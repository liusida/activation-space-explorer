import json

import pytest

from core.runs import ExperimentPlan, RunContext, Stage, run_experiment


def test_runner_records_and_reuses_durable_stages(tmp_path) -> None:
    calls = {"durable": 0, "render": 0}

    def durable(ctx: RunContext) -> None:
        calls["durable"] += 1
        with ctx.progress(2, "items") as progress:
            progress.substage("first")
            progress.advance()
            print("captured output")
            progress.advance()
        ctx.complete({"items": 2})

    def render(_ctx: RunContext) -> None:
        calls["render"] += 1

    plan = ExperimentPlan(
        name="test/example",
        output_dir=tmp_path,
        config={"value": 1},
        stages=[
            Stage("durable", "Durable stage", durable),
            Stage("render", "Render", render, durable=False),
        ],
    )
    run_experiment(plan)
    run_experiment(plan)
    assert calls == {"durable": 1, "render": 2}
    state = json.loads((tmp_path / "run.json").read_text())
    assert state["stages"]["durable"]["summary"] == {"items": 2}
    logs = "\n".join(path.read_text() for path in (tmp_path / "logs").iterdir())
    assert "captured output" in logs


def test_runner_rejects_configuration_mismatch(tmp_path) -> None:
    run_experiment(ExperimentPlan("test", tmp_path, {"value": 1}, []))
    with pytest.raises(RuntimeError, match="different configuration"):
        run_experiment(ExperimentPlan("test", tmp_path, {"value": 2}, []))


def test_runner_imports_valid_artifacts_but_invalidates_downstream(tmp_path) -> None:
    calls = {"capture": 0, "cluster": 0}
    (tmp_path / "capture.done").write_text("complete")

    def capture(_ctx: RunContext) -> None:
        calls["capture"] += 1

    def cluster(_ctx: RunContext) -> None:
        calls["cluster"] += 1

    plan = ExperimentPlan(
        "test/import",
        tmp_path,
        {"value": 1},
        [
            Stage(
                "capture",
                "Capture",
                capture,
                validator=lambda ctx: (ctx.output_dir / "capture.done").exists(),
            ),
            Stage(
                "cluster",
                "Cluster",
                cluster,
                dependencies=("capture",),
                validator=lambda _ctx: False,
            ),
        ],
    )
    run_experiment(plan)
    assert calls == {"capture": 0, "cluster": 1}
