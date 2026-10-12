"""Experiment lifecycle and resume orchestration."""

from __future__ import annotations

import time
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from core.runs.display import RunDisplay
from core.runs.logging import OutputCapture
from core.runs.model import ExperimentPlan, RunContext
from core.runs.state import config_hash, load_state, save_state
from core.version import __version__, repository_state


def run_experiment(plan: ExperimentPlan) -> None:
    output = Path(plan.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "run.json"
    expected_hash = config_hash(plan.config)
    state = load_state(state_path)
    fresh_state = state is None
    if state is not None and state.get("config_hash") != expected_hash:
        raise RuntimeError(
            f"{state_path} belongs to a different configuration; use a new output directory"
        )

    source = repository_state()
    run_id = uuid.uuid4().hex[:8]
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / f"{time.strftime('%Y%m%d-%H%M%S')}-{run_id}.log"
    if state is None:
        state = {
            "schema_version": 1,
            "experiment": plan.name,
            "config_hash": expected_hash,
            "core_version": __version__,
            **source,
            "stages": {},
        }
        save_state(state_path, state)

    warning = (
        "WARNING: source contains uncommitted changes; the commit alone cannot reproduce this run."
        if source["git_dirty"]
        else None
    )
    display = RunDisplay(
        plan.name,
        plan.description,
        len(plan.stages),
        log_path,
        state_path,
        warning,
    )
    context = RunContext(output, log_path, plan.config, display, state)
    changed_stages: set[str] = set()
    with display:
        context.log(f"started run {run_id}")
        try:
            for stage in plan.stages:
                context._stage = stage
                display.begin_stage(stage.title)
                previous = state["stages"].get(stage.id, {})
                reusable = stage.durable and previous.get("status") == "complete"
                dependencies_changed = bool(changed_stages.intersection(stage.dependencies))
                if reusable and (dependencies_changed or stage.validator is None):
                    reusable = not dependencies_changed and stage.validator is None
                if reusable and stage.validator is not None:
                    reusable = stage.validator(context)
                if (
                    fresh_state
                    and stage.durable
                    and not dependencies_changed
                    and stage.validator is not None
                ):
                    reusable = stage.validator(context)
                if reusable:
                    state["stages"][stage.id] = {
                        "status": "complete",
                        "duration_seconds": previous.get("duration_seconds", 0.0),
                        "summary": previous.get("summary", {"imported": True}),
                    }
                    save_state(state_path, state)
                    context.log(f"reused completed stage: {stage.title}")
                    display.finish_stage(reused=True)
                    continue

                started = time.monotonic()
                state["stages"][stage.id] = {"status": "running"}
                save_state(state_path, state)
                context._summary = {}
                stdout = OutputCapture(context, "info")
                stderr = OutputCapture(context, "info")
                try:
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        stage.function(context)
                finally:
                    stdout.flush()
                    stderr.flush()
                state["stages"][stage.id] = {
                    "status": "complete",
                    "duration_seconds": time.monotonic() - started,
                    "summary": context._summary,
                }
                save_state(state_path, state)
                context.log(f"completed stage: {stage.title}")
                display.finish_stage()
                changed_stages.add(stage.id)
        except KeyboardInterrupt:
            if context._stage is not None:
                state["stages"][context._stage.id] = {"status": "interrupted"}
            save_state(state_path, state)
            context.log("run interrupted", level="warning")
            display.status = "interrupted"
            raise
        except Exception as error:
            if context._stage is not None:
                state["stages"][context._stage.id] = {
                    "status": "failed",
                    "error": repr(error),
                }
            save_state(state_path, state)
            context.log(f"run failed: {error}", level="error")
            display.status = "failed"
            raise
        context.log("run complete")
        display.status = "complete"
