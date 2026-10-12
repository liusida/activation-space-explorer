"""Public experiment-runner models."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

StageFunction = Callable[["RunContext"], None]
StageValidator = Callable[["RunContext"], bool]


@dataclass(frozen=True)
class Stage:
    id: str
    title: str
    function: StageFunction
    durable: bool = True
    dependencies: tuple[str, ...] = ()
    validator: StageValidator | None = None


@dataclass(frozen=True)
class ExperimentPlan:
    name: str
    output_dir: Path
    config: Mapping[str, Any]
    stages: Sequence[Stage]
    description: str = ""


@dataclass
class ProgressHandle:
    context: RunContext
    total: int
    unit: str
    completed: int = 0

    def advance(self, amount: int = 1) -> None:
        self.completed = min(self.total, self.completed + amount)
        self.context._update_progress(self.completed, self.total, self.unit)

    def update(self, completed: int) -> None:
        self.completed = min(self.total, completed)
        self.context._update_progress(self.completed, self.total, self.unit)

    def substage(self, label: str) -> None:
        self.context._set_substage(label)

    def callback(self, completed: int, total: int) -> None:
        if total != self.total:
            self.total = total
        self.update(completed)


@dataclass
class RunContext:
    output_dir: Path
    log_path: Path
    config: Mapping[str, Any]
    _display: Any
    _state: dict[str, Any]
    _stage: Stage | None = None
    _summary: dict[str, Any] = field(default_factory=dict)

    def log(self, message: str, level: str = "info") -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        line = f"[{timestamp}] {level.upper():7s} {message}"
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
        self._display.add_output(message)

    @contextmanager
    def progress(self, total: int, unit: str) -> Iterator[ProgressHandle]:
        handle = ProgressHandle(self, total=total, unit=unit)
        self._update_progress(0, total, unit)
        yield handle
        handle.update(total)

    def complete(self, summary: Mapping[str, Any] | None = None) -> None:
        self._summary = dict(summary or {})

    def save_json_atomic(self, path: str | Path, value: Any) -> Path:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(target)
        return target

    def save_npz_atomic(self, path: str | Path, **arrays: np.ndarray) -> Path:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(target)
        return target

    def _resolve(self, path: str | Path) -> Path:
        value = Path(path)
        return value if value.is_absolute() else self.output_dir / value

    def _update_progress(self, completed: int, total: int, unit: str) -> None:
        self._display.update_stage_progress(completed, total, unit)

    def _set_substage(self, label: str) -> None:
        self._display.set_substage(label)
