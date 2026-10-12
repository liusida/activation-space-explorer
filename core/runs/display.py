"""Rich live display for experiment runs."""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Self

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text


class RunDisplay:
    def __init__(
        self,
        name: str,
        description: str,
        total_stages: int,
        log_path: Path,
        state_path: Path,
        source_warning: str | None,
    ) -> None:
        self.name = name
        self.description = description
        self.total_stages = total_stages
        self.log_path = log_path
        self.state_path = state_path
        self.source_warning = source_warning
        self.completed_stages = 0
        self.stage_title = "Preparing"
        self.substage = "—"
        self.status = "starting"
        self.recent: deque[str] = deque(maxlen=8)
        self.overall_progress = Progress(
            TextColumn("{task.completed:,.0f}/{task.total:,.0f} stages"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        self.overall_task = self.overall_progress.add_task("overall", total=total_stages)
        self.stage_progress = Progress(
            TextColumn("{task.completed:,.0f}/{task.total:,.0f} {task.fields[unit]}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        self.stage_task = self.stage_progress.add_task("stage", total=1, unit="unit")
        # Bind once before stage stdout/stderr redirection. Without this, Rich
        # resolves the redirected stream during refresh and captures its own UI.
        self.console = Console(file=sys.stderr)
        self.live = Live(
            self.render(),
            console=self.console,
            refresh_per_second=6,
            transient=False,
        )

    def __enter__(self) -> Self:
        self.live.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self.live.update(self.render(), refresh=True)
        self.live.__exit__(*args)

    def begin_stage(self, title: str) -> None:
        self.stage_title = title
        self.substage = "—"
        self.status = "running"
        self.update_stage_progress(0, 1, "unit")

    def finish_stage(self, reused: bool = False) -> None:
        self.completed_stages += 1
        self.overall_progress.update(self.overall_task, completed=self.completed_stages)
        self.status = "reused" if reused else "complete"
        self.live.update(self.render(), refresh=True)

    def update_stage_progress(self, completed: int, total: int, unit: str) -> None:
        self.stage_progress.update(
            self.stage_task,
            completed=completed,
            total=max(total, 1),
            unit=unit,
        )
        self.live.update(self.render(), refresh=True)

    def set_substage(self, label: str) -> None:
        self.substage = label
        self.live.update(self.render(), refresh=True)

    def add_output(self, line: str) -> None:
        self.recent.append(line.rstrip())
        self.live.update(self.render(), refresh=True)

    def render(self) -> Group:
        header = Panel(
            self.description or self.name,
            title=f"Activation Space Explorer · {self.name}",
            border_style="blue",
        )
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", width=11)
        table.add_column()
        table.add_row("Overall", self.overall_progress)
        table.add_row("Stage", self.stage_title)
        table.add_row("Progress", self.stage_progress)
        table.add_row("Substage", self.substage)
        table.add_row("Status", self.status)
        output = "\n".join(self.recent) if self.recent else "Waiting for output…"
        parts: list[object] = [header]
        if self.source_warning:
            parts.append(Panel(Text(self.source_warning, style="bold yellow"), title="Source warning"))
        parts.extend(
            [
                Panel(table),
                Panel(output, title="Recent output", height=11),
                Text(f"Full log  {self.log_path}\nState     {self.state_path}", style="dim"),
            ]
        )
        return Group(*parts)
