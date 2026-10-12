"""Line-buffered stdout/stderr capture for the live runner."""

from __future__ import annotations

import io
from typing import Any


class OutputCapture(io.TextIOBase):
    def __init__(self, context: Any, level: str) -> None:
        self.context = context
        self.level = level
        self.buffer = ""

    def write(self, value: str) -> int:
        normalized = value.replace("\r", "\n")
        self.buffer += normalized
        parts = self.buffer.split("\n")
        self.buffer = parts.pop()
        for line in parts:
            if line.strip():
                self.context.log(line.rstrip(), level=self.level)
        return len(value)

    def flush(self) -> None:
        if self.buffer.strip():
            self.context.log(self.buffer.rstrip(), level=self.level)
        self.buffer = ""
