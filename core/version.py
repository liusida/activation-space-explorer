"""Core version checks and repository provenance."""

from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TypedDict

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version

try:
    __version__ = version("activation-space-explorer")
except PackageNotFoundError:
    __version__ = "0.1.0"


class RepositoryState(TypedDict):
    git_commit: str | None
    git_dirty: bool | None


def require_version(specifier: str) -> None:
    """Raise when the installed core version does not satisfy an experiment requirement."""
    try:
        allowed = SpecifierSet(specifier)
    except InvalidSpecifier as error:
        raise ValueError(f"invalid core version requirement: {specifier!r}") from error
    if Version(__version__) not in allowed:
        raise RuntimeError(
            f"experiment requires activation-space-explorer {specifier}, "
            f"but {__version__} is installed"
        )


def repository_state() -> RepositoryState:
    """Return the current Git commit and whether tracked or untracked files differ."""
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(status)}
