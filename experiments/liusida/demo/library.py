"""Shared capture/cluster library and per-run output directories."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from core.runs.state import config_hash

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "output"
LIBRARY_ROOT = OUTPUT_ROOT / "library"
RUNS_ROOT = OUTPUT_ROOT / "runs"

SHARED_NAMES = (
    "activations.f32",
    "token_ids.i32",
    "capture.json",
    "clusters.npz",
)


def base_config(config: dict[str, Any]) -> dict[str, Any]:
    """Model + capture + analysis only (shared across targets)."""
    return {
        "model": config["model"],
        "capture": config["capture"],
        "analysis": config["analysis"],
    }


def base_config_hash(config: dict[str, Any]) -> str:
    return config_hash(base_config(config)).removeprefix("sha256:")[:12]


def library_dir_for(config: dict[str, Any]) -> Path:
    return LIBRARY_ROOT / base_config_hash(config)


def slugify(text: str, *, fallback: str = "run") -> str:
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in text.strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part).lower()
    return (cleaned[:48] or fallback)


def default_run_dir(config: dict[str, Any]) -> Path:
    token = slugify(str(config.get("target", {}).get("token", "token")), fallback="token")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return RUNS_ROOT / f"{stamp}_{token}"


def migrate_flat_output_into_library(config: dict[str, Any]) -> Path | None:
    """If legacy output/ has shared artifacts, move/link them into library/<hash>/."""
    library = library_dir_for(config)
    flat_ready = all((OUTPUT_ROOT / name).exists() for name in SHARED_NAMES)
    if not flat_ready:
        return library if all((library / name).exists() for name in SHARED_NAMES) else None

    library.mkdir(parents=True, exist_ok=True)
    for name in SHARED_NAMES:
        src = OUTPUT_ROOT / name
        dst = library / name
        if dst.exists() or dst.is_symlink():
            continue
        src.rename(dst)
    meta = {
        "base_config_hash": base_config_hash(config),
        "base_config": base_config(config),
    }
    (library / "library.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return library


def ensure_library_links(library: Path, run_dir: Path) -> None:
    """Symlink shared capture/cluster artifacts into a run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in SHARED_NAMES if not (library / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"shared library incomplete at {library}; missing {missing}. "
            "Run the demo once to capture/cluster, or point --library at a complete folder."
        )
    for name in SHARED_NAMES:
        src = (library / name).resolve()
        dst = run_dir / name
        if dst.exists() or dst.is_symlink():
            continue
        dst.symlink_to(src)


def find_latest_library() -> Path | None:
    if not LIBRARY_ROOT.exists():
        # Fall back to flat output/ layout.
        if all((OUTPUT_ROOT / name).exists() for name in SHARED_NAMES):
            return OUTPUT_ROOT
        return None
    candidates = [
        path
        for path in LIBRARY_ROOT.iterdir()
        if path.is_dir() and all((path / name).exists() for name in SHARED_NAMES)
    ]
    if not candidates:
        if all((OUTPUT_ROOT / name).exists() for name in SHARED_NAMES):
            return OUTPUT_ROOT
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def write_run_pointer(run_dir: Path, *, library: Path, config: dict[str, Any]) -> None:
    payload = {
        "library": str(library),
        "base_config_hash": base_config_hash(config),
        "full_config_hash": config_hash(config),
        "target": config.get("target"),
        "visualization": config.get("visualization"),
    }
    (run_dir / "run_pointer.json").write_text(json.dumps(payload, indent=2) + "\n")


def promote_shared_into_library(run_dir: Path, library: Path, config: dict[str, Any]) -> None:
    """Copy newly written shared artifacts from a run into the library (no-op if linked)."""
    library.mkdir(parents=True, exist_ok=True)
    for name in SHARED_NAMES:
        src = run_dir / name
        dst = library / name
        if not src.exists():
            continue
        if src.is_symlink():
            continue
        if dst.exists() or dst.is_symlink():
            continue
        # Hardlink when possible (same filesystem), else copy.
        try:
            dst.hardlink_to(src)
        except OSError:
            import shutil

            shutil.copy2(src, dst)
    meta = {
        "base_config_hash": base_config_hash(config),
        "base_config": base_config(config),
    }
    (library / "library.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
