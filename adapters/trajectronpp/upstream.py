"""Helpers for locating the local Trajectron++ checkout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]


def resolve_upstream(upstream_dir: str | Path | None = None) -> Path:
    candidates = []
    if upstream_dir:
        candidates.append(Path(upstream_dir).expanduser())
    candidates.extend([
        EXPERIMENT_ROOT / "external" / "trajectronpp",
        EXPERIMENT_ROOT.parent / "trajectronPP",
        EXPERIMENT_ROOT.parent / "Trajectron-plus-plus",
    ])
    for path in candidates:
        path = path if path.is_absolute() else (EXPERIMENT_ROOT / path).resolve()
        if (path / "trajectron").exists():
            return path
    raise FileNotFoundError(
        "Trajectron++ checkout not found. Expected external/trajectronpp, "
        "../trajectronPP, or ../Trajectron-plus-plus."
    )


def add_upstream_to_path(upstream_dir: str | Path | None = None) -> Path:
    path = resolve_upstream(upstream_dir)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    trajectron_dir = path / "trajectron"
    if str(trajectron_dir) not in sys.path:
        sys.path.insert(0, str(trajectron_dir))
    return path


def upstream_commit(upstream_dir: str | Path | None = None) -> str | None:
    path = resolve_upstream(upstream_dir)
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
