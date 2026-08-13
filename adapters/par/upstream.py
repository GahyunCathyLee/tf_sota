"""Helpers for locating the official PAR checkout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]


def resolve_upstream(upstream_dir: str | Path = "external/par") -> Path:
    path = Path(upstream_dir).expanduser()
    if not path.is_absolute():
        path = (EXPERIMENT_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"PAR upstream checkout not found: {path}\n"
            "Clone it with: git clone https://github.com/neerjathakkar/PAR.git external/par"
        )
    return path


def add_upstream_to_path(upstream_dir: str | Path = "external/par") -> Path:
    path = resolve_upstream(upstream_dir)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def upstream_commit(upstream_dir: str | Path = "external/par") -> str | None:
    path = resolve_upstream(upstream_dir)
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None

