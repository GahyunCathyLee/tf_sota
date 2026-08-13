"""Helpers for importing the official SIMPL checkout."""

from __future__ import annotations

import subprocess
import sys
import fractions
import math
import types
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]


def resolve_upstream_dir(path: str | Path | None = None) -> Path:
    upstream = Path(path) if path is not None else EXPERIMENT_ROOT / "external" / "simpl"
    upstream = upstream.expanduser()
    if not upstream.is_absolute():
        upstream = (EXPERIMENT_ROOT / upstream).resolve()
    if not upstream.exists():
        raise FileNotFoundError(
            f"SIMPL upstream checkout not found: {upstream}\n"
            "Clone it with: git clone https://github.com/HKUST-Aerial-Robotics/SIMPL.git external/simpl"
        )
    return upstream


def add_upstream_to_path(path: str | Path | None = None) -> Path:
    upstream = resolve_upstream_dir(path)
    if not hasattr(fractions, "gcd"):
        fractions.gcd = math.gcd
    if "imp" not in sys.modules:
        sys.modules["imp"] = types.ModuleType("imp")
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    return upstream


def upstream_commit(path: str | Path | None = None) -> str | None:
    upstream = resolve_upstream_dir(path)
    try:
        return subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
