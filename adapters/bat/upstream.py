"""Helpers for importing the official BAT checkout."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]


def resolve_upstream_dir(path: str | Path | None = None) -> Path:
    upstream = Path(path) if path is not None else EXPERIMENT_ROOT / "external" / "bat"
    upstream = upstream.expanduser()
    if not upstream.is_absolute():
        upstream = (EXPERIMENT_ROOT / upstream).resolve()
    if not upstream.exists():
        raise FileNotFoundError(
            f"BAT upstream checkout not found: {upstream}\n"
            "Clone it with: git clone https://github.com/Petrichor625/BATraj-Behavior-aware-Model.git external/bat"
        )
    return upstream


def _install_einops_stub() -> None:
    """BAT imports einops symbols that are unused in the released model file."""
    if "einops" in sys.modules:
        return
    try:
        __import__("einops")
        return
    except ImportError:
        pass
    module = types.ModuleType("einops")

    def _missing(*_: Any, **__: Any) -> None:
        raise RuntimeError("einops is required for this operation")

    module.rearrange = _missing
    module.repeat = _missing
    sys.modules["einops"] = module


def import_bat_model(path: str | Path | None = None):
    upstream = resolve_upstream_dir(path)
    _install_einops_stub()
    model_file = upstream / "model5f_BAT_new.py"
    if not model_file.exists():
        raise FileNotFoundError(model_file)
    spec = importlib.util.spec_from_file_location("sota_bat_model5f_BAT_new", model_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import BAT model from {model_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GDEncoder, module.Generator, upstream


def upstream_commit(path: str | Path | None = None) -> str | None:
    upstream = resolve_upstream_dir(path)
    try:
        return subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
