#!/usr/bin/env python3
"""Colab-friendly training dispatcher for model adapters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_raw_config(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise SystemExit("Circular config base chain: " + " -> ".join(str(p) for p in (*seen, path)))
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base_ref = raw.pop("base", None)
    if not base_ref:
        return raw
    candidates = [Path(base_ref)] if Path(base_ref).is_absolute() else [path.parent / base_ref, ROOT / base_ref]
    for cand in candidates:
        if cand.exists():
            return _deep_merge(load_raw_config(cand, (*seen, path)), raw)
    raise SystemExit(f"{path}: base config '{base_ref}' not found")


def detect_adapter(config: Path) -> str:
    raw = load_raw_config(config)
    adapter = raw.get("adapter")
    if adapter:
        return str(adapter)
    if "/simpl/" in str(config).replace("\\", "/"):
        return "simpl"
    return "qcnet"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True, type=Path)
    known, _ = parser.parse_known_args(argv)
    adapter = detect_adapter(known.config)
    if adapter == "simpl":
        from adapters.simpl.train import main as adapter_main
    elif adapter == "qcnet":
        from adapters.qcnet.train import main as adapter_main
    else:
        raise SystemExit(f"Unknown adapter '{adapter}' in {known.config}")
    return adapter_main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
