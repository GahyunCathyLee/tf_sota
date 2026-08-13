#!/usr/bin/env python3
"""Colab-friendly evaluation dispatcher for model adapters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent


def detect_adapter(ckpt_path: Path) -> str:
    path = ckpt_path if ckpt_path.is_absolute() else (ROOT / ckpt_path).resolve()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    adapter = cfg.get("adapter")
    if adapter:
        return str(adapter)
    if "model_cfg" in ckpt or "/simpl/" in str(path).replace("\\", "/"):
        return "simpl"
    return "qcnet"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ckpt", required=True, type=Path)
    known, _ = parser.parse_known_args(argv)
    adapter = detect_adapter(known.ckpt)
    if adapter == "simpl":
        from adapters.simpl.evaluate import main as adapter_main
    elif adapter == "qcnet":
        from adapters.qcnet.evaluate import main as adapter_main
    else:
        raise SystemExit(f"Unknown adapter '{adapter}' in {known.ckpt}")
    return adapter_main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
