#!/usr/bin/env python3
"""Colab-friendly evaluation dispatcher for model adapters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_checkpoint(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def detect_adapter(ckpt_path: Path) -> str:
    path = ckpt_path if ckpt_path.is_absolute() else (ROOT / ckpt_path).resolve()
    ckpt = load_checkpoint(path)
    cfg = ckpt.get("cfg", {})
    adapter = cfg.get("adapter")
    if adapter:
        return str(adapter)
    hp = ckpt.get("hyper_parameters", {}).get("args")
    if hp is not None and getattr(hp, "feature_mode", None) is not None:
        return "mtp_go"
    if "/mtp_go/" in str(path).replace("\\", "/"):
        return "mtp_go"
    if "/hivt/" in str(path).replace("\\", "/"):
        return "hivt"
    if "/par/" in str(path).replace("\\", "/"):
        return "par"
    if "/bat/" in str(path).replace("\\", "/"):
        return "bat"
    if "model_cfg" in ckpt or "/simpl/" in str(path).replace("\\", "/"):
        return "simpl"
    return "qcnet"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", help="Optional legacy selector; adapter is inferred from --ckpt.")
    parser.add_argument("--ckpt", required=True, type=Path)
    known, _ = parser.parse_known_args(argv)
    adapter = str(known.model) if known.model else detect_adapter(known.ckpt)
    if adapter == "mtp_go":
        from adapters.mtp_go.evaluate import main as adapter_main
    elif adapter == "hivt":
        from adapters.hivt.evaluate import main as adapter_main
    elif adapter == "mtrpp":
        from adapters.mtrpp.evaluate import main as adapter_main
    elif adapter == "simpl":
        from adapters.simpl.evaluate import main as adapter_main
    elif adapter == "par":
        from adapters.par.evaluate import main as adapter_main
    elif adapter == "bat":
        from adapters.bat.evaluate import main as adapter_main
    elif adapter == "qcnet":
        from adapters.qcnet.evaluate import main as adapter_main
    elif adapter == "trajectronpp":
        from adapters.trajectronpp.evaluate import main as adapter_main
    else:
        raise SystemExit(f"Unknown adapter '{adapter}' in {known.ckpt}")
    return adapter_main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
