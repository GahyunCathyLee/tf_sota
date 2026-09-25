#!/usr/bin/env python3
"""Measure exiD baseline/+I latency for SIMPL adapter."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CASES = [("exiD-baseline", "exiD0-5/best.pt"), ("exiD-+I", "exiD2-5/best.pt")]
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--ckpt-root", type=Path, default=Path("ckpts/simpl"))
    p.add_argument("--exid-base-ckpt", type=Path)
    p.add_argument("--exid-i-ckpt", type=Path)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--log-dir", type=Path, default=Path("logs/latency/simpl"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    args.log_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "auto":
        probe = subprocess.run(
            [args.python, "-c", "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if probe.returncode == 0:
            device = probe.stdout.strip() or "cpu"
        else:
            device = "cpu"
    failures = 0
    for name, rel_ckpt in CASES:
        ckpt = args.exid_base_ckpt if name.endswith("baseline") else args.exid_i_ckpt
        ckpt = ckpt or args.ckpt_root / rel_ckpt
        if not ckpt.is_absolute():
            ckpt = root / ckpt
        if not ckpt.exists():
            print(f"[SKIP] {name}: missing {ckpt}")
            continue
        cmd = [
            args.python, "adapters/simpl/evaluate.py", "--ckpt", str(ckpt),
            "--split", "test", "--measure-time", "--warmup", str(args.warmup), "--iters", str(args.iters),
            "--device", device,
        ]
        if args.data_root:
            cmd += ["--data-root", str(args.data_root)]
        print(f"[RUN] simpl {name}\n  {' '.join(cmd)}")
        if args.dry_run:
            continue
        with (args.log_dir / f"{name}.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False)
        failures += int(proc.returncode != 0)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
