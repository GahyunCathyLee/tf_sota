#!/usr/bin/env python3
"""Preprocess NeighFormer npy data into reusable MTR++ adapter shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ADAPTER_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ADAPTER_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from adapters.common import dataset_dir, split_indices_path  # noqa: E402
from adapters.mtrpp.dataset import (  # noqa: E402
    NeighFormerMTRBuilder,
    build_intention_points_from_data,
    processed_root,
    save_processed_split,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["highD", "exiD"])
    p.add_argument("--feature-mode", required=True, choices=["baseline", "dimI"])
    p.add_argument("--data-root", type=Path, default=Path("/home/gahyun/neighformer/data"))
    p.add_argument("--processed-dir", type=Path, default=Path("processed/mtrpp"))
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    p.add_argument("--max-samples", type=int)
    p.add_argument("--shard-size", type=int, default=4096)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dt", type=float, default=0.32)
    p.add_argument("--num-motion-modes", type=int, default=6)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    data_path = dataset_dir(data_root, args.dataset)
    out_root = processed_root(args.processed_dir, args.dataset, args.feature_mode)
    builder = NeighFormerMTRBuilder(data_path, args.dataset, args.feature_mode, "preprocess", dt=args.dt)
    report: dict[str, Any] = {"processed_root": str(out_root), "splits": {}}
    train_indices = np.load(split_indices_path(data_root, args.dataset, "train"))
    build_intention_points_from_data(
        data_path,
        train_indices,
        out_root / "intention_points.pkl",
        num_modes=args.num_motion_modes,
    )
    for split in args.splits:
        indices = np.load(split_indices_path(data_root, args.dataset, split))
        if args.max_samples is not None:
            indices = indices[: args.max_samples]
        builder.split = split
        report["splits"][split] = save_processed_split(
            builder,
            indices,
            out_root,
            split,
            shard_size=args.shard_size,
            overwrite=args.overwrite,
        )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
