#!/usr/bin/env python3
"""Inspect canonical NeighFormer highD/exiD npy files used by SOTA adapters."""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("SOTA_DATA_ROOT", EXPERIMENT_ROOT.parent / "neighformer" / "data")
).resolve()
sys.path.insert(0, str(EXPERIMENT_ROOT.parent))

from sota_experiments.adapters.common import (
    DatasetSpec,
    dataset_dir,
    ego_channel_count,
    feature_mode_indices,
    feature_mode_names,
    inspect_neighformer_dir,
    inspect_split,
    neighbor_channel_count,
    split_indices_path,
    validate_dataset_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["highD", "exiD"], required=True)
    parser.add_argument("--feature-mode", choices=["baseline", "dimI"], required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--require-split", action="store_true")
    args = parser.parse_args()

    data_dir = dataset_dir(args.root, args.dataset)
    split_path = split_indices_path(args.root, args.dataset, args.split)
    spec = DatasetSpec(args.dataset, args.feature_mode, args.split, data_dir, split_path)
    validate_dataset_spec(spec, require_split=args.require_split)

    print(f"data_root     : {args.root}")
    print(f"data_dir      : {data_dir}")
    print(f"dataset       : {args.dataset}")
    print(f"feature_mode  : {args.feature_mode}")
    print(f"ego_channels  : {ego_channel_count()}")
    print(f"nb_channels   : {neighbor_channel_count(args.feature_mode)}")
    print(f"nb_indices    : {feature_mode_indices(args.feature_mode)}")
    print(f"nb_names      : {feature_mode_names(args.feature_mode)}")
    print()

    for key, info in inspect_neighformer_dir(data_dir).items():
        if not info["exists"]:
            print(f"{key:24s} MISSING")
        elif "shape" in info:
            print(f"{key:24s} shape={info['shape']} dtype={info['dtype']}")
        else:
            print(f"{key:24s} path={info['path']}")

    split_info = inspect_split(split_path)
    print()
    if split_info["exists"]:
        print(
            f"{split_path.name:24s} shape={split_info['shape']} "
            f"dtype={split_info['dtype']} min={split_info['min']} max={split_info['max']}"
        )
    else:
        print(f"{split_path.name:24s} MISSING ({split_path})")


if __name__ == "__main__":
    main()
