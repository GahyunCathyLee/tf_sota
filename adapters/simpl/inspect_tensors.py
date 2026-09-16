#!/usr/bin/env python3
"""Inspect SIMPL actor tensors for matched highD/exiD samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ADAPTER_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ADAPTER_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from adapters.common import dataset_dir, split_indices_path  # noqa: E402
from adapters.simpl.dataset import NeighFormerSIMPLDataset, SIMPL_FEATURE_MODES  # noqa: E402
from adapters.simpl.train import resolve_path  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["highD", "exiD"])
    p.add_argument("--data-root", default="data", type=Path)
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--feature-mode", default="dimI", choices=sorted(SIMPL_FEATURE_MODES))
    p.add_argument("--compare-mode", default="baseline", choices=sorted(SIMPL_FEATURE_MODES))
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--lane-cache-root", type=Path)
    p.add_argument("--lane-half-length", type=float, default=120.0)
    p.add_argument("--lane-radius", type=float, default=120.0)
    p.add_argument("--lane-max-segments", type=int, default=192)
    return p.parse_args(argv)


def _as_np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _one_actor_tensor(ds: NeighFormerSIMPLDataset, sample_offset: int) -> tuple[dict[str, Any], np.ndarray]:
    item = ds[sample_offset]
    batch = ds.collate_fn([item])
    # ACTORS is (num_actors, feature_dim, history); transpose for readability.
    actors = _as_np(batch["ACTORS"]).transpose(0, 2, 1)
    return item, actors


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = resolve_path(args.data_root)
    data_path = dataset_dir(data_root, args.dataset)
    indices = np.load(split_indices_path(data_root, args.dataset, args.split))
    indices = indices[args.start : args.start + args.num_samples]
    lane_cache_root = resolve_path(args.lane_cache_root) if args.lane_cache_root else None
    print(f"[INFO] data={data_path}")
    print(f"[INFO] split={args.split} samples={indices.tolist()}")
    print(f"[INFO] lane_cache={lane_cache_root if lane_cache_root else 'pseudo fallback'} exists={bool(lane_cache_root and lane_cache_root.exists())}")

    base = NeighFormerSIMPLDataset(
        data_path,
        indices,
        args.dataset,
        args.compare_mode,
        args.split,
        args.lane_half_length,
        lane_cache_root=lane_cache_root,
        lane_radius=args.lane_radius,
        lane_max_segments=args.lane_max_segments,
    )
    test = NeighFormerSIMPLDataset(
        data_path,
        indices,
        args.dataset,
        args.feature_mode,
        args.split,
        args.lane_half_length,
        lane_cache_root=lane_cache_root,
        lane_radius=args.lane_radius,
        lane_max_segments=args.lane_max_segments,
    )

    for j in range(len(indices)):
        base_item, base_actor = _one_actor_tensor(base, j)
        test_item, test_actor = _one_actor_tensor(test, j)
        assert base_item["SAMPLE_INDEX"] == test_item["SAMPLE_INDEX"]
        assert base_item["META"] == test_item["META"]
        assert base_actor.shape[0] == test_actor.shape[0], "actor ordering/count changed"
        assert np.allclose(base_actor[:, :, :3], test_actor[:, :, :3]), "original SIMPL channels differ"
        assert np.allclose(base_item["TRAJS_FUT"], test_item["TRAJS_FUT"]), "future target differs"
        assert np.array_equal(base_item["PAD_OBS"], test_item["PAD_OBS"]), "PAD_OBS differs"
        assert np.array_equal(base_item["PAD_FUT"], test_item["PAD_FUT"]), "PAD_FUT differs"
        if test_actor.shape[-1] > 3:
            extras = test_actor[:, :, 3:]
            padded = test_item["PAD_OBS"] == 0
            if padded.any():
                assert np.allclose(extras[padded], 0.0), "side channels leak through padded timesteps"
            assert np.allclose(extras[0], 0.0), "ego side channels are not neutral zero"

        print("\n" + "=" * 80)
        print(f"sample_offset={j} sample_index={base_item['SAMPLE_INDEX']} meta={base_item['META']}")
        print(f"actor_count={base_actor.shape[0]}")
        print(f"{args.compare_mode} actor_shape={base_actor.shape} features={base.actor_feature_names}")
        print(f"{args.feature_mode} actor_shape={test_actor.shape} features={test.actor_feature_names}")
        print(f"ego {args.compare_mode} first_rows=\n{base_actor[0, :min(4, base_actor.shape[1])]}")
        print(f"ego {args.feature_mode} first_rows=\n{test_actor[0, :min(4, test_actor.shape[1])]}")
        if base_actor.shape[0] > 1:
            print(f"neighbor0 {args.compare_mode} first_rows=\n{base_actor[1, :min(4, base_actor.shape[1])]}")
            print(f"neighbor0 {args.feature_mode} first_rows=\n{test_actor[1, :min(4, test_actor.shape[1])]}")
        print(f"pad_obs first_actor={test_item['PAD_OBS'][0].astype(int).tolist()}")
        if test_item["PAD_OBS"].shape[0] > 1:
            print(f"pad_obs neighbor0={test_item['PAD_OBS'][1].astype(int).tolist()}")
        print("assertions=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
