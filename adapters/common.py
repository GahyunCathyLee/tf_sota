"""Common utilities for highD/exiD SOTA model adapters.

The canonical data source is the NeighFormer mmap/npy preprocessing output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_MODES = {
    "baseline": {
        "neighbor_indices": [0, 1, 2, 3, 4, 5],
        "neighbor_names": ["dx", "dy", "dvx", "dvy", "dax", "day"],
    },
    "dimI": {
        "neighbor_indices": [0, 1, 2, 3, 4, 5, 8, 9],
        "neighbor_names": ["dx", "dy", "dvx", "dvy", "dax", "day", "dim", "I"],
    },
}

REQUIRED_ARRAYS = (
    "x_ego.npy",
    "x_nb.npy",
    "nb_mask.npy",
    "y.npy",
    "x_last_abs.npy",
)

OPTIONAL_ARRAYS = (
    "y_vel.npy",
    "y_acc.npy",
    "meta_recordingId.npy",
    "meta_trackId.npy",
    "meta_frame.npy",
    "scenario_labels.csv",
)


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    feature_mode: str
    split: str
    data_dir: Path
    split_indices_path: Path | None = None


def dataset_dir(root: Path, dataset: str) -> Path:
    """Return the NeighFormer mmap/npy data directory for a dataset."""
    return root / dataset / "dimI"


def split_indices_path(root: Path, dataset: str, split: str) -> Path:
    return root / dataset / "splits" / f"{split}_indices.npy"


def feature_mode_indices(feature_mode: str) -> list[int]:
    return list(FEATURE_MODES[feature_mode]["neighbor_indices"])


def feature_mode_names(feature_mode: str) -> list[str]:
    return list(FEATURE_MODES[feature_mode]["neighbor_names"])


def validate_dataset_spec(spec: DatasetSpec, require_split: bool = False) -> None:
    if spec.dataset not in {"highD", "exiD"}:
        raise ValueError(f"Unsupported dataset: {spec.dataset}")
    if spec.feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unsupported feature mode: {spec.feature_mode}")
    if spec.split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {spec.split}")
    if not spec.data_dir.exists():
        raise FileNotFoundError(spec.data_dir)
    for filename in REQUIRED_ARRAYS:
        path = spec.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
    if require_split and spec.split_indices_path is not None and not spec.split_indices_path.exists():
        raise FileNotFoundError(spec.split_indices_path)


def inspect_neighformer_dir(data_dir: Path) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for filename in REQUIRED_ARRAYS + OPTIONAL_ARRAYS:
        path = data_dir / filename
        if not path.exists():
            info[filename] = {"exists": False}
            continue
        if path.suffix == ".npy":
            arr = np.load(path, mmap_mode="r")
            info[filename] = {
                "exists": True,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }
        else:
            info[filename] = {
                "exists": True,
                "path": str(path),
            }
    return info


def inspect_split(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    arr = np.load(path, mmap_mode="r")
    return {
        "exists": True,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": int(arr.min()) if arr.size else None,
        "max": int(arr.max()) if arr.size else None,
    }


def ego_channel_count() -> int:
    return 6


def neighbor_channel_count(feature_mode: str) -> int:
    return len(feature_mode_indices(feature_mode))
