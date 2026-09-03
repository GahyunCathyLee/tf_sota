"""Build Trajectron++ Environment pickles from NeighFormer npy windows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dill
import numpy as np

from adapters.common import feature_mode_indices, feature_mode_names

BASE_STATE = {
    "position": ["x", "y"],
    "velocity": ["x", "y"],
    "acceleration": ["x", "y"],
}
DIMI_STATE = {
    "position": ["x", "y"],
    "velocity": ["x", "y"],
    "acceleration": ["x", "y"],
    "neighbor": ["dim", "I"],
}
PRED_STATE = {"position": ["x", "y"]}


def state_spec(feature_mode: str) -> dict[str, list[str]]:
    return DIMI_STATE if feature_mode == "dimI" else BASE_STATE


def state_header(feature_mode: str) -> list[tuple[str, str]]:
    return [(group, dim) for group, dims in state_spec(feature_mode).items() for dim in dims]


def standardization(feature_mode: str) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    spec = state_spec(feature_mode)
    out: dict[str, dict[str, dict[str, dict[str, float]]]] = {"VEHICLE": {}}
    defaults = {
        ("position", "x"): 1.0,
        ("position", "y"): 1.0,
        ("velocity", "x"): 5.0,
        ("velocity", "y"): 2.0,
        ("acceleration", "x"): 1.0,
        ("acceleration", "y"): 1.0,
        ("neighbor", "dim"): 1.0,
        ("neighbor", "I"): 1.0,
    }
    for group, dims in spec.items():
        out["VEHICLE"][group] = {
            dim: {"mean": 0.0, "std": float(defaults.get((group, dim), 1.0))}
            for dim in dims
        }
    return out


def _future_kinematics(arrays: dict[str, np.ndarray], real_idx: int, last_hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(arrays["y"][real_idx], dtype=np.float32)
    if "y_vel" in arrays:
        y_vel = np.asarray(arrays["y_vel"][real_idx], dtype=np.float32)
    else:
        pos = np.concatenate([last_hist[None, 0:2], y], axis=0)
        y_vel = np.diff(pos, axis=0).astype(np.float32)
    if "y_acc" in arrays:
        y_acc = np.asarray(arrays["y_acc"][real_idx], dtype=np.float32)
    else:
        vel = np.concatenate([last_hist[None, 2:4], y_vel], axis=0)
        y_acc = np.diff(vel, axis=0).astype(np.float32)
    return y_vel, y_acc


def _fill_history(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = values.copy()
    if mask.all():
        return out
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        return np.zeros_like(out)
    for t in range(out.shape[0]):
        if mask[t]:
            continue
        nearest = valid[np.argmin(np.abs(valid - t))]
        out[t] = out[nearest]
    return out


def _sample_to_scene(
    arrays: dict[str, np.ndarray],
    real_idx: int,
    dataset_name: str,
    feature_mode: str,
    env_node_type: Any,
    scene_cls: Any,
    node_cls: Any,
    dha_cls: Any,
    dt: float,
) -> Any:
    ego = np.asarray(arrays["x_ego"][real_idx], dtype=np.float32)
    nb = np.asarray(arrays["x_nb"][real_idx], dtype=np.float32)
    mask = np.asarray(arrays["nb_mask"][real_idx], dtype=bool)
    fut = np.asarray(arrays["y"][real_idx], dtype=np.float32)
    y_vel, y_acc = _future_kinematics(arrays, real_idx, ego[-1])
    th, tf = ego.shape[0], fut.shape[0]
    feat_dim = len(state_header(feature_mode))
    header = state_header(feature_mode)

    scene = scene_cls(timesteps=th + tf, dt=dt, name=f"{dataset_name}_{real_idx}")

    ego_data = np.zeros((th + tf, feat_dim), dtype=np.float32)
    ego_data[:th, :6] = ego[:, :6]
    ego_data[th:, 0:2] = fut
    ego_data[th:, 2:4] = y_vel
    ego_data[th:, 4:6] = y_acc
    if feature_mode == "dimI":
        ego_data[:, 6:] = -1.0
    scene.nodes.append(
        node_cls(
            node_type=env_node_type,
            node_id=f"ego_{real_idx}",
            data=dha_cls(ego_data, header),
            first_timestep=0,
        )
    )

    # Only neighbors present at the prediction instant are valid Trajectron++
    # context nodes. Missing earlier history is nearest-filled to preserve the
    # contiguous-track assumption in the original dataloader.
    for slot in np.flatnonzero(mask[-1]):
        hist = np.zeros((th, feat_dim), dtype=np.float32)
        hist[:, 0:2] = ego[:, 0:2] + nb[:, slot, 0:2]
        hist[:, 2:4] = ego[:, 2:4] + nb[:, slot, 2:4]
        hist[:, 4:6] = ego[:, 4:6] + nb[:, slot, 4:6]
        if feature_mode == "dimI":
            hist[:, 6:] = nb[:, slot, [8, 9]]
        hist = _fill_history(hist, mask[:, slot])
        scene.nodes.append(
            node_cls(
                node_type=env_node_type,
                node_id=f"nb{int(slot)}_{real_idx}",
                data=dha_cls(hist, header),
                first_timestep=0,
            )
        )
    return scene


def load_arrays(data_dir: Path) -> dict[str, np.ndarray]:
    arrays = {
        "x_ego": np.load(data_dir / "x_ego.npy", mmap_mode="r"),
        "x_nb": np.load(data_dir / "x_nb.npy", mmap_mode="r"),
        "nb_mask": np.load(data_dir / "nb_mask.npy", mmap_mode="r"),
        "y": np.load(data_dir / "y.npy", mmap_mode="r"),
    }
    for name in ("y_vel", "y_acc", "meta_recordingId", "meta_trackId", "meta_frame"):
        path = data_dir / f"{name}.npy"
        if path.exists():
            arrays[name] = np.load(path, mmap_mode="r")
    return arrays


def build_environment(
    data_dir: Path,
    indices: np.ndarray,
    dataset_name: str,
    feature_mode: str,
    split: str,
    upstream_dir: Path,
    dt: float = 0.32,
) -> tuple[Any, dict[str, Any]]:
    import sys

    if str(upstream_dir) not in sys.path:
        sys.path.insert(0, str(upstream_dir))
    if str(upstream_dir / "trajectron") not in sys.path:
        sys.path.insert(0, str(upstream_dir / "trajectron"))

    from environment.environment import Environment
    from environment.scene import Scene
    from environment.node import Node
    from environment.data_structures import DoubleHeaderNumpyArray

    env = Environment(node_type_list=["VEHICLE"], standardization=standardization(feature_mode))
    env.attention_radius = {(env.NodeType.VEHICLE, env.NodeType.VEHICLE): 100.0}
    arrays = load_arrays(data_dir)
    scenes = [
        _sample_to_scene(
            arrays,
            int(real_idx),
            dataset_name,
            feature_mode,
            env.NodeType.VEHICLE,
            Scene,
            Node,
            DoubleHeaderNumpyArray,
            dt,
        )
        for real_idx in np.asarray(indices, dtype=np.int64)
    ]
    env.scenes = scenes
    report = {
        "dataset": dataset_name,
        "split": split,
        "feature_mode": feature_mode,
        "data_dir": str(data_dir),
        "num_samples": int(len(indices)),
        "num_scenes": int(len(scenes)),
        "history_len": int(arrays["x_ego"].shape[1]),
        "future_len": int(arrays["y"].shape[1]),
        "state": {"VEHICLE": state_spec(feature_mode)},
        "pred_state": {"VEHICLE": PRED_STATE},
        "neighbor_indices": feature_mode_indices(feature_mode),
        "neighbor_names": feature_mode_names(feature_mode),
        "attention_radius": 100.0,
        "dt": float(dt),
        "nodes": {
            "min": int(min(len(s.nodes) for s in scenes)) if scenes else 0,
            "max": int(max(len(s.nodes) for s in scenes)) if scenes else 0,
            "mean": float(np.mean([len(s.nodes) for s in scenes])) if scenes else 0.0,
        },
    }
    return env, report


def write_environment(path: Path, env: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
