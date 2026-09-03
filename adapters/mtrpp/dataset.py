"""NeighFormer npy -> official MTR batch dictionaries."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from adapters.common import feature_mode_indices, feature_mode_names

OBJECT_TYPE_VEHICLE = "TYPE_VEHICLE"
EGO_EXTRA_FILL = -1.0


def _heading_from_velocity(vxy: np.ndarray) -> np.ndarray:
    return np.arctan2(vxy[..., 1], vxy[..., 0]).astype(np.float32)


def _velocity_from_positions(pos: np.ndarray, dt: float) -> np.ndarray:
    vel = np.zeros_like(pos, dtype=np.float32)
    if pos.shape[0] > 1:
        vel[1:] = (pos[1:] - pos[:-1]) / float(dt)
        vel[0] = vel[1]
    return vel


def _as_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _as_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_serializable(v) for v in value]
    return value


class NeighFormerMTRBuilder:
    """Build MTR scene-level samples from canonical NeighFormer arrays."""

    def __init__(
        self,
        data_dir: str | Path,
        dataset_name: str,
        feature_mode: str,
        split: str,
        dt: float = 0.32,
        vehicle_length: float = 4.8,
        vehicle_width: float = 2.0,
        vehicle_height: float = 1.6,
        map_polylines: int = 9,
        map_points_each_polyline: int = 20,
        lane_half_length: float = 160.0,
        lane_width: float = 3.7,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.feature_mode = feature_mode
        self.split = split
        self.dt = float(dt)
        self.vehicle_size = np.array([vehicle_length, vehicle_width, vehicle_height], dtype=np.float32)
        self.map_polylines = int(map_polylines)
        self.map_points_each_polyline = int(map_points_each_polyline)
        self.lane_half_length = float(lane_half_length)
        self.lane_width = float(lane_width)
        self.nb_feature_indices = np.asarray(feature_mode_indices(feature_mode), dtype=np.int64)
        self.nb_feature_names = feature_mode_names(feature_mode)
        self._arrays: dict[str, np.ndarray] | None = None

        x_ego = np.load(self.data_dir / "x_ego.npy", mmap_mode="r")
        x_nb = np.load(self.data_dir / "x_nb.npy", mmap_mode="r")
        y = np.load(self.data_dir / "y.npy", mmap_mode="r")
        self.n_samples_total = int(x_ego.shape[0])
        self.history_len = int(x_ego.shape[1])
        self.future_len = int(y.shape[1])
        self.raw_max_neighbors = int(x_nb.shape[2])
        self.extra_dim = 2 if feature_mode == "dimI" else 0
        self.agent_attr_dim = 6 + 5 + (self.history_len + 1) + 2 + 2 + 2 + self.extra_dim
        if int(x_ego.shape[2]) != 6:
            raise ValueError(f"Expected x_ego[..., 6], got {x_ego.shape}")
        if int(x_nb.shape[3]) < int(self.nb_feature_indices.max()) + 1:
            raise ValueError(
                f"x_nb has {x_nb.shape[3]} channels; {feature_mode} needs index "
                f"{int(self.nb_feature_indices.max())}"
            )

    def _ensure_open(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            arrays = {
                "x_ego": np.load(self.data_dir / "x_ego.npy", mmap_mode="r"),
                "x_nb": np.load(self.data_dir / "x_nb.npy", mmap_mode="r"),
                "nb_mask": np.load(self.data_dir / "nb_mask.npy", mmap_mode="r"),
                "y": np.load(self.data_dir / "y.npy", mmap_mode="r"),
            }
            for name in ("x_last_abs", "y_vel", "meta_recordingId", "meta_trackId", "meta_frame"):
                path = self.data_dir / f"{name}.npy"
                if path.exists():
                    arrays[name] = np.load(path, mmap_mode="r")
            self._arrays = arrays
        return self._arrays

    def build(self, real_idx: int) -> dict[str, Any]:
        arrays = self._ensure_open()
        ego = np.asarray(arrays["x_ego"][real_idx], dtype=np.float32)
        nb = np.asarray(arrays["x_nb"][real_idx], dtype=np.float32)
        mask = np.asarray(arrays["nb_mask"][real_idx], dtype=bool)
        fut = np.asarray(arrays["y"][real_idx], dtype=np.float32)
        fut_vel = (
            np.asarray(arrays["y_vel"][real_idx], dtype=np.float32)
            if "y_vel" in arrays
            else _velocity_from_positions(np.vstack([ego[-1:, 0:2], fut]), self.dt)[1:]
        )

        slots = np.flatnonzero(mask.any(axis=0))
        num_objects = 1 + int(slots.size)
        past_state = np.zeros((num_objects, self.history_len, 10), dtype=np.float32)
        past_mask = np.zeros((num_objects, self.history_len), dtype=bool)
        extra = np.zeros((num_objects, self.history_len, self.extra_dim), dtype=np.float32)
        if self.extra_dim:
            extra[0, :, :] = EGO_EXTRA_FILL

        past_state[0, :, 0:2] = ego[:, 0:2]
        past_state[0, :, 3:6] = self.vehicle_size
        past_state[0, :, 7:9] = ego[:, 2:4]
        past_state[0, :, 6] = _heading_from_velocity(ego[:, 2:4])
        past_state[0, :, 9] = 1.0
        past_mask[0] = True

        for obj_idx, slot in enumerate(slots, start=1):
            slot_mask = mask[:, slot]
            hist_pos = ego[:, 0:2] + nb[:, slot, 0:2]
            hist_vel = ego[:, 2:4] + nb[:, slot, 2:4]
            past_state[obj_idx, :, 0:2] = hist_pos
            past_state[obj_idx, :, 3:6] = self.vehicle_size
            past_state[obj_idx, :, 6] = _heading_from_velocity(hist_vel)
            past_state[obj_idx, :, 7:9] = hist_vel
            past_state[obj_idx, :, 9] = slot_mask.astype(np.float32)
            past_mask[obj_idx] = slot_mask
            if self.extra_dim:
                extra[obj_idx, :, :] = nb[:, slot, 8:10]

        obj_trajs = self._pack_agent_features(past_state, past_mask, extra)
        obj_trajs_pos = past_state[None, :, :, 0:3].copy()
        obj_trajs_last_pos = np.zeros((1, num_objects, 3), dtype=np.float32)
        for obj_idx in range(num_objects):
            valid_steps = np.flatnonzero(past_mask[obj_idx])
            if valid_steps.size:
                obj_trajs_last_pos[0, obj_idx] = past_state[obj_idx, valid_steps[-1], 0:3]

        future_state = np.zeros((1, num_objects, self.future_len, 4), dtype=np.float32)
        future_mask = np.zeros((1, num_objects, self.future_len), dtype=bool)
        future_state[0, 0, :, 0:2] = fut
        future_state[0, 0, :, 2:4] = fut_vel
        future_mask[0, 0, :] = True

        center_gt = future_state[:, 0].copy()
        center_gt_mask = future_mask[:, 0].copy()
        center_src = np.zeros((1, self.future_len, 10), dtype=np.float32)
        center_src[0, :, 0:2] = fut
        center_src[0, :, 3:6] = self.vehicle_size
        center_src[0, :, 6] = _heading_from_velocity(fut_vel)
        center_src[0, :, 7:9] = fut_vel
        center_src[0, :, 9] = 1.0

        center_world = np.zeros((1, 10), dtype=np.float32)
        center_world[0, 0:2] = arrays["x_last_abs"][real_idx] if "x_last_abs" in arrays else 0.0
        center_world[0, 3:6] = self.vehicle_size
        center_world[0, 6] = 0.0
        center_world[0, 7:9] = ego[-1, 2:4]
        center_world[0, 9] = 1.0

        scenario_id = self._scenario_id(real_idx, arrays)
        map_data, map_mask, map_center = self._pseudo_map()
        return {
            "scenario_id": np.array([scenario_id]),
            "obj_trajs": obj_trajs[None],
            "obj_trajs_mask": past_mask[None],
            "track_index_to_predict": np.array([0], dtype=np.int64),
            "obj_trajs_pos": obj_trajs_pos,
            "obj_trajs_last_pos": obj_trajs_last_pos,
            "obj_types": np.array([OBJECT_TYPE_VEHICLE] * num_objects),
            "obj_ids": np.arange(num_objects, dtype=np.int64),
            "center_objects_world": center_world,
            "center_objects_id": np.array([self._object_id(real_idx, arrays)], dtype=np.int64),
            "center_objects_type": np.array([OBJECT_TYPE_VEHICLE]),
            "obj_trajs_future_state": future_state,
            "obj_trajs_future_mask": future_mask,
            "center_gt_trajs": center_gt,
            "center_gt_trajs_mask": center_gt_mask,
            "center_gt_final_valid_idx": np.array([self.future_len - 1], dtype=np.float32),
            "center_gt_trajs_src": center_src,
            "map_polylines": map_data[None],
            "map_polylines_mask": map_mask[None],
            "map_polylines_center": map_center[None],
            "sample_index": np.array([real_idx], dtype=np.int64),
        }

    def _pack_agent_features(self, state: np.ndarray, valid: np.ndarray, extra: np.ndarray) -> np.ndarray:
        num_objects = state.shape[0]
        onehot = np.zeros((num_objects, self.history_len, 5), dtype=np.float32)
        onehot[:, :, 0] = 1.0
        onehot[0, :, 3] = 1.0
        onehot[0, :, 4] = 1.0

        time_embed = np.zeros((num_objects, self.history_len, self.history_len + 1), dtype=np.float32)
        for t in range(self.history_len):
            time_embed[:, t, t] = 1.0
            time_embed[:, t, -1] = (t - self.history_len + 1) * self.dt

        heading_embed = np.stack([np.sin(state[:, :, 6]), np.cos(state[:, :, 6])], axis=-1).astype(np.float32)
        vel = state[:, :, 7:9]
        acc = np.zeros_like(vel)
        if self.history_len > 1:
            acc[:, 1:] = (vel[:, 1:] - vel[:, :-1]) / self.dt
            acc[:, 0] = acc[:, 1]

        parts = [state[:, :, 0:6], onehot, time_embed, heading_embed, vel, acc]
        if self.extra_dim:
            parts.append(extra)
        out = np.concatenate(parts, axis=-1).astype(np.float32)
        out[~valid] = 0.0
        return out

    def _pseudo_map(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = max(1, self.map_polylines)
        m = max(2, self.map_points_each_polyline)
        xs = np.linspace(-self.lane_half_length, self.lane_half_length, m, dtype=np.float32)
        offsets = (np.arange(p, dtype=np.float32) - (p - 1) * 0.5) * self.lane_width
        polylines = np.zeros((p, m, 9), dtype=np.float32)
        mask = np.ones((p, m), dtype=bool)
        for i, y in enumerate(offsets):
            polylines[i, :, 0] = xs
            polylines[i, :, 1] = y
            polylines[i, :, 3] = 1.0
            polylines[i, :, 6] = 1.0
            polylines[i, :, 7] = np.roll(xs, 1)
            polylines[i, 0, 7] = xs[0]
            polylines[i, :, 8] = y
        center = polylines[:, :, 0:3].mean(axis=1)
        return polylines, mask, center

    def _scenario_id(self, real_idx: int, arrays: dict[str, np.ndarray]) -> str:
        if all(k in arrays for k in ("meta_recordingId", "meta_trackId", "meta_frame")):
            return (
                f"{self.dataset_name}_{int(arrays['meta_recordingId'][real_idx])}_"
                f"{int(arrays['meta_trackId'][real_idx])}_{int(arrays['meta_frame'][real_idx])}"
            )
        return f"{self.dataset_name}_{real_idx}"

    def _object_id(self, real_idx: int, arrays: dict[str, np.ndarray]) -> int:
        return int(arrays["meta_trackId"][real_idx]) if "meta_trackId" in arrays else int(real_idx)

    def describe(self, num_samples: int) -> dict[str, Any]:
        return {
            "split": self.split,
            "data_dir": str(self.data_dir),
            "dataset": self.dataset_name,
            "feature_mode": self.feature_mode,
            "num_samples": int(num_samples),
            "num_samples_available": self.n_samples_total,
            "history_len": self.history_len,
            "future_len": self.future_len,
            "raw_max_neighbors": self.raw_max_neighbors,
            "agent_attr_dim": self.agent_attr_dim,
            "map_polylines": self.map_polylines,
            "map_points_each_polyline": self.map_points_each_polyline,
            "neighbor_indices": [int(v) for v in self.nb_feature_indices],
            "neighbor_names": self.nb_feature_names,
            "map_strategy": "straight pseudo-lane polylines in ego-centered coordinates",
            "dimI_mapping": "appended per-timestep agent attributes" if self.extra_dim else "disabled in baseline",
        }

    def channel_stats(self, indices: np.ndarray, n_samples: int = 256) -> dict[str, Any]:
        arrays = self._ensure_open()
        rows = []
        for real_idx in indices[: min(n_samples, len(indices))]:
            mask = np.asarray(arrays["nb_mask"][int(real_idx)], dtype=bool)
            nb = np.asarray(arrays["x_nb"][int(real_idx)], dtype=np.float32)
            if mask.any():
                rows.append(nb[mask][:, self.nb_feature_indices])
        if not rows:
            return {}
        stacked = np.concatenate(rows, axis=0)
        return {
            "samples_inspected": min(n_samples, len(indices)),
            "neighbor_rows": int(stacked.shape[0]),
            "per_channel": {
                name: {
                    "min": float(stacked[:, i].min()),
                    "max": float(stacked[:, i].max()),
                    "mean": float(stacked[:, i].mean()),
                    "nonzero_fraction": float((stacked[:, i] != 0).mean()),
                }
                for i, name in enumerate(self.nb_feature_names)
            },
        }


class NeighFormerMTRDataset:
    """MTR-compatible dataset with optional preprocessed shard reuse."""

    def __init__(
        self,
        data_dir: str | Path,
        indices: np.ndarray,
        dataset_name: str,
        feature_mode: str,
        split: str,
        builder_kwargs: dict[str, Any] | None = None,
        processed_dir: str | Path | None = None,
        reuse_processed: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.sample_indices = np.asarray(indices, dtype=np.int64)
        self.dataset_name = dataset_name
        self.feature_mode = feature_mode
        self.split = split
        self.builder = NeighFormerMTRBuilder(
            self.data_dir, dataset_name, feature_mode, split, **(builder_kwargs or {})
        )
        self.processed_root = processed_root(processed_dir, dataset_name, feature_mode) if processed_dir else None
        self.reuse_processed = bool(reuse_processed and self.processed_root)
        self._manifest = None
        self._shard_cache: dict[int, list[dict[str, Any]]] = {}
        if self.reuse_processed:
            self._manifest = load_manifest(self.processed_root, split)

    def __len__(self) -> int:
        return int(self.sample_indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._manifest is not None:
            return self._load_processed(idx)
        return self.builder.build(int(self.sample_indices[idx]))

    def _load_processed(self, idx: int) -> dict[str, Any]:
        assert self._manifest is not None and self.processed_root is not None
        shard_size = int(self._manifest["shard_size"])
        shard_idx = idx // shard_size
        offset = idx % shard_size
        if shard_idx not in self._shard_cache:
            shard_path = self.processed_root / self._manifest["shards"][shard_idx]
            with open(shard_path, "rb") as f:
                self._shard_cache = {shard_idx: pickle.load(f)}
        return self._shard_cache[shard_idx][offset]

    def collate_batch(self, batch_list: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        key_to_list = {key: [item[key] for item in batch_list] for key in batch_list[0].keys()}
        input_dict = {}
        pad_keys = {
            "obj_trajs",
            "obj_trajs_mask",
            "map_polylines",
            "map_polylines_mask",
            "map_polylines_center",
            "obj_trajs_pos",
            "obj_trajs_last_pos",
            "obj_trajs_future_state",
            "obj_trajs_future_mask",
        }
        concat_np_keys = {"scenario_id", "obj_types", "obj_ids", "center_objects_type", "center_objects_id"}
        for key, val_list in key_to_list.items():
            if key in pad_keys:
                tensors = [torch.from_numpy(x) for x in val_list]
                input_dict[key] = _merge_batch_by_padding_2nd_dim(tensors)
            elif key in concat_np_keys:
                input_dict[key] = np.concatenate(val_list, axis=0)
            else:
                tensors = [torch.from_numpy(x) for x in val_list]
                input_dict[key] = torch.cat(tensors, dim=0)
        batch_sample_count = [len(x["track_index_to_predict"]) for x in batch_list]
        return {"batch_size": len(batch_list), "input_dict": input_dict, "batch_sample_count": batch_sample_count}

    def describe(self) -> dict[str, Any]:
        out = self.builder.describe(len(self))
        out["processed_reuse"] = bool(self._manifest is not None)
        if self._manifest:
            out["processed_manifest"] = self._manifest
        return _as_serializable(out)

    def channel_stats(self, n_samples: int = 256) -> dict[str, Any]:
        return self.builder.channel_stats(self.sample_indices, n_samples=n_samples)


def _merge_batch_by_padding_2nd_dim(tensor_list):
    import torch

    assert tensor_list[0].dim() in {3, 4}
    max_feat0 = max(x.shape[1] for x in tensor_list)
    _, _, *rest = tensor_list[0].shape
    out = tensor_list[0].new_zeros((sum(x.shape[0] for x in tensor_list), max_feat0, *rest))
    offset = 0
    for tensor in tensor_list:
        out[offset: offset + tensor.shape[0], : tensor.shape[1]] = tensor
        offset += tensor.shape[0]
    return out


def processed_root(processed_dir: str | Path | None, dataset_name: str, feature_mode: str) -> Path:
    root = Path(processed_dir or "processed/mtrpp").expanduser()
    return root / dataset_name / feature_mode


def manifest_path(root: Path, split: str) -> Path:
    return root / f"{split}_manifest.json"


def load_manifest(root: Path, split: str) -> dict[str, Any]:
    path = manifest_path(root, split)
    if not path.exists():
        raise FileNotFoundError(f"Processed MTR++ manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_processed_split(
    builder: NeighFormerMTRBuilder,
    indices: np.ndarray,
    root: Path,
    split: str,
    shard_size: int = 4096,
    overwrite: bool = False,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_path(root, split)
    if manifest_file.exists() and not overwrite:
        return load_manifest(root, split)

    shards = []
    total = int(len(indices))
    shard_size = max(1, int(shard_size))
    for shard_idx, start in enumerate(range(0, total, shard_size)):
        end = min(start + shard_size, total)
        records = [builder.build(int(real_idx)) for real_idx in indices[start:end]]
        shard_name = f"{split}_{shard_idx:06d}.pkl"
        with open(root / shard_name, "wb") as f:
            pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
        shards.append(shard_name)
    manifest = {
        "split": split,
        "num_samples": total,
        "shard_size": shard_size,
        "num_shards": len(shards),
        "shards": shards,
        "builder": builder.describe(total),
    }
    manifest_file.write_text(json.dumps(_as_serializable(manifest), indent=2), encoding="utf-8")
    return manifest


def build_intention_points_from_data(
    data_dir: str | Path,
    indices: np.ndarray,
    output_path: str | Path,
    num_modes: int = 6,
    max_samples: int = 50000,
) -> Path:
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    y = np.load(data_dir / "y.npy", mmap_mode="r")
    if len(indices) == 0:
        endpoints = np.array([[20.0, 0.0]], dtype=np.float32)
    else:
        sample = indices[: min(int(max_samples), len(indices))]
        endpoints = np.asarray(y[sample, -1, 0:2], dtype=np.float32)
        finite = np.isfinite(endpoints).all(axis=1)
        endpoints = endpoints[finite]
        if endpoints.size == 0:
            endpoints = np.array([[20.0, 0.0]], dtype=np.float32)
    order = np.argsort(endpoints[:, 0])
    endpoints = endpoints[order]
    quantiles = np.linspace(0.05, 0.95, int(num_modes), dtype=np.float32)
    points = np.zeros((int(num_modes), 2), dtype=np.float32)
    for i, q in enumerate(quantiles):
        pos = min(len(endpoints) - 1, max(0, int(round(q * (len(endpoints) - 1)))))
        local = endpoints[max(0, pos - 128): min(len(endpoints), pos + 129)]
        points[i] = np.median(local, axis=0)
    if not np.isfinite(points).all() or math.isclose(float(np.linalg.norm(points)), 0.0):
        xs = np.linspace(20.0, 120.0, int(num_modes), dtype=np.float32)
        points = np.stack([xs, np.zeros_like(xs)], axis=-1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {OBJECT_TYPE_VEHICLE: points}
    with open(output_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path
