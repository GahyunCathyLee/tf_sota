"""NeighFormer npy -> QCNet HeteroData conversion.

The official QCNet model consumes Argoverse2-like ``HeteroData``. highD/exiD
do not provide lane maps, so each sample receives one simple pseudo-lane
polygon centered on the ego vehicle. This keeps QCNet's map attention path
active while all real trajectory information comes from the NeighFormer arrays.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Dataset, HeteroData

from adapters.common import feature_mode_indices, feature_mode_names


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return -math.pi + (angle + math.pi) % (2 * math.pi)


def _heading_from_velocity(velocity: torch.Tensor) -> torch.Tensor:
    return torch.atan2(velocity[..., 1], velocity[..., 0])


class NeighFormerQCNetDataset(Dataset):
    """QCNet-compatible highD/exiD dataset backed by NeighFormer mmap npy files."""

    def __init__(
        self,
        data_dir: str | Path,
        indices: np.ndarray,
        dataset_name: str,
        feature_mode: str,
        split: str,
        lane_half_length: float = 120.0,
    ) -> None:
        super().__init__(None, None, None)
        self.data_dir = Path(data_dir)
        self.sample_indices = np.asarray(indices, dtype=np.int64)
        self.dataset_name = dataset_name
        self.feature_mode = feature_mode
        self.split = split
        self.lane_half_length = float(lane_half_length)
        self.nb_feature_indices = np.asarray(feature_mode_indices(feature_mode), dtype=np.int64)
        self.nb_feature_names = feature_mode_names(feature_mode)
        self._arrays: dict[str, np.ndarray] | None = None

        x_ego = np.load(self.data_dir / "x_ego.npy", mmap_mode="r")
        x_nb = np.load(self.data_dir / "x_nb.npy", mmap_mode="r")
        y = np.load(self.data_dir / "y.npy", mmap_mode="r")
        self.n_samples_total = int(x_ego.shape[0])
        self.num_historical_steps = int(x_ego.shape[1])
        self.num_future_steps = int(y.shape[1])
        self.num_steps = self.num_historical_steps + self.num_future_steps
        self.max_neighbors = int(x_nb.shape[2])
        self.has_y_vel = (self.data_dir / "y_vel.npy").exists()

        if int(x_ego.shape[2]) != 6:
            raise ValueError(f"Expected x_ego[..., 6], got {x_ego.shape}")
        if int(x_nb.shape[3]) < int(self.nb_feature_indices.max()) + 1:
            raise ValueError(
                f"x_nb has {x_nb.shape[3]} channels; {feature_mode} needs "
                f"index {int(self.nb_feature_indices.max())}"
            )

    def _ensure_open(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            arrays = {
                "x_ego": np.load(self.data_dir / "x_ego.npy", mmap_mode="r"),
                "x_nb": np.load(self.data_dir / "x_nb.npy", mmap_mode="r"),
                "nb_mask": np.load(self.data_dir / "nb_mask.npy", mmap_mode="r"),
                "y": np.load(self.data_dir / "y.npy", mmap_mode="r"),
            }
            if self.has_y_vel:
                arrays["y_vel"] = np.load(self.data_dir / "y_vel.npy", mmap_mode="r")
            self._arrays = arrays
        return self._arrays

    def len(self) -> int:
        return int(self.sample_indices.shape[0])

    def get(self, idx: int) -> HeteroData:
        arrays = self._ensure_open()
        real_idx = int(self.sample_indices[idx])
        th = self.num_historical_steps
        tf = self.num_future_steps

        ego = np.array(arrays["x_ego"][real_idx], dtype=np.float32, copy=True)
        nb = np.array(arrays["x_nb"][real_idx], dtype=np.float32, copy=True)
        mask = np.array(arrays["nb_mask"][real_idx], dtype=bool, copy=True)
        future_pos = np.array(arrays["y"][real_idx], dtype=np.float32, copy=True)

        slots = np.flatnonzero(mask.any(axis=0))
        num_agents = 1 + int(slots.size)

        position = torch.zeros(num_agents, th + tf, 2, dtype=torch.float32)
        velocity = torch.zeros(num_agents, th + tf, 2, dtype=torch.float32)
        valid_mask = torch.zeros(num_agents, th + tf, dtype=torch.bool)
        predict_mask = torch.zeros(num_agents, th + tf, dtype=torch.bool)
        attrs = None
        if self.feature_mode == "dimI":
            attrs = torch.full((num_agents, th, 2), -1.0, dtype=torch.float32)

        position[0, :th] = torch.from_numpy(ego[:, 0:2])
        velocity[0, :th] = torch.from_numpy(ego[:, 2:4])
        position[0, th:] = torch.from_numpy(future_pos)
        if self.has_y_vel:
            velocity[0, th:] = torch.from_numpy(
                np.array(arrays["y_vel"][real_idx], dtype=np.float32, copy=True)
            )
        else:
            deltas = torch.diff(position[0, th - 1:], dim=0)
            velocity[0, th:] = deltas
        valid_mask[0] = True
        predict_mask[0, th:] = True

        for agent_offset, slot in enumerate(slots, start=1):
            slot_mask = mask[:, slot]
            nb_hist = nb[:, slot]
            position[agent_offset, :th] = torch.from_numpy(ego[:, 0:2] + nb_hist[:, 0:2])
            velocity[agent_offset, :th] = torch.from_numpy(ego[:, 2:4] + nb_hist[:, 2:4])
            valid_mask[agent_offset, :th] = torch.from_numpy(slot_mask)
            if attrs is not None:
                attrs[agent_offset] = torch.from_numpy(nb_hist[:, [8, 9]])

        heading = _heading_from_velocity(velocity)
        target = torch.zeros(num_agents, tf, 4, dtype=torch.float32)
        origin = position[:, th - 1]
        theta = heading[:, th - 1]
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(num_agents, 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        target[..., :2] = torch.bmm(position[:, th:, :2] - origin[:, :2].unsqueeze(1), rot_mat)
        target[..., 3] = _wrap_angle(heading[:, th:] - theta.unsqueeze(-1))

        data = HeteroData()
        data["scenario_id"] = f"{self.dataset_name}_{real_idx}"
        data["sample_index"] = torch.tensor([real_idx], dtype=torch.long)
        data["agent"]["num_nodes"] = num_agents
        data["agent"]["av_index"] = torch.tensor([0], dtype=torch.long)
        data["agent"]["valid_mask"] = valid_mask
        data["agent"]["predict_mask"] = predict_mask
        data["agent"]["id"] = [f"ego_{real_idx}", *[f"nb{int(s)}_{real_idx}" for s in slots]]
        data["agent"]["type"] = torch.zeros(num_agents, dtype=torch.uint8)
        data["agent"]["category"] = torch.zeros(num_agents, dtype=torch.uint8)
        data["agent"]["category"][0] = 3
        data["agent"]["position"] = position
        data["agent"]["heading"] = heading
        data["agent"]["velocity"] = velocity
        data["agent"]["target"] = target
        if attrs is not None:
            data["agent"]["attrs"] = attrs

        self._add_pseudo_map(data)
        return data

    def _add_pseudo_map(self, data: HeteroData) -> None:
        l = self.lane_half_length
        data["map_polygon"]["num_nodes"] = 1
        data["map_polygon"]["position"] = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        data["map_polygon"]["orientation"] = torch.tensor([0.0], dtype=torch.float32)
        data["map_polygon"]["type"] = torch.tensor([0], dtype=torch.uint8)
        data["map_polygon"]["is_intersection"] = torch.tensor([1], dtype=torch.uint8)

        point_pos = torch.tensor([[-l, 0.0], [0.0, 0.0], [l, 0.0]], dtype=torch.float32)
        data["map_point"]["num_nodes"] = int(point_pos.shape[0])
        data["map_point"]["position"] = point_pos
        data["map_point"]["orientation"] = torch.zeros(point_pos.shape[0], dtype=torch.float32)
        data["map_point"]["magnitude"] = torch.full((point_pos.shape[0],), l, dtype=torch.float32)
        data["map_point"]["type"] = torch.full((point_pos.shape[0],), 16, dtype=torch.uint8)
        data["map_point"]["side"] = torch.full((point_pos.shape[0],), 2, dtype=torch.uint8)
        data["map_point", "to", "map_polygon"]["edge_index"] = torch.tensor(
            [[0, 1, 2], [0, 0, 0]], dtype=torch.long
        )
        data["map_polygon", "to", "map_polygon"]["edge_index"] = torch.empty(2, 0, dtype=torch.long)
        data["map_polygon", "to", "map_polygon"]["type"] = torch.empty(0, dtype=torch.uint8)

    def channel_stats(self, n_samples: int = 256) -> dict[str, Any]:
        n = min(n_samples, self.len())
        rows: list[np.ndarray] = []
        arrays = self._ensure_open()
        for j in range(n):
            real_idx = int(self.sample_indices[j])
            mask = np.asarray(arrays["nb_mask"][real_idx], dtype=bool)
            nb = np.asarray(arrays["x_nb"][real_idx], dtype=np.float32)
            if mask.any():
                rows.append(nb[mask][:, self.nb_feature_indices])
        if not rows:
            return {}
        stacked = np.concatenate(rows, axis=0)
        return {
            "samples_inspected": n,
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

    def describe(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "data_dir": str(self.data_dir),
            "dataset": self.dataset_name,
            "feature_mode": self.feature_mode,
            "num_samples": self.len(),
            "num_samples_available": self.n_samples_total,
            "history_len": self.num_historical_steps,
            "future_len": self.num_future_steps,
            "max_neighbors": self.max_neighbors,
            "neighbor_indices": [int(v) for v in self.nb_feature_indices],
            "neighbor_names": self.nb_feature_names,
            "pseudo_map": True,
            "agent_attr_dim": 2 if self.feature_mode == "dimI" else 0,
        }
