"""QCNet HeteroData conversion for persistent multi-agent arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Dataset, HeteroData

from adapters.multiagent_common import (
    MultiAgentArrays,
    agent_positions,
    agent_velocities,
    scene_agent_indices,
    scored_local_indices,
)
from adapters.qcnet.dataset import _heading_from_velocity, _wrap_angle
from data.multiagent.interaction import InteractionConfig, build_pair_features


INTERACTION_EDGE_TYPE = ("agent", "interaction_importance", "agent")


class MultiAgentQCNetDataset(Dataset):
    """QCNet-compatible dataset backed by full multi-agent scene arrays."""

    def __init__(
        self,
        data_dir: str | Path,
        dataset_name: str,
        split: str,
        indices: np.ndarray | None = None,
        use_interaction_importance: bool = False,
        normalize_i: bool = False,
    ) -> None:
        super().__init__(None, None, None)
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.split = split
        self.use_interaction_importance = bool(use_interaction_importance)
        self.normalize_i = bool(normalize_i)
        if self.normalize_i:
            raise ValueError("QCNet edge-level I normalization is not implemented; keep normalize_i=false.")
        self.store = MultiAgentArrays(self.data_dir)
        self.scene_indices = np.arange(self.store.num_scenes, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        self.num_historical_steps = self.store.history_len
        self.num_future_steps = self.store.future_len
        self.num_steps = self.num_historical_steps + self.num_future_steps

    def len(self) -> int:
        return int(self.scene_indices.size)

    def get(self, idx: int) -> HeteroData:
        arrays = self.store.open()
        scene_idx = int(self.scene_indices[idx])
        agent_ids = np.asarray(arrays["agent_ids"][scene_idx])
        x = np.asarray(arrays["x_agents"][scene_idx], dtype=np.float32)
        obs = np.asarray(arrays["obs_valid"][scene_idx], dtype=bool)
        y = np.asarray(arrays["y_agents"][scene_idx], dtype=np.float32)
        fut_valid = np.asarray(arrays["future_valid"][scene_idx], dtype=bool)
        scored = np.asarray(arrays["scored_agent_mask"][scene_idx], dtype=bool)
        keep = scene_agent_indices(agent_ids, obs)
        th, tf = self.num_historical_steps, self.num_future_steps
        n = int(keep.size)

        position = torch.from_numpy(agent_positions(x, y, keep))
        velocity = torch.from_numpy(agent_velocities(x, y, keep))
        valid_mask = torch.zeros(n, th + tf, dtype=torch.bool)
        valid_mask[:, :th] = torch.from_numpy(obs[keep])
        valid_mask[:, th:] = torch.from_numpy(fut_valid[keep])
        scored_local = scored_local_indices(scored, keep)
        predict_mask = torch.zeros(n, th + tf, dtype=torch.bool)
        for local_i in scored_local.tolist():
            predict_mask[local_i, th:] = valid_mask[local_i, th:]

        heading = _heading_from_velocity(velocity)
        target = torch.zeros(n, tf, 4, dtype=torch.float32)
        origin = position[:, th - 1]
        theta = heading[:, th - 1]
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(n, 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        target[..., :2] = torch.bmm(position[:, th:, :2] - origin[:, :2].unsqueeze(1), rot_mat)
        target[..., 3] = _wrap_angle(heading[:, th:] - theta.unsqueeze(-1))

        data = HeteroData()
        data["scenario_id"] = f"{self.dataset_name}_{self.split}_{scene_idx}"
        data["sample_index"] = torch.tensor([scene_idx], dtype=torch.long)
        data["agent"]["num_nodes"] = n
        data["agent"]["av_index"] = torch.tensor([0], dtype=torch.long)
        data["agent"]["valid_mask"] = valid_mask
        data["agent"]["predict_mask"] = predict_mask
        data["agent"]["id"] = [str(int(v)) for v in agent_ids[keep].tolist()]
        data["agent"]["type"] = torch.zeros(n, dtype=torch.uint8)
        data["agent"]["category"] = torch.zeros(n, dtype=torch.uint8)
        if scored_local.size:
            data["agent"]["category"][torch.from_numpy(scored_local)] = 3
        data["agent"]["position"] = position
        data["agent"]["heading"] = heading
        data["agent"]["velocity"] = velocity
        data["agent"]["target"] = target
        data["agent"]["attrs"] = torch.zeros(n, th, 2, dtype=torch.float32)
        data["agent"]["scored_index"] = torch.from_numpy(scored_local)
        if self.use_interaction_importance:
            self._add_interaction_importance(data, arrays, scene_idx, keep)
        self._add_pseudo_map(data)
        return data

    def _add_interaction_importance(
        self,
        data: HeteroData,
        arrays: dict[str, np.ndarray],
        scene_idx: int,
        keep: np.ndarray,
    ) -> None:
        """Attach directed pairwise I as sparse scene-local edges.

        ``build_pair_features`` indexes pair features as
        ``pair_features[target_i, source_j, timestep, 9]``.  The edge store
        below uses PyG's normal ``edge_index=[source_j, target_i]`` convention,
        so the model can later map QCNet social edges via ``I = pair_I[dst, src, t]``.
        """
        x = np.asarray(arrays["x_agents"][scene_idx], dtype=np.float32)[keep]
        obs = np.asarray(arrays["obs_valid"][scene_idx], dtype=bool)[keep]
        pair_features, pair_valid = build_pair_features(
            x,
            obs,
            np.asarray(arrays["agent_length"][scene_idx], dtype=np.float32)[keep],
            np.asarray(arrays["agent_width"][scene_idx], dtype=np.float32)[keep],
            np.asarray(arrays["agent_type"][scene_idx], dtype=np.int8)[keep],
            np.asarray(arrays["lane_id"][scene_idx], dtype=np.int32)[keep],
            np.asarray(arrays["lane_offset"][scene_idx], dtype=np.float32)[keep],
            np.asarray(arrays["lane_width"][scene_idx], dtype=np.float32)[keep],
            lane_level=np.asarray(arrays["lane_level"][scene_idx], dtype=np.int16)[keep],
            heading=np.asarray(arrays["heading"][scene_idx], dtype=np.float32)[keep],
            lateral_velocity=np.asarray(arrays["lateral_velocity"][scene_idx], dtype=np.float32)[keep],
            config=InteractionConfig(dataset=self.dataset_name, apply_topn=False),
        )
        if pair_features.shape[:3] != pair_valid.shape:
            raise ValueError(
                f"interaction shape mismatch: pair_features={pair_features.shape}, pair_valid={pair_valid.shape}"
            )
        if not np.isfinite(pair_features[..., 9]).all():
            raise ValueError(f"NaN/Inf found in interaction importance for scene {scene_idx}")

        target, source, time = np.nonzero(pair_valid)
        importance = pair_features[target, source, time, 9].astype(np.float32)
        edge_index = np.stack([source, target], axis=0).astype(np.int64)

        data[INTERACTION_EDGE_TYPE]["edge_index"] = torch.from_numpy(edge_index)
        data[INTERACTION_EDGE_TYPE]["time"] = torch.from_numpy(time.astype(np.int64))
        data[INTERACTION_EDGE_TYPE]["importance"] = torch.from_numpy(importance)
        data[INTERACTION_EDGE_TYPE]["pair_valid"] = torch.ones(importance.shape[0], dtype=torch.bool)

    def _add_pseudo_map(self, data: HeteroData) -> None:
        l = 120.0
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
        data["map_point", "to", "map_polygon"]["edge_index"] = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
        data["map_polygon", "to", "map_polygon"]["edge_index"] = torch.empty(2, 0, dtype=torch.long)
        data["map_polygon", "to", "map_polygon"]["type"] = torch.empty(0, dtype=torch.uint8)

    def describe(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "split": self.split,
            "data_dir": str(self.data_dir),
            "num_scenes": self.len(),
            "history_len": self.num_historical_steps,
            "future_len": self.num_future_steps,
            "source": "multiagent",
            "use_interaction_importance": self.use_interaction_importance,
            "normalize_i": self.normalize_i,
        }

    def channel_stats(self, n_samples: int = 256) -> dict[str, Any]:
        arrays = self.store.open()
        n = min(int(n_samples), self.len())
        retained = []
        scored = []
        i_rows = []
        for j in range(n):
            scene_idx = int(self.scene_indices[j])
            retained.append(int(np.count_nonzero(arrays["agent_ids"][scene_idx] >= 0)))
            scored.append(int(np.count_nonzero(arrays["scored_agent_mask"][scene_idx])))
            if self.use_interaction_importance:
                keep = scene_agent_indices(np.asarray(arrays["agent_ids"][scene_idx]), np.asarray(arrays["obs_valid"][scene_idx], dtype=bool))
                pair, valid = build_pair_features(
                    np.asarray(arrays["x_agents"][scene_idx], dtype=np.float32)[keep],
                    np.asarray(arrays["obs_valid"][scene_idx], dtype=bool)[keep],
                    np.asarray(arrays["agent_length"][scene_idx], dtype=np.float32)[keep],
                    np.asarray(arrays["agent_width"][scene_idx], dtype=np.float32)[keep],
                    np.asarray(arrays["agent_type"][scene_idx], dtype=np.int8)[keep],
                    np.asarray(arrays["lane_id"][scene_idx], dtype=np.int32)[keep],
                    np.asarray(arrays["lane_offset"][scene_idx], dtype=np.float32)[keep],
                    np.asarray(arrays["lane_width"][scene_idx], dtype=np.float32)[keep],
                    lane_level=np.asarray(arrays["lane_level"][scene_idx], dtype=np.int16)[keep],
                    heading=np.asarray(arrays["heading"][scene_idx], dtype=np.float32)[keep],
                    lateral_velocity=np.asarray(arrays["lateral_velocity"][scene_idx], dtype=np.float32)[keep],
                    config=InteractionConfig(dataset=self.dataset_name, apply_topn=False),
                )
                if valid.any():
                    i_rows.append(pair[..., 9][valid])
        i_stats = None
        if i_rows:
            vals = np.concatenate(i_rows).astype(np.float64)
            i_stats = {
                "count": int(vals.size),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "p50": float(np.percentile(vals, 50)),
                "p75": float(np.percentile(vals, 75)),
                "p85": float(np.percentile(vals, 85)),
                "p90": float(np.percentile(vals, 90)),
                "p95": float(np.percentile(vals, 95)),
                "p99": float(np.percentile(vals, 99)),
            }
        return {
            "samples_inspected": n,
            "retained_agents_mean": float(np.mean(retained)) if retained else 0.0,
            "scored_agents_mean": float(np.mean(scored)) if scored else 0.0,
            "interaction_importance": i_stats,
        }
