"""HiVT TemporalData conversion for persistent multi-agent arrays."""

from __future__ import annotations

from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Dataset

from adapters.hivt.dataset import _heading_from_velocity, _step_displacements
from adapters.multiagent_common import MultiAgentArrays, agent_positions, scene_agent_indices, scored_local_indices


class MultiAgentHiVTDataset(Dataset):
    """HiVT-compatible dataset backed by full multi-agent scene arrays."""

    def __init__(self, data_dir: str | Path, dataset_name: str, split: str, indices: np.ndarray | None = None) -> None:
        super().__init__(None, None, None)
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.split = split
        self.store = MultiAgentArrays(self.data_dir)
        self.scene_indices = np.arange(self.store.num_scenes, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        self.history_len = self.store.history_len
        self.future_len = self.store.future_len
        self.num_steps = self.history_len + self.future_len
        self.node_dim = 6
        self.edge_dim = 2

    def len(self) -> int:
        return int(self.scene_indices.size)

    def get(self, idx: int):
        from utils import TemporalData

        arrays = self.store.open()
        scene_idx = int(self.scene_indices[idx])
        agent_ids = np.asarray(arrays["agent_ids"][scene_idx])
        x = np.asarray(arrays["x_agents"][scene_idx], dtype=np.float32)
        obs = np.asarray(arrays["obs_valid"][scene_idx], dtype=bool)
        y = np.asarray(arrays["y_agents"][scene_idx], dtype=np.float32)
        fut_valid = np.asarray(arrays["future_valid"][scene_idx], dtype=bool)
        scored = np.asarray(arrays["scored_agent_mask"][scene_idx], dtype=bool)
        keep = scene_agent_indices(agent_ids, obs)
        n = int(keep.size)
        th, tf = self.history_len, self.future_len

        positions = agent_positions(x, y, keep)
        padding_mask = np.ones((n, th + tf), dtype=bool)
        padding_mask[:, :th] = ~obs[keep]
        padding_mask[:, th:] = ~fut_valid[keep]
        features = np.zeros((n, th, self.node_dim), dtype=np.float32)
        for i, src in enumerate(keep):
            features[i, :, 0:2] = _step_displacements(x[src, :, 0:2], obs[src])
            features[i, :, 2:6] = x[src, :, 2:6]
            features[i, ~obs[src]] = 0.0
        bos_mask = np.zeros((n, th), dtype=bool)
        bos_mask[:, 0] = ~padding_mask[:, 0]
        bos_mask[:, 1:th] = padding_mask[:, : th - 1] & ~padding_mask[:, 1:th]
        rotate_angles = _heading_from_velocity(features[:, -1, 2:4])
        edge_index = (
            torch.tensor(list(permutations(range(n), 2)), dtype=torch.long).t().contiguous()
            if n > 1
            else torch.empty(2, 0, dtype=torch.long)
        )
        lane = self._pseudo_lane_features(positions[:, th - 1])
        scored_local = scored_local_indices(scored, keep)
        return TemporalData(
            x=torch.from_numpy(features),
            positions=torch.from_numpy(positions),
            edge_index=edge_index,
            y=torch.from_numpy(positions[:, th:] - positions[:, th - 1 : th]),
            num_nodes=n,
            padding_mask=torch.from_numpy(padding_mask),
            bos_mask=torch.from_numpy(bos_mask),
            rotate_angles=torch.from_numpy(rotate_angles),
            lane_vectors=lane["lane_vectors"],
            is_intersections=lane["is_intersections"],
            turn_directions=lane["turn_directions"],
            traffic_controls=lane["traffic_controls"],
            lane_actor_index=lane["lane_actor_index"],
            lane_actor_vectors=lane["lane_actor_vectors"],
            seq_id=scene_idx,
            agent_index=int(scored_local[0]) if scored_local.size else 0,
            av_index=0,
            scored_agent_index=torch.from_numpy(scored_local),
            sample_index=torch.tensor([scene_idx], dtype=torch.long),
            recording_id=torch.tensor([int(arrays["recordingId"][scene_idx])], dtype=torch.long),
            track_id=torch.tensor([int(arrays["ego_trackId"][scene_idx])], dtype=torch.long),
            frame_id=torch.tensor([int(arrays["t0_frame"][scene_idx])], dtype=torch.long),
        )

    def _pseudo_lane_features(self, node_positions: np.ndarray) -> dict[str, torch.Tensor]:
        xs = np.linspace(-120.0, 120.0, 25, dtype=np.float32)
        lane_pos = np.stack([xs[:-1], np.zeros_like(xs[:-1])], axis=-1)
        lane_vec2 = np.stack([np.diff(xs), np.zeros(xs.shape[0] - 1, dtype=np.float32)], axis=-1)
        lane_vectors = np.zeros((lane_vec2.shape[0], self.node_dim), dtype=np.float32)
        lane_vectors[:, :2] = lane_vec2
        src, dst, vecs = [], [], []
        for lane_idx, pos in enumerate(lane_pos):
            rel = pos.reshape(1, 2) - node_positions
            for node_idx, vec in enumerate(rel):
                if float(np.linalg.norm(vec)) < 120.0:
                    src.append(lane_idx)
                    dst.append(node_idx)
                    vecs.append(vec.astype(np.float32))
        if not src:
            src, dst, vecs = [0], [0], [lane_pos[0] - node_positions[0]]
        return {
            "lane_vectors": torch.from_numpy(lane_vectors),
            "is_intersections": torch.zeros(lane_vectors.shape[0], dtype=torch.uint8),
            "turn_directions": torch.zeros(lane_vectors.shape[0], dtype=torch.uint8),
            "traffic_controls": torch.zeros(lane_vectors.shape[0], dtype=torch.uint8),
            "lane_actor_index": torch.tensor([src, dst], dtype=torch.long),
            "lane_actor_vectors": torch.as_tensor(np.asarray(vecs, dtype=np.float32)),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "split": self.split,
            "data_dir": str(self.data_dir),
            "num_scenes": self.len(),
            "history_len": self.history_len,
            "future_len": self.future_len,
            "node_dim": self.node_dim,
            "source": "multiagent",
        }

    def channel_stats(self, n_samples: int = 256) -> dict[str, Any]:
        arrays = self.store.open()
        n = min(int(n_samples), self.len())
        retained = []
        scored = []
        for j in range(n):
            scene_idx = int(self.scene_indices[j])
            retained.append(int(np.count_nonzero(arrays["agent_ids"][scene_idx] >= 0)))
            scored.append(int(np.count_nonzero(arrays["scored_agent_mask"][scene_idx])))
        return {
            "samples_inspected": n,
            "retained_agents_mean": float(np.mean(retained)) if retained else 0.0,
            "scored_agents_mean": float(np.mean(scored)) if scored else 0.0,
        }
