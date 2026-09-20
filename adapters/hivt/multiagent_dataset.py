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
from data.multiagent.interaction import InteractionConfig, build_pair_features


class MultiAgentHiVTDataset(Dataset):
    """HiVT-compatible dataset backed by full multi-agent scene arrays."""

    def __init__(
        self,
        data_dir: str | Path,
        dataset_name: str,
        split: str,
        indices: np.ndarray | None = None,
        use_importance: bool = False,
    ) -> None:
        super().__init__(None, None, None)
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.split = split
        self.use_importance = bool(use_importance)
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
        importance = self._edge_importance(arrays, scene_idx, keep, edge_index) if self.use_importance else None
        lane = self._pseudo_lane_features(positions[:, th - 1])
        scored_local = scored_local_indices(scored, keep)
        data = TemporalData(
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
        if importance is not None:
            data.edge_importance = importance["importance"]
            data.edge_pair_valid = importance["pair_valid"]
            data.agent_original_idx = torch.from_numpy(keep.astype(np.int64))
        return data

    def _edge_importance(
        self,
        arrays: dict[str, np.ndarray],
        scene_idx: int,
        keep: np.ndarray,
        edge_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return ``I`` aligned with ``edge_index=[source_local, target_local]``.

        ``build_pair_features`` uses original scene-agent axes and stores
        directed importance as ``pair_features[target, source, t, 9]``.
        """
        th = self.history_len
        edge_count = int(edge_index.size(1))
        importance = np.zeros((edge_count, th), dtype=np.float32)
        pair_valid_edge = np.zeros((edge_count, th), dtype=bool)
        if edge_count == 0:
            return {
                "importance": torch.from_numpy(importance),
                "pair_valid": torch.from_numpy(pair_valid_edge),
            }

        pair_features, pair_valid = build_pair_features(
            np.asarray(arrays["x_agents"][scene_idx], dtype=np.float32),
            np.asarray(arrays["obs_valid"][scene_idx], dtype=bool),
            np.asarray(arrays["agent_length"][scene_idx], dtype=np.float32),
            np.asarray(arrays["agent_width"][scene_idx], dtype=np.float32),
            np.asarray(arrays["agent_type"][scene_idx], dtype=np.int16),
            np.asarray(arrays["lane_id"][scene_idx], dtype=np.int32),
            np.asarray(arrays["lane_offset"][scene_idx], dtype=np.float32),
            np.asarray(arrays["lane_width"][scene_idx], dtype=np.float32),
            lane_level=np.asarray(arrays["lane_level"][scene_idx], dtype=np.int16) if "lane_level" in arrays else None,
            heading=np.asarray(arrays["heading"][scene_idx], dtype=np.float32) if "heading" in arrays else None,
            lateral_velocity=(
                np.asarray(arrays["lateral_velocity"][scene_idx], dtype=np.float32)
                if "lateral_velocity" in arrays
                else None
            ),
            config=InteractionConfig(dataset=self.dataset_name, apply_topn=False),
        )
        if pair_features.shape[:3] != pair_valid.shape or pair_features.shape[2] != th:
            raise ValueError(
                f"interaction shape mismatch: pair_features={pair_features.shape}, pair_valid={pair_valid.shape}"
            )
        if not np.isfinite(pair_features[..., 9]).all():
            raise ValueError(f"NaN/Inf found in interaction importance for scene {scene_idx}")

        src_local = edge_index[0].numpy()
        dst_local = edge_index[1].numpy()
        src_orig = keep[src_local]
        dst_orig = keep[dst_local]
        importance[:, :] = pair_features[dst_orig, src_orig, :, 9]
        pair_valid_edge[:, :] = pair_valid[dst_orig, src_orig, :]
        return {
            "importance": torch.from_numpy(importance),
            "pair_valid": torch.from_numpy(pair_valid_edge),
        }

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
            "edge_dim": self.edge_dim,
            "use_importance": self.use_importance,
            "source": "multiagent",
        }

    def channel_stats(self, n_samples: int = 256) -> dict[str, Any]:
        arrays = self.store.open()
        n = min(int(n_samples), self.len())
        retained = []
        scored = []
        importance_rows = []
        for j in range(n):
            scene_idx = int(self.scene_indices[j])
            agent_ids = np.asarray(arrays["agent_ids"][scene_idx])
            obs = np.asarray(arrays["obs_valid"][scene_idx], dtype=bool)
            keep = scene_agent_indices(agent_ids, obs)
            retained.append(int(keep.size))
            scored.append(int(np.count_nonzero(arrays["scored_agent_mask"][scene_idx])))
            if self.use_importance and keep.size > 1:
                edge_index = torch.tensor(list(permutations(range(int(keep.size)), 2)), dtype=torch.long).t().contiguous()
                imp = self._edge_importance(arrays, scene_idx, keep, edge_index)
                valid_imp = imp["importance"][imp["pair_valid"]]
                if valid_imp.numel():
                    importance_rows.append(valid_imp.numpy())
        i_stats = None
        if importance_rows:
            values = np.concatenate(importance_rows).astype(np.float64)
            i_stats = {
                "count": int(values.size),
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std()),
            }
        return {
            "samples_inspected": n,
            "retained_agents_mean": float(np.mean(retained)) if retained else 0.0,
            "scored_agents_mean": float(np.mean(scored)) if scored else 0.0,
            "importance": i_stats,
        }
