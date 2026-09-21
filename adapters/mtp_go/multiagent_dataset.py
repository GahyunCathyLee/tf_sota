"""MTP-GO graph conversion for persistent multi-agent scene arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Dataset

from adapters.mtp_go.dataset import MTPGoData, TARGET_CHANNELS, build_edges
from adapters.multiagent_common import MultiAgentArrays, meta_dict, scene_agent_indices, scored_local_indices
from data.multiagent.interaction import InteractionConfig, build_pair_features

NATIVE_NODE_FEATURES = 6


class MultiAgentMTPGoDataset(Dataset):
    """MTP-GO-compatible dataset backed by ``data/*_multiagent/*_full`` arrays."""

    def __init__(
        self,
        data_dir: str | Path,
        dataset_name: str,
        feature_mode: str,
        split: str,
        indices: np.ndarray | None = None,
        use_importance: bool = False,
    ) -> None:
        super().__init__(None, None, None)
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.feature_mode = feature_mode
        self.use_importance = bool(use_importance)
        self.split = split
        self.store = MultiAgentArrays(self.data_dir)
        self.scene_indices = (
            np.arange(self.store.num_scenes, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self.history_len = self.store.history_len
        self.future_len = self.store.future_len
        self.n_node_features = NATIVE_NODE_FEATURES
        self.edge_feature_dim = 2 if self.use_importance else 1

    def len(self) -> int:
        return int(self.scene_indices.size)

    def get(self, idx: int):
        arrays = self.store.open()
        scene_idx = int(self.scene_indices[idx])
        agent_ids = np.asarray(arrays["agent_ids"][scene_idx])
        x_all = np.asarray(arrays["x_agents"][scene_idx], dtype=np.float32)
        obs = np.asarray(arrays["obs_valid"][scene_idx], dtype=bool)
        y_all = np.asarray(arrays["y_agents"][scene_idx], dtype=np.float32)
        fut_valid = np.asarray(arrays["future_valid"][scene_idx], dtype=bool)
        scored = np.asarray(arrays["scored_agent_mask"][scene_idx], dtype=bool)
        keep = scene_agent_indices(agent_ids, obs)
        if keep.size == 0:
            keep = np.asarray([0], dtype=np.int64)
        scored_local = scored_local_indices(scored, keep)
        if scored_local.size == 0:
            scored_local = np.asarray([0], dtype=np.int64)

        n = int(keep.size)
        th, tf = self.history_len, self.future_len
        x_src = x_all[keep]
        obs_src = obs[keep]
        y_src = y_all[keep]
        fut_src = fut_valid[keep]

        x = np.zeros((n, th, NATIVE_NODE_FEATURES), dtype=np.float32)
        x[:, :, :NATIVE_NODE_FEATURES] = x_src[:, :, :NATIVE_NODE_FEATURES]
        x[~obs_src] = 0.0

        pair = self._pair_importance(arrays, scene_idx) if self.use_importance else None
        hist_ei: list[torch.Tensor] = []
        hist_ef: list[torch.Tensor] = []
        for t in range(th):
            ei, ef = build_edges(x_src[:, t, 0:2], obs_src[:, t])
            if self.use_importance:
                ef = self._append_edge_importance(ef, ei, pair, keep, t)
            hist_ei.append(ei)
            hist_ef.append(ef)
        fut_ei = [hist_ei[-1]] * tf
        fut_ef = [hist_ef[-1]] * tf

        y = np.zeros((n, tf, TARGET_CHANNELS), dtype=np.float32)
        n_target_channels = min(TARGET_CHANNELS, int(y_src.shape[-1]))
        y[:, :, :n_target_channels] = y_src[:, :, :n_target_channels]
        real_mask = np.zeros((n, tf, TARGET_CHANNELS), dtype=bool)
        real_mask[scored_local, :, :n_target_channels] = fut_src[scored_local, :, None]

        v_type = torch.zeros(n, 2, dtype=torch.float32)
        v_type[:, 0] = 1.0
        dim = np.zeros((n, 2), dtype=np.float32)
        if "agent_length" in arrays:
            dim[:, 0] = np.asarray(arrays["agent_length"][scene_idx][keep], dtype=np.float32)
        if "agent_width" in arrays:
            dim[:, 1] = np.asarray(arrays["agent_width"][scene_idx][keep], dtype=np.float32)

        data = MTPGoData(
            x=torch.from_numpy(x),
            edge_index=hist_ei,
            edge_features=hist_ef,
            y=torch.from_numpy(y),
            tar_edge_index=fut_ei,
            tar_edge_features=fut_ef,
            tar_real_mask=torch.from_numpy(real_mask),
            cf=torch.full((n,), 3, dtype=torch.long),
            dim=torch.from_numpy(dim),
            v_type=v_type,
            sample_index=torch.tensor([scene_idx], dtype=torch.long),
            scored_agent_index=torch.from_numpy(scored_local.astype(np.int64)),
            agent_original_idx=torch.from_numpy(keep.astype(np.int64)),
        )
        for key, value in meta_dict(arrays, scene_idx).items():
            setattr(data, key, value)
        return data

    def _pair_importance(
        self,
        arrays: dict[str, np.ndarray],
        scene_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
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
        if pair_features.shape[:3] != pair_valid.shape:
            raise ValueError(
                f"interaction shape mismatch: pair_features={pair_features.shape}, pair_valid={pair_valid.shape}"
            )
        if not np.isfinite(pair_features[..., 9]).all():
            raise ValueError(f"NaN/Inf found in interaction importance for scene {scene_idx}")
        return pair_features, pair_valid

    def _append_edge_importance(
        self,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        pair: tuple[np.ndarray, np.ndarray] | None,
        keep: np.ndarray,
        timestep: int,
    ) -> torch.Tensor:
        if pair is None:
            return edge_attr
        pair_features, pair_valid = pair
        src_local = edge_index[0].numpy()
        dst_local = edge_index[1].numpy()
        src_orig = keep[src_local]
        dst_orig = keep[dst_local]
        imp = np.zeros(src_local.shape[0], dtype=np.float32)
        valid = pair_valid[dst_orig, src_orig, timestep]
        imp[valid] = pair_features[dst_orig[valid], src_orig[valid], timestep, 9]
        return torch.cat([edge_attr, torch.from_numpy(imp).unsqueeze(1)], dim=1)

    def describe(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "split": self.split,
            "data_dir": str(self.data_dir),
            "source": "multiagent",
            "num_scenes": self.len(),
            "num_scenes_available": self.store.num_scenes,
            "history_len": self.history_len,
            "future_len": self.future_len,
            "max_agents": self.store.max_agents,
            "feature_mode": self.feature_mode,
            "node_feature_channels": self.n_node_features,
            "node_feature_names": ["x", "y", "xV", "yV", "xA", "yA"],
            "edge_feature_channels": self.edge_feature_dim,
            "edge_feature_names": ["distance", "I"] if self.use_importance else ["distance"],
            "use_importance": self.use_importance,
            "excluded_proposed_node_channels": ["lc_state", "lit", "lis", "gate", "I_x", "I_y", "dim"],
            "target_scope": "scored agents",
        }

    def channel_stats(self, n_samples: int = 256) -> dict[str, Any]:
        arrays = self.store.open()
        n = min(int(n_samples), self.len())
        retained: list[int] = []
        scored: list[int] = []
        for j in range(n):
            scene_idx = int(self.scene_indices[j])
            keep = scene_agent_indices(arrays["agent_ids"][scene_idx], arrays["obs_valid"][scene_idx])
            retained.append(int(keep.size))
            scored.append(int(np.count_nonzero(arrays["scored_agent_mask"][scene_idx][keep])))
        return {
            "samples_inspected": n,
            "retained_agents_mean": float(np.mean(retained)) if retained else 0.0,
            "retained_agents_max": int(max(retained)) if retained else 0,
            "scored_agents_mean": float(np.mean(scored)) if scored else 0.0,
            "scored_agents_max": int(max(scored)) if scored else 0,
            "consumed_node_features": ["x", "y", "xV", "yV", "xA", "yA"],
            "consumed_edge_features": ["distance", "I"] if self.use_importance else ["distance"],
            "excluded_proposed_node_channels": ["lc_state", "lit", "lis", "gate", "I_x", "I_y", "dim"],
        }
