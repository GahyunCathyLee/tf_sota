"""SIMPL batch conversion for persistent multi-agent arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from adapters.multiagent_common import MultiAgentArrays, meta_dict, scene_agent_indices, scored_local_indices
from adapters.simpl.dataset import NeighFormerSIMPLDataset, build_rpe


class MultiAgentSIMPLDataset(Dataset):
    """SIMPL-compatible dataset backed by full multi-agent scene arrays."""

    def __init__(
        self,
        data_dir: str | Path,
        dataset_name: str,
        split: str,
        indices: np.ndarray | None = None,
        lane_half_length: float = 120.0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.split = split
        self.store = MultiAgentArrays(self.data_dir)
        self.indices = np.arange(self.store.num_scenes, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        self.history_len = self.store.history_len
        self.future_len = self.store.future_len
        self.lane_half_length = float(lane_half_length)
        self.actor_feature_names = ["step_dx", "step_dy", "valid"]
        self.actor_feature_dim = 3

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        arrays = self.store.open()
        scene_idx = int(self.indices[idx])
        agent_ids = np.asarray(arrays["agent_ids"][scene_idx])
        x = np.asarray(arrays["x_agents"][scene_idx], dtype=np.float32)
        obs = np.asarray(arrays["obs_valid"][scene_idx], dtype=bool)
        y = np.asarray(arrays["y_agents"][scene_idx], dtype=np.float32)
        fut_valid = np.asarray(arrays["future_valid"][scene_idx], dtype=bool)
        scored = np.asarray(arrays["scored_agent_mask"][scene_idx], dtype=bool)
        keep = scene_agent_indices(agent_ids, obs)
        n_agents = int(keep.size)
        th, tf = self.history_len, self.future_len

        scene_obs = np.asarray(x[keep, :, 0:2], dtype=np.float32)
        pad_obs = np.asarray(obs[keep], dtype=np.float32)
        trajs_obs = np.zeros((n_agents, th, 2), dtype=np.float32)
        centers = np.zeros((n_agents, 2), dtype=np.float32)
        vecs = np.zeros((n_agents, 2), dtype=np.float32)
        for i in range(n_agents):
            trajs_obs[i], centers[i], vecs[i] = NeighFormerSIMPLDataset._actor_local_positions(
                scene_obs[i],
                pad_obs[i].astype(bool),
            )

        trajs_fut = np.zeros((n_agents, tf, 2), dtype=np.float32)
        pad_fut = np.asarray(fut_valid[keep], dtype=np.float32)
        for i, src in enumerate(keep):
            valid = fut_valid[src]
            if bool(valid.any()):
                trajs_fut[i, valid] = NeighFormerSIMPLDataset._to_actor_local(y[src, valid, 0:2], centers[i], vecs[i])

        lane_graph = self._pseudo_lane_graph()
        scene_ctrs = torch.from_numpy(centers)
        scene_vecs = torch.from_numpy(vecs)
        rpe = build_rpe(scene_ctrs, scene_vecs)
        meta = meta_dict(arrays, scene_idx)
        return {
            "SEQ_ID": f"{self.dataset_name}_{self.split}_{scene_idx}",
            "SAMPLE_INDEX": scene_idx,
            "META": meta,
            "AGENT_IDS": agent_ids[keep].astype(np.int64),
            "SCORED_AGENT_IDCS": scored_local_indices(scored, keep),
            "TRAJS_OBS": trajs_obs,
            "TRAJS_FUT": trajs_fut,
            "PAD_OBS": pad_obs,
            "PAD_FUT": pad_fut,
            "ACTOR_EXTRA": np.zeros((n_agents, th, 0), dtype=np.float32),
            "TRAJS_CTRS": centers,
            "TRAJS_VECS": vecs,
            "LANE_GRAPH": lane_graph,
            "RPE": rpe,
        }

    def _pseudo_lane_graph(self) -> dict[str, np.ndarray | int]:
        xs = np.linspace(-self.lane_half_length, self.lane_half_length, 11, dtype=np.float32)
        ctrs = np.stack([xs[:-1], np.zeros(xs.shape[0] - 1, dtype=np.float32)], axis=-1)
        vecs = np.stack([np.diff(xs), np.zeros(xs.shape[0] - 1, dtype=np.float32)], axis=-1)
        n = ctrs.shape[0]
        return {
            "num_lanes": int(n),
            "node_ctrs": ctrs,
            "node_vecs": vecs,
            "turn": np.zeros((n, 2), dtype=np.float32),
            "control": np.zeros((n,), dtype=np.float32),
            "intersect": np.zeros((n,), dtype=np.float32),
            "left": np.zeros((n,), dtype=np.float32),
            "right": np.zeros((n,), dtype=np.float32),
            "lane_ctrs": ctrs,
            "lane_vecs": vecs,
        }

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return NeighFormerSIMPLDataset.collate_fn(self, batch)

    def actor_gather(
        self,
        actors: list[torch.Tensor],
        pad_flags: list[torch.Tensor],
        actor_extra: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return NeighFormerSIMPLDataset.actor_gather(self, actors, pad_flags, actor_extra)

    def graph_gather(self, graphs: list[dict[str, Any]]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return NeighFormerSIMPLDataset.graph_gather(self, graphs)

    def channel_stats(self, n_samples: int = 256) -> dict[str, Any]:
        arrays = self.store.open()
        n = min(int(n_samples), len(self))
        retained = []
        scored = []
        for j in range(n):
            scene_idx = int(self.indices[j])
            retained.append(int(np.count_nonzero(arrays["agent_ids"][scene_idx] >= 0)))
            scored.append(int(np.count_nonzero(arrays["scored_agent_mask"][scene_idx])))
        return {
            "samples_inspected": n,
            "retained_agents_mean": float(np.mean(retained)) if retained else 0.0,
            "scored_agents_mean": float(np.mean(scored)) if scored else 0.0,
        }

    def describe(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "split": self.split,
            "data_dir": str(self.data_dir),
            "num_scenes": len(self),
            "history_len": self.history_len,
            "future_len": self.future_len,
            "source": "multiagent",
        }
