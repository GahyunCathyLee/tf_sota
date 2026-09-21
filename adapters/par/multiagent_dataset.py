"""PAR target-agent token dataset for persistent multi-agent scene arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from adapters.multiagent_common import MultiAgentArrays, scene_agent_indices, scored_local_indices
from adapters.par.dataset import (
    NeighFormerPARDataset,
    get_bins_first_order,
    positions_to_accel_tokens,
    second_order_dict,
)
from data.multiagent.interaction import InteractionConfig, build_pair_features

def _fill_nan_positions(pos: np.ndarray) -> np.ndarray:
    out = np.asarray(pos, dtype=np.float32).copy()
    valid = np.isfinite(out).all(axis=-1)
    if not valid.any():
        return np.zeros_like(out, dtype=np.float32)
    first = int(np.flatnonzero(valid)[0])
    out[:first] = out[first]
    last = out[first].copy()
    for i in range(first + 1, out.shape[0]):
        if valid[i]:
            last = out[i].copy()
        else:
            out[i] = last
    return out


class MultiAgentPARDataset(NeighFormerPARDataset):
    """PAR-compatible samples flattened over scored target agents per scene."""

    def __init__(
        self,
        data_dir: str | Path,
        dataset_name: str,
        feature_mode: str,
        split: str,
        indices: np.ndarray | None = None,
        acc_token_size: int = 13,
        velocity_bins: int = 128,
        use_importance: bool = False,
    ) -> None:
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
        self.acc_token_size = int(acc_token_size)
        self.pad_index = self.acc_token_size * self.acc_token_size
        self.bins = get_bins_first_order(int(velocity_bins))
        self.second_table = second_order_dict(self.acc_token_size)
        self.history_len = self.store.history_len
        self.future_len = self.store.future_len
        self.raw_max_neighbors = max(0, self.store.max_agents - 1)
        self.max_neighbors = self.raw_max_neighbors
        self.num_agents = self.store.max_agents
        self.ego_agent_id = self.num_agents - 1
        self.token_steps = self.history_len + self.future_len - 2
        self.obs_token_steps = self.history_len - 2
        self.side_dim = 1 if self.use_importance else 0
        self.has_neighbor_future = False
        self.has_fixed_neighbor_history = True
        self.has_neighbor_attrs = self.side_dim > 0
        self._arrays: dict[str, np.ndarray] | None = None
        self.sample_map = self._build_sample_map()
        self.n_samples_total = int(self.sample_map.shape[0])

    def _ensure_open(self) -> dict[str, np.ndarray]:
        return self.store.open()

    def _build_sample_map(self) -> np.ndarray:
        arrays = self.store.open()
        rows: list[tuple[int, int]] = []
        for scene_idx in self.scene_indices.tolist():
            scene_idx = int(scene_idx)
            keep = scene_agent_indices(arrays["agent_ids"][scene_idx], arrays["obs_valid"][scene_idx])
            scored_local = scored_local_indices(arrays["scored_agent_mask"][scene_idx], keep)
            if scored_local.size == 0 and keep.size:
                scored_local = np.asarray([0], dtype=np.int64)
            for local in scored_local.tolist():
                rows.append((scene_idx, int(keep[int(local)])))
        if not rows:
            return np.zeros((0, 2), dtype=np.int64)
        return np.asarray(rows, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.sample_map.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        arrays = self._ensure_open()
        scene_idx, target_orig = (int(v) for v in self.sample_map[int(idx)])
        agent_ids = np.asarray(arrays["agent_ids"][scene_idx])
        x = np.asarray(arrays["x_agents"][scene_idx], dtype=np.float32)
        obs = np.asarray(arrays["obs_valid"][scene_idx], dtype=bool)
        y = np.asarray(arrays["y_agents"][scene_idx], dtype=np.float32)
        fut_valid = np.asarray(arrays["future_valid"][scene_idx], dtype=bool)
        keep = scene_agent_indices(agent_ids, obs)

        context = [int(v) for v in keep.tolist() if int(v) != target_orig][: self.num_agents - 1]
        slot_for_orig = {orig: slot for slot, orig in enumerate(context)}
        slot_for_orig[target_orig] = self.ego_agent_id

        total_steps = self.history_len + self.future_len
        agent_pos = np.full((self.num_agents, total_steps, 2), np.nan, dtype=np.float32)
        for orig, slot in slot_for_orig.items():
            agent_pos[slot, : self.history_len] = np.where(obs[orig, :, None], x[orig, :, 0:2], np.nan)
            agent_pos[slot, self.history_len :] = np.where(fut_valid[orig, :, None], y[orig, :, 0:2], np.nan)

        per_agent_tokens = np.stack(
            [
                positions_to_accel_tokens(pos, self.bins, self.second_table, self.acc_token_size, self.pad_index)
                for pos in agent_pos
            ],
            axis=0,
        )
        tokens = per_agent_tokens.T.reshape(-1).astype(np.int64)
        agent_ids_flat = np.tile(np.arange(self.num_agents, dtype=np.int64), self.token_steps)
        token_time = np.repeat(np.arange(self.token_steps, dtype=np.int64), self.num_agents)
        future_mask = token_time >= self.obs_token_steps
        loss_mask = (agent_ids_flat == self.ego_agent_id) & future_mask & (tokens != self.pad_index)

        if self.use_importance:
            side = self._importance_side_channel(arrays, scene_idx, keep, slot_for_orig, target_orig)
            side_flat = side.reshape(-1, 1)
            side_flat[tokens == self.pad_index, 0] = 0.0
        else:
            side_flat = np.zeros((tokens.shape[0], 0), dtype=np.float32)

        target = y[target_orig, :, 0:2].copy()
        target[~fut_valid[target_orig]] = np.nan
        hist = x[target_orig, :, 0:2].copy()
        hist[~obs[target_orig]] = np.nan
        hist = _fill_nan_positions(hist)
        return {
            "tokens": torch.from_numpy(tokens),
            "loss_mask": torch.from_numpy(loss_mask),
            "side": torch.from_numpy(side_flat),
            "agent_ids": torch.from_numpy(agent_ids_flat),
            "target": torch.from_numpy(target),
            "hist_pos": torch.from_numpy(hist),
            "ego_tokens": torch.from_numpy(per_agent_tokens[self.ego_agent_id].copy()),
            "sample_index": torch.tensor(scene_idx, dtype=torch.long),
            "target_agent_index": torch.tensor(target_orig, dtype=torch.long),
        }

    def _importance_side_channel(
        self,
        arrays: dict[str, np.ndarray],
        scene_idx: int,
        keep: np.ndarray,
        slot_for_orig: dict[int, int],
        target_orig: int,
    ) -> np.ndarray:
        """Build PAR token-aligned ``I`` for target->context interactions.

        Future tokens reuse the last observed importance value so no future
        interaction labels leak into the autoregressive training target.
        """
        side = np.zeros((self.token_steps, self.num_agents, 1), dtype=np.float32)
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

        keep_set = set(int(v) for v in keep.tolist())
        if int(target_orig) not in keep_set:
            return side
        for orig, slot in slot_for_orig.items():
            orig = int(orig)
            if orig == target_orig or orig not in keep_set:
                continue
            last_i = 0.0
            for tok_t in range(self.obs_token_steps):
                src_t = min(tok_t + 2, self.history_len - 1)
                if pair_valid[target_orig, orig, src_t]:
                    last_i = float(pair_features[target_orig, orig, src_t, 9])
                    side[tok_t, slot, 0] = last_i
            side[self.obs_token_steps :, slot, 0] = last_i
        return side

    def describe(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "data_dir": str(self.data_dir),
            "dataset": self.dataset_name,
            "feature_mode": self.feature_mode,
            "source": "multiagent",
            "num_samples": len(self),
            "num_scenes": int(self.scene_indices.size),
            "history_len": self.history_len,
            "future_len": self.future_len,
            "max_agents": self.store.max_agents,
            "num_agents": self.num_agents,
            "token_steps": self.token_steps,
            "observed_token_steps": self.obs_token_steps,
            "vocab_size": self.pad_index + 1,
            "side_channel_dim": self.side_dim,
            "use_importance": self.use_importance,
            "consumed_continuous_side_channels": ["I"] if self.use_importance else [],
            "excluded_proposed_channels": ["lc_state", "lit", "lis", "gate", "I_x", "I_y", "dim"],
            "future_loss_agents": "target_scored_agent",
        }

    def channel_stats(self, n_samples: int = 256) -> dict[str, Any]:
        arrays = self.store.open()
        n = min(int(n_samples), int(self.scene_indices.size))
        retained: list[int] = []
        scored: list[int] = []
        for j in range(n):
            scene_idx = int(self.scene_indices[j])
            keep = scene_agent_indices(arrays["agent_ids"][scene_idx], arrays["obs_valid"][scene_idx])
            retained.append(int(keep.size))
            scored.append(int(scored_local_indices(arrays["scored_agent_mask"][scene_idx], keep).size))
        return {
            "samples_inspected": n,
            "retained_agents_mean": float(np.mean(retained)) if retained else 0.0,
            "retained_agents_max": int(max(retained)) if retained else 0,
            "scored_targets_mean": float(np.mean(scored)) if scored else 0.0,
            "scored_targets_max": int(max(scored)) if scored else 0,
            "consumed_features": ["trajectory_tokens"] + (["I"] if self.use_importance else []),
            "excluded_proposed_channels": ["lc_state", "lit", "lis", "gate", "I_x", "I_y", "dim"],
        }
