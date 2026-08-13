"""NeighFormer npy -> PAR trajectory-token batches.

PAR's car task is an autoregressive token model over xy trajectories.  When
``data/par/preprocess.py`` has produced fixed-ID neighbour futures, this adapter
uses multi-agent future tokens for the training loss.  It falls back to ego-only
future loss for canonical NeighFormer arrays without ``y_nb.npy``.  In ``dimI``
mode the extra neighbour ``dim`` and ``I`` channels are kept as continuous
side-channel features attached to the token embeddings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from adapters.common import feature_mode_indices, feature_mode_names

EGO_EXTRA_FILL = -1.0


def get_bins_first_order(num_bins: int = 128, range_min: float = -18.0, range_max: float = 18.0) -> np.ndarray:
    """Match upstream PAR's first-order velocity-token bin edges."""
    fine_bins = 20
    fine_range_min = -2.0
    fine_range_max = 2.0
    epsilon = 1e-5
    coarse_each_side = (num_bins - fine_bins - 2) // 2
    fine_each_side = fine_bins // 2
    coarse_left = np.linspace(range_min, fine_range_min, coarse_each_side + 1, endpoint=False)
    coarse_right = np.linspace(fine_range_max, range_max, coarse_each_side + 1)[1:]
    fine_left = np.linspace(-1.0, -epsilon, fine_each_side + 1, endpoint=False)[1:]
    fine_right = np.linspace(epsilon, 1.0, fine_each_side + 1, endpoint=False)[1:]
    zero_bins = np.array([0.0, epsilon], dtype=np.float64)
    return np.concatenate((coarse_left, fine_left, zero_bins, fine_right, coarse_right))


def second_order_dict(n: int) -> dict[int, int]:
    middle = n // 2
    return {i - middle: i for i in range(n)}


def _velocity_bin(delta: float, bins: np.ndarray) -> int:
    idx = int(np.digitize(delta, bins) - 1)
    return int(np.clip(idx, 0, len(bins) - 2))


def _accel_index(delta: int, table: dict[int, int], n: int) -> int:
    half = n // 2
    return table[int(np.clip(delta, -half, half))]


def positions_to_accel_tokens(
    positions: np.ndarray,
    bins: np.ndarray,
    second_table: dict[int, int],
    acc_token_size: int,
    pad_index: int,
) -> np.ndarray:
    """Convert ``[T, 2]`` positions to PAR acceleration vocabulary indices."""
    pos = np.asarray(positions, dtype=np.float32)
    token_len = max(0, pos.shape[0] - 2)
    out = np.full(token_len, pad_index, dtype=np.int64)
    if token_len == 0:
        return out

    for t in range(token_len):
        triad = pos[t : t + 3]
        if not np.isfinite(triad).all():
            continue
        deltas = np.diff(triad, axis=0)
        vx0 = _velocity_bin(float(deltas[0, 0]), bins)
        vy0 = _velocity_bin(float(deltas[0, 1]), bins)
        vx1 = _velocity_bin(float(deltas[1, 0]), bins)
        vy1 = _velocity_bin(float(deltas[1, 1]), bins)
        ax_idx = _accel_index(vx1 - vx0, second_table, acc_token_size)
        ay_idx = _accel_index(vy1 - vy0, second_table, acc_token_size)
        out[t] = ax_idx * acc_token_size + ay_idx
    return out


def reconstruct_accel_tokens(
    start: torch.Tensor,
    start_velocity: torch.Tensor,
    tokens: torch.Tensor,
    bins: np.ndarray,
    acc_token_size: int,
    pad_index: int,
) -> torch.Tensor:
    """Reconstruct xy positions from PAR acceleration tokens.

    ``start`` and ``start_velocity`` are ``[B, 2]``. ``tokens`` is ``[B, S]`` and
    reconstructs ``S + 2`` positions, matching upstream PAR's acceleration
    tokenizer.
    """
    device = tokens.device
    dtype = start.dtype
    bins_t = torch.as_tensor(bins, device=device, dtype=dtype)
    reverse = {v: k for k, v in second_order_dict(acc_token_size).items()}
    batch, steps = tokens.shape
    pos = torch.zeros(batch, steps + 2, 2, device=device, dtype=dtype)
    pos[:, 0] = start
    vel = start_velocity.clone()
    pos[:, 1] = start + vel
    vel_token_x = torch.bucketize(vel[:, 0].detach().cpu(), torch.as_tensor(bins)).to(device) - 1
    vel_token_y = torch.bucketize(vel[:, 1].detach().cpu(), torch.as_tensor(bins)).to(device) - 1
    vel_token_x = vel_token_x.clamp(0, len(bins) - 2)
    vel_token_y = vel_token_y.clamp(0, len(bins) - 2)

    cur = pos[:, 1].clone()
    for t in range(steps):
        tok = tokens[:, t]
        valid = tok != pad_index
        ax_idx = torch.div(tok.clamp(0, pad_index - 1), acc_token_size, rounding_mode="floor")
        ay_idx = tok.clamp(0, pad_index - 1) % acc_token_size
        ax_delta = torch.tensor([reverse[int(v)] for v in ax_idx.detach().cpu()], device=device)
        ay_delta = torch.tensor([reverse[int(v)] for v in ay_idx.detach().cpu()], device=device)
        vel_token_x = (vel_token_x + ax_delta).clamp(0, len(bins) - 2)
        vel_token_y = (vel_token_y + ay_delta).clamp(0, len(bins) - 2)
        dx = (bins_t[vel_token_x] + bins_t[vel_token_x + 1]) * 0.5
        dy = (bins_t[vel_token_y] + bins_t[vel_token_y + 1]) * 0.5
        cur = cur + torch.stack([dx, dy], dim=-1)
        pos[:, t + 2] = torch.where(valid.unsqueeze(-1), cur, torch.nan)
    return pos


class NeighFormerPARDataset(Dataset):
    """PAR-compatible highD/exiD dataset backed by NeighFormer mmap npy files."""

    def __init__(
        self,
        data_dir: str | Path,
        indices: np.ndarray,
        dataset_name: str,
        feature_mode: str,
        split: str,
        acc_token_size: int = 13,
        velocity_bins: int = 128,
        max_neighbors: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.sample_indices = np.asarray(indices, dtype=np.int64)
        self.dataset_name = dataset_name
        self.feature_mode = feature_mode
        self.split = split
        self.nb_feature_indices = np.asarray(feature_mode_indices(feature_mode), dtype=np.int64)
        self.nb_feature_names = feature_mode_names(feature_mode)
        self.acc_token_size = int(acc_token_size)
        self.pad_index = self.acc_token_size * self.acc_token_size
        self.bins = get_bins_first_order(int(velocity_bins))
        self.second_table = second_order_dict(self.acc_token_size)
        self._arrays: dict[str, np.ndarray] | None = None

        x_ego = np.load(self.data_dir / "x_ego.npy", mmap_mode="r")
        x_nb = np.load(self.data_dir / "x_nb.npy", mmap_mode="r")
        y = np.load(self.data_dir / "y.npy", mmap_mode="r")
        self.n_samples_total = int(x_ego.shape[0])
        self.history_len = int(x_ego.shape[1])
        self.future_len = int(y.shape[1])
        self.raw_max_neighbors = int(x_nb.shape[2])
        self.max_neighbors = int(max_neighbors or self.raw_max_neighbors)
        self.num_agents = self.max_neighbors + 1
        self.ego_agent_id = self.num_agents - 1
        self.token_steps = self.history_len + self.future_len - 2
        self.obs_token_steps = self.history_len - 2
        self.side_dim = 2 if self.feature_mode == "dimI" else 0
        self.has_neighbor_future = (self.data_dir / "y_nb.npy").exists() and (self.data_dir / "y_nb_mask.npy").exists()
        self.has_fixed_neighbor_history = (
            (self.data_dir / "x_nb_abs.npy").exists()
            and (self.data_dir / "x_nb_abs_mask.npy").exists()
        )
        self.has_neighbor_attrs = (self.data_dir / "nb_attr.npy").exists() and (self.data_dir / "nb_attr_mask.npy").exists()
        if int(x_ego.shape[2]) != 6:
            raise ValueError(f"Expected x_ego[..., 6], got {x_ego.shape}")
        if int(x_nb.shape[3]) < int(self.nb_feature_indices.max()) + 1:
            raise ValueError(
                f"x_nb has {x_nb.shape[3]} channels; {feature_mode} needs index "
                f"{int(self.nb_feature_indices.max())}"
            )
        if self.history_len < 3:
            raise ValueError("PAR acceleration tokens require at least 3 history positions")

    def _ensure_open(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            arrays = {
                "x_ego": np.load(self.data_dir / "x_ego.npy", mmap_mode="r"),
                "x_nb": np.load(self.data_dir / "x_nb.npy", mmap_mode="r"),
                "nb_mask": np.load(self.data_dir / "nb_mask.npy", mmap_mode="r"),
                "y": np.load(self.data_dir / "y.npy", mmap_mode="r"),
            }
            optional = {
                "x_nb_abs": "x_nb_abs.npy",
                "x_nb_abs_mask": "x_nb_abs_mask.npy",
                "y_nb": "y_nb.npy",
                "y_nb_mask": "y_nb_mask.npy",
                "nb_ids": "nb_ids.npy",
                "nb_attr": "nb_attr.npy",
                "nb_attr_mask": "nb_attr_mask.npy",
            }
            for key, filename in optional.items():
                path = self.data_dir / filename
                if path.exists():
                    arrays[key] = np.load(path, mmap_mode="r")
            for name in ("meta_recordingId", "meta_trackId", "meta_frame"):
                path = self.data_dir / f"{name}.npy"
                if path.exists():
                    arrays[name] = np.load(path, mmap_mode="r")
            self._arrays = arrays
        return self._arrays

    def __len__(self) -> int:
        return int(self.sample_indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        arrays = self._ensure_open()
        real_idx = int(self.sample_indices[idx])
        ego = np.asarray(arrays["x_ego"][real_idx], dtype=np.float32)
        nb = np.asarray(arrays["x_nb"][real_idx], dtype=np.float32)
        mask = np.asarray(arrays["nb_mask"][real_idx], dtype=bool)
        fut = np.asarray(arrays["y"][real_idx], dtype=np.float32)

        total_steps = self.history_len + self.future_len
        agent_pos = np.full((self.num_agents, total_steps, 2), np.nan, dtype=np.float32)
        agent_pos[self.ego_agent_id, : self.history_len] = ego[:, :2]
        agent_pos[self.ego_agent_id, self.history_len :] = fut
        if self.has_fixed_neighbor_history and "x_nb_abs" in arrays and "x_nb_abs_mask" in arrays:
            hist_abs = np.asarray(arrays["x_nb_abs"][real_idx], dtype=np.float32)
            hist_abs_mask = np.asarray(arrays["x_nb_abs_mask"][real_idx], dtype=bool)
            for slot in range(min(self.max_neighbors, hist_abs.shape[1])):
                agent_pos[slot, : self.history_len] = np.where(
                    hist_abs_mask[:, slot, None],
                    hist_abs[:, slot, 0:2],
                    np.nan,
                )
        else:
            for slot in range(min(self.max_neighbors, self.raw_max_neighbors)):
                hist_pos = ego[:, :2] + nb[:, slot, 0:2]
                hist_pos = np.where(mask[:, slot, None], hist_pos, np.nan)
                agent_pos[slot, : self.history_len] = hist_pos
        if self.has_neighbor_future and "y_nb" in arrays and "y_nb_mask" in arrays:
            fut_abs = np.asarray(arrays["y_nb"][real_idx], dtype=np.float32)
            fut_abs_mask = np.asarray(arrays["y_nb_mask"][real_idx], dtype=bool)
            for slot in range(min(self.max_neighbors, fut_abs.shape[1])):
                agent_pos[slot, self.history_len :] = np.where(
                    fut_abs_mask[:, slot, None],
                    fut_abs[:, slot, 0:2],
                    np.nan,
                )

        per_agent_tokens = np.stack(
            [
                positions_to_accel_tokens(pos, self.bins, self.second_table, self.acc_token_size, self.pad_index)
                for pos in agent_pos
            ],
            axis=0,
        )
        tokens = per_agent_tokens.T.reshape(-1).astype(np.int64)
        agent_ids = np.tile(np.arange(self.num_agents, dtype=np.int64), self.token_steps)
        token_time = np.repeat(np.arange(self.token_steps, dtype=np.int64), self.num_agents)
        future_mask = token_time >= self.obs_token_steps
        if self.has_neighbor_future:
            loss_mask = future_mask & (tokens != self.pad_index)
        else:
            loss_mask = (agent_ids == self.ego_agent_id) & future_mask & (tokens != self.pad_index)

        if self.side_dim:
            side = np.zeros((self.token_steps, self.num_agents, self.side_dim), dtype=np.float32)
            side[:, self.ego_agent_id, :] = EGO_EXTRA_FILL
            if self.has_neighbor_attrs and "nb_attr" in arrays and "nb_attr_mask" in arrays:
                attr = np.asarray(arrays["nb_attr"][real_idx], dtype=np.float32)
                attr_mask = np.asarray(arrays["nb_attr_mask"][real_idx], dtype=bool)
                for slot in range(min(self.max_neighbors, attr.shape[0])):
                    if attr_mask[slot]:
                        side[:, slot, :] = attr[slot]
            else:
                for slot in range(min(self.max_neighbors, self.raw_max_neighbors)):
                    last_attr = None
                    for tok_t in range(self.obs_token_steps):
                        src_t = min(tok_t + 2, self.history_len - 1)
                        if mask[src_t, slot]:
                            last_attr = nb[src_t, slot, 8:10]
                            side[tok_t, slot, :] = last_attr
                    if last_attr is not None:
                        side[self.obs_token_steps :, slot, :] = last_attr
            side_flat = side.reshape(-1, self.side_dim)
        else:
            side_flat = np.zeros((tokens.shape[0], 0), dtype=np.float32)

        return {
            "tokens": torch.from_numpy(tokens),
            "loss_mask": torch.from_numpy(loss_mask),
            "side": torch.from_numpy(side_flat),
            "agent_ids": torch.from_numpy(agent_ids),
            "target": torch.from_numpy(fut.copy()),
            "hist_pos": torch.from_numpy(ego[:, :2].copy()),
            "ego_tokens": torch.from_numpy(per_agent_tokens[self.ego_agent_id].copy()),
            "sample_index": torch.tensor(real_idx, dtype=torch.long),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "data_dir": str(self.data_dir),
            "dataset": self.dataset_name,
            "feature_mode": self.feature_mode,
            "num_samples": len(self),
            "num_samples_available": self.n_samples_total,
            "history_len": self.history_len,
            "future_len": self.future_len,
            "max_neighbors": self.max_neighbors,
            "num_agents": self.num_agents,
            "token_steps": self.token_steps,
            "observed_token_steps": self.obs_token_steps,
            "vocab_size": self.pad_index + 1,
            "side_channel_dim": self.side_dim,
            "neighbor_indices": [int(v) for v in self.nb_feature_indices],
            "neighbor_names": self.nb_feature_names,
            "dimI_mapping": "continuous side-channel embedding" if self.side_dim else "disabled in baseline",
            "neighbor_future_used": self.has_neighbor_future,
            "fixed_neighbor_history_used": self.has_fixed_neighbor_history,
            "neighbor_attrs_used": self.has_neighbor_attrs,
            "future_loss_agents": "ego+neighbors" if self.has_neighbor_future else "ego_only",
        }

    def channel_stats(self, n_samples: int = 256) -> dict[str, Any]:
        arrays = self._ensure_open()
        n = min(n_samples, len(self))
        rows = []
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
