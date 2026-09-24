"""Dataset mapping from NeighFormer dimI arrays to MTFT agent sequences.

The processed data are already in the ego-centered frame whose origin is the
target vehicle's last observed position. MTFT consumes target and surrounding
vehicle coordinates, so neighbor coordinates are reconstructed as:

    neighbor_xy[t, k] = ego_xy[t] + x_nb[t, k, 0:2]

For the +I condition, channel 9 from x_nb is appended only to surrounding
vehicles. The target vehicle receives neutral I=0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

I_CHANNEL = 9


@dataclass(frozen=True)
class MTFTBatchShape:
    history_steps: int
    future_steps: int
    num_neighbors: int
    input_dim: int


class MTFTDataset(Dataset):
    """Single-target MTFT samples from highD/exiD dimI mmap arrays."""

    def __init__(
        self,
        data_dir: str | Path,
        indices: np.ndarray | None = None,
        use_i: bool = False,
        missing_enabled: bool = False,
        missing_min_ratio: float = 0.0,
        missing_max_ratio: float = 0.0,
        seed: int = 42,
        max_samples: int | None = None,
        return_meta: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.use_i = bool(use_i)
        self.missing_enabled = bool(missing_enabled)
        self.missing_min_ratio = float(missing_min_ratio)
        self.missing_max_ratio = float(missing_max_ratio)
        self.seed = int(seed)
        self.return_meta = bool(return_meta)

        self.x_ego = np.load(self.data_dir / "x_ego.npy", mmap_mode="r")
        self.x_nb = np.load(self.data_dir / "x_nb.npy", mmap_mode="r")
        self.nb_mask = np.load(self.data_dir / "nb_mask.npy", mmap_mode="r")
        self.y = np.load(self.data_dir / "y.npy", mmap_mode="r")
        self.y_vel = self._load_optional("y_vel.npy")
        self.y_acc = self._load_optional("y_acc.npy")
        self.x_last_abs = self._load_optional("x_last_abs.npy")
        self.meta_recording = self._load_optional("meta_recordingId.npy") if return_meta else None
        self.meta_track = self._load_optional("meta_trackId.npy") if return_meta else None
        self.meta_frame = self._load_optional("meta_frame.npy") if return_meta else None

        n = int(self.x_ego.shape[0])
        base_indices = np.arange(n, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        if max_samples is not None:
            base_indices = base_indices[: int(max_samples)]
        self.indices = base_indices

        if self.x_nb.shape[-1] <= I_CHANNEL:
            raise ValueError(f"{self.data_dir}/x_nb.npy must contain channel {I_CHANNEL} for I")
        if self.x_ego.shape[1] != self.x_nb.shape[1] or self.x_ego.shape[1] != self.nb_mask.shape[1]:
            raise ValueError("history length mismatch among x_ego, x_nb, nb_mask")
        if not 0.0 <= self.missing_min_ratio <= self.missing_max_ratio <= 1.0:
            raise ValueError("missing ratios must satisfy 0 <= min <= max <= 1")

    def _load_optional(self, filename: str) -> np.ndarray | None:
        path = self.data_dir / filename
        return np.load(path, mmap_mode="r") if path.exists() else None

    @property
    def shape(self) -> MTFTBatchShape:
        return MTFTBatchShape(
            history_steps=int(self.x_ego.shape[1]),
            future_steps=int(self.y.shape[1]),
            num_neighbors=int(self.x_nb.shape[2]),
            input_dim=3 if self.use_i else 2,
        )

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> dict[str, Any]:
        idx = int(self.indices[item])
        ego = np.asarray(self.x_ego[idx], dtype=np.float32)
        nb = np.asarray(self.x_nb[idx], dtype=np.float32)
        nb_mask = np.asarray(self.nb_mask[idx], dtype=np.bool_)

        ego_xy = ego[:, :2]
        nb_xy = ego_xy[:, None, :] + nb[..., :2]
        agents_xy = np.concatenate([ego_xy[:, None, :], nb_xy], axis=1)

        valid = np.concatenate(
            [np.ones((ego_xy.shape[0], 1), dtype=np.bool_), nb_mask],
            axis=1,
        )

        if self.use_i:
            target_i = np.zeros((ego_xy.shape[0], 1, 1), dtype=np.float32)
            nb_i = nb[..., I_CHANNEL : I_CHANNEL + 1]
            agents = np.concatenate([agents_xy, np.concatenate([target_i, nb_i], axis=1)], axis=-1)
        else:
            agents = agents_xy

        obs_mask = valid.copy()
        if self.missing_enabled and self.missing_max_ratio > 0.0:
            agents, obs_mask = self._apply_missing(agents, obs_mask, idx)

        target = np.asarray(self.y[idx], dtype=np.float32)
        y_vel = np.asarray(self.y_vel[idx], dtype=np.float32) if self.y_vel is not None else np.zeros_like(target)
        y_acc = np.asarray(self.y_acc[idx], dtype=np.float32) if self.y_acc is not None else np.zeros_like(target)

        out: dict[str, Any] = {
            "agents": torch.from_numpy(agents.copy()),
            "obs_mask": torch.from_numpy(obs_mask.copy()),
            "agent_mask": torch.from_numpy(valid.any(axis=0).copy()),
            "target": torch.from_numpy(target.copy()),
            "y_vel": torch.from_numpy(y_vel.copy()),
            "y_acc": torch.from_numpy(y_acc.copy()),
            "sample_index": torch.tensor(idx, dtype=torch.long),
        }
        if self.x_last_abs is not None:
            out["x_last_abs"] = torch.from_numpy(np.asarray(self.x_last_abs[idx], dtype=np.float32).copy())
        if self.return_meta and self.meta_recording is not None:
            out["meta"] = {
                "recordingId": int(self.meta_recording[idx]),
                "trackId": int(self.meta_track[idx]) if self.meta_track is not None else -1,
                "t0_frame": int(self.meta_frame[idx]) if self.meta_frame is not None else -1,
            }
        return out

    def _apply_missing(
        self,
        agents: np.ndarray,
        obs_mask: np.ndarray,
        real_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed + int(real_idx))
        t_hist = agents.shape[0]
        valid_agents = np.where(obs_mask.any(axis=0))[0]
        for agent_idx in valid_agents:
            valid_times = np.where(obs_mask[:, agent_idx])[0]
            if valid_times.size == 0:
                continue
            ratio = rng.uniform(self.missing_min_ratio, self.missing_max_ratio)
            count = int(round(float(valid_times.size) * ratio))
            if count <= 0:
                continue
            count = min(count, valid_times.size)
            drop_times = rng.choice(valid_times, size=count, replace=False)
            obs_mask[drop_times, agent_idx] = False
            agents[drop_times, agent_idx, :] = 0.0
        if not obs_mask[:, 0].any():
            obs_mask[t_hist - 1, 0] = True
        return agents, obs_mask


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "agents": torch.stack([b["agents"] for b in batch]),
        "obs_mask": torch.stack([b["obs_mask"] for b in batch]),
        "agent_mask": torch.stack([b["agent_mask"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "y_vel": torch.stack([b["y_vel"] for b in batch]),
        "y_acc": torch.stack([b["y_acc"] for b in batch]),
        "sample_index": torch.stack([b["sample_index"] for b in batch]),
    }
    if "x_last_abs" in batch[0]:
        out["x_last_abs"] = torch.stack([b["x_last_abs"] for b in batch])
    if "meta" in batch[0]:
        out["meta"] = [b["meta"] for b in batch]
    return out


def dataset_summary(dataset: MTFTDataset) -> dict[str, Any]:
    shape = dataset.shape
    nb_mask = dataset.nb_mask[dataset.indices]
    valid_counts = nb_mask.sum(axis=(1, 2))
    i_vals = dataset.x_nb[dataset.indices, ..., I_CHANNEL][nb_mask]
    return {
        "size": len(dataset),
        "history_steps": shape.history_steps,
        "future_steps": shape.future_steps,
        "num_neighbors": shape.num_neighbors,
        "input_dim": shape.input_dim,
        "valid_neighbor_obs_min": int(valid_counts.min()) if valid_counts.size else 0,
        "valid_neighbor_obs_mean": float(valid_counts.mean()) if valid_counts.size else 0.0,
        "valid_neighbor_obs_max": int(valid_counts.max()) if valid_counts.size else 0,
        "i_min": float(i_vals.min()) if i_vals.size else float("nan"),
        "i_mean": float(i_vals.mean()) if i_vals.size else float("nan"),
        "i_max": float(i_vals.max()) if i_vals.size else float("nan"),
    }

