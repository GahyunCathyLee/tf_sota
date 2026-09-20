"""Shared helpers for persistent multi-agent adapter datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

MULTIAGENT_ARRAYS = (
    "agent_ids",
    "x_agents",
    "obs_valid",
    "y_agents",
    "future_valid",
    "scored_agent_mask",
    "agent_length",
    "agent_width",
    "agent_type",
    "heading",
    "lateral_velocity",
    "lane_id",
    "lane_level",
    "lane_offset",
    "lane_width",
    "ego_index",
    "recordingId",
    "ego_trackId",
    "t0_frame",
    "source_index",
)


def multiagent_split_dir(root: Path, dataset: str, split: str) -> Path:
    suffix = split if split.endswith("_full") else f"{split}_full"
    return root / f"{dataset}_multiagent" / suffix


def multiagent_indices(split_root: str | Path, limit: int | None = None) -> np.ndarray:
    n = int(np.load(Path(split_root) / "agent_ids.npy", mmap_mode="r").shape[0])
    indices = np.arange(n, dtype=np.int64)
    return indices if limit is None else indices[: int(limit)]


class MultiAgentArrays:
    """Lazy memmap loader for ``data/multiagent/preprocess.py`` outputs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._arrays: dict[str, np.ndarray] | None = None
        required = self.root / "x_agents.npy"
        if not required.exists():
            raise FileNotFoundError(required)
        head = np.load(required, mmap_mode="r")
        fut = np.load(self.root / "y_agents.npy", mmap_mode="r")
        self.num_scenes = int(head.shape[0])
        self.max_agents = int(head.shape[1])
        self.history_len = int(head.shape[2])
        self.future_len = int(fut.shape[2])

    def open(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            self._arrays = {
                name: np.load(self.root / f"{name}.npy", mmap_mode="r")
                for name in MULTIAGENT_ARRAYS
                if (self.root / f"{name}.npy").exists()
            }
        return self._arrays


def scene_agent_indices(agent_ids: np.ndarray, obs_valid: np.ndarray) -> np.ndarray:
    current = np.asarray(obs_valid[:, -1], dtype=bool)
    present = np.asarray(agent_ids >= 0, dtype=bool)
    keep = np.flatnonzero(present & current)
    if keep.size == 0 and present.any():
        keep = np.flatnonzero(present)[:1]
    return keep.astype(np.int64)


def agent_positions(x_agents: np.ndarray, y_agents: np.ndarray, keep: np.ndarray) -> np.ndarray:
    hist = np.asarray(x_agents[keep, :, 0:2], dtype=np.float32)
    fut = np.asarray(y_agents[keep, :, 0:2], dtype=np.float32)
    return np.concatenate([hist, fut], axis=1)


def agent_velocities(x_agents: np.ndarray, y_agents: np.ndarray, keep: np.ndarray) -> np.ndarray:
    hist = np.asarray(x_agents[keep, :, 2:4], dtype=np.float32)
    fut = np.asarray(y_agents[keep, :, 2:4], dtype=np.float32)
    return np.concatenate([hist, fut], axis=1)


def scored_local_indices(scored_mask: np.ndarray, keep: np.ndarray) -> np.ndarray:
    scored = set(int(v) for v in np.flatnonzero(np.asarray(scored_mask, dtype=bool)))
    return np.asarray([i for i, src in enumerate(keep.tolist()) if int(src) in scored], dtype=np.int64)


def heading_from_velocity(vxy: np.ndarray) -> np.ndarray:
    return np.arctan2(vxy[..., 1], vxy[..., 0]).astype(np.float32)


def velocity_from_positions(pos: np.ndarray, valid: np.ndarray, dt: float = 1.0) -> np.ndarray:
    vel = np.zeros_like(pos, dtype=np.float32)
    if pos.shape[-2] > 1:
        pair_valid = valid[..., 1:] & valid[..., :-1]
        vel[..., 1:, :] = np.where(pair_valid[..., None], (pos[..., 1:, :] - pos[..., :-1, :]) / float(dt), 0.0)
        vel[..., 0, :] = vel[..., 1, :]
    return vel


def meta_dict(arrays: dict[str, np.ndarray], scene_idx: int) -> dict[str, Any]:
    return {
        "scene_index": int(scene_idx),
        "source_index": int(arrays["source_index"][scene_idx]) if "source_index" in arrays else int(scene_idx),
        "recordingId": int(arrays["recordingId"][scene_idx]) if "recordingId" in arrays else -1,
        "ego_trackId": int(arrays["ego_trackId"][scene_idx]) if "ego_trackId" in arrays else -1,
        "t0_frame": int(arrays["t0_frame"][scene_idx]) if "t0_frame" in arrays else -1,
    }
