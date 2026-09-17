"""MTR++ batch dictionaries for persistent multi-agent arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pickle

from adapters.mtrpp.dataset import NeighFormerMTRDataset, OBJECT_TYPE_VEHICLE, _heading_from_velocity
from adapters.multiagent_common import MultiAgentArrays, scene_agent_indices, scored_local_indices


class MultiAgentMTRDataset:
    """MTR-compatible dataset backed by full multi-agent scene arrays."""

    def __init__(
        self,
        data_dir: str | Path,
        dataset_name: str,
        split: str,
        indices: np.ndarray | None = None,
        target_agent_mode: bool = False,
        dt: float = 1.0 / 3.0,
        map_polylines: int = 9,
        map_points_each_polyline: int = 20,
        lane_half_length: float = 160.0,
        lane_width: float = 3.7,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.split = split
        self.store = MultiAgentArrays(self.data_dir)
        self.scene_indices = np.arange(self.store.num_scenes, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        self.target_agent_mode = bool(target_agent_mode)
        self.history_len = self.store.history_len
        self.future_len = self.store.future_len
        self.dt = float(dt)
        self.vehicle_height = 1.6
        self.map_polylines = int(map_polylines)
        self.map_points_each_polyline = int(map_points_each_polyline)
        self.lane_half_length = float(lane_half_length)
        self.lane_width = float(lane_width)
        self.agent_attr_dim = 6 + 5 + (self.history_len + 1) + 2 + 2 + 2
        self.builder = self
        self._target_scene_indices: np.ndarray | None = None
        self._target_agent_indices: np.ndarray | None = None
        if self.target_agent_mode:
            self._build_target_index()

    def _build_target_index(self) -> None:
        arrays = self.store.open()
        scenes: list[int] = []
        agents: list[int] = []
        for scene_idx in self.scene_indices.tolist():
            agent_ids = np.asarray(arrays["agent_ids"][scene_idx])
            obs = np.asarray(arrays["obs_valid"][scene_idx], dtype=bool)
            scored = np.asarray(arrays["scored_agent_mask"][scene_idx], dtype=bool)
            keep = scene_agent_indices(agent_ids, obs)
            scored_local = scored_local_indices(scored, keep)
            if scored_local.size == 0 and keep.size:
                scored_local = np.asarray([0], dtype=np.int64)
            for local_i in scored_local.tolist():
                scenes.append(int(scene_idx))
                agents.append(int(keep[int(local_i)]))
        self._target_scene_indices = np.asarray(scenes, dtype=np.int64)
        self._target_agent_indices = np.asarray(agents, dtype=np.int64)

    def __len__(self) -> int:
        if self.target_agent_mode:
            assert self._target_scene_indices is not None
            return int(self._target_scene_indices.size)
        return int(self.scene_indices.size)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        arrays = self.store.open()
        forced_target_agent = None
        if self.target_agent_mode:
            assert self._target_scene_indices is not None and self._target_agent_indices is not None
            scene_idx = int(self._target_scene_indices[idx])
            forced_target_agent = int(self._target_agent_indices[idx])
        else:
            scene_idx = int(self.scene_indices[idx])
        agent_ids = np.asarray(arrays["agent_ids"][scene_idx])
        x = np.asarray(arrays["x_agents"][scene_idx], dtype=np.float32)
        obs = np.asarray(arrays["obs_valid"][scene_idx], dtype=bool)
        y = np.asarray(arrays["y_agents"][scene_idx], dtype=np.float32)
        fut_valid = np.asarray(arrays["future_valid"][scene_idx], dtype=bool)
        scored = np.asarray(arrays["scored_agent_mask"][scene_idx], dtype=bool)
        lengths = np.asarray(arrays["agent_length"][scene_idx], dtype=np.float32)
        widths = np.asarray(arrays["agent_width"][scene_idx], dtype=np.float32)
        keep = scene_agent_indices(agent_ids, obs)
        scored_local = scored_local_indices(scored, keep)
        if forced_target_agent is not None:
            where = np.flatnonzero(keep == forced_target_agent)
            scored_local = where[:1].astype(np.int64)
        if scored_local.size == 0:
            scored_local = np.asarray([0], dtype=np.int64)
        n_obj = int(keep.size)
        n_ctr = int(scored_local.size)

        past_state = np.zeros((n_obj, self.history_len, 10), dtype=np.float32)
        past_mask = np.asarray(obs[keep], dtype=bool)
        for obj_i, src in enumerate(keep):
            past_state[obj_i, :, 0:2] = x[src, :, 0:2]
            past_state[obj_i, :, 3] = max(float(lengths[src]), 0.1)
            past_state[obj_i, :, 4] = max(float(widths[src]), 0.1)
            past_state[obj_i, :, 5] = self.vehicle_height
            past_state[obj_i, :, 6] = _heading_from_velocity(x[src, :, 2:4])
            past_state[obj_i, :, 7:9] = x[src, :, 2:4]
            past_state[obj_i, :, 9] = past_mask[obj_i].astype(np.float32)

        obj_trajs_single = self._pack_agent_features(past_state, past_mask)
        obj_trajs = np.repeat(obj_trajs_single[None], n_ctr, axis=0)
        obj_trajs_mask = np.repeat(past_mask[None], n_ctr, axis=0)
        obj_trajs_pos = np.repeat(past_state[None, :, :, 0:3], n_ctr, axis=0)
        obj_trajs_last_pos = np.zeros((n_ctr, n_obj, 3), dtype=np.float32)
        for obj_i in range(n_obj):
            valid_steps = np.flatnonzero(past_mask[obj_i])
            if valid_steps.size:
                obj_trajs_last_pos[:, obj_i] = past_state[obj_i, valid_steps[-1], 0:3]

        future_state_single = np.zeros((n_obj, self.future_len, 4), dtype=np.float32)
        future_mask_single = np.asarray(fut_valid[keep], dtype=bool)
        for obj_i, src in enumerate(keep):
            future_state_single[obj_i, :, 0:2] = y[src, :, 0:2]
            future_state_single[obj_i, :, 2:4] = y[src, :, 2:4]
        future_state = np.repeat(future_state_single[None], n_ctr, axis=0)
        future_mask = np.repeat(future_mask_single[None], n_ctr, axis=0)

        center_gt = future_state_single[scored_local].copy()
        center_gt_mask = future_mask_single[scored_local].copy()
        center_src = np.zeros((n_ctr, self.future_len, 10), dtype=np.float32)
        for center_i, obj_i in enumerate(scored_local):
            src = int(keep[obj_i])
            center_src[center_i, :, 0:2] = y[src, :, 0:2]
            center_src[center_i, :, 3] = max(float(lengths[src]), 0.1)
            center_src[center_i, :, 4] = max(float(widths[src]), 0.1)
            center_src[center_i, :, 5] = self.vehicle_height
            center_src[center_i, :, 6] = _heading_from_velocity(y[src, :, 2:4])
            center_src[center_i, :, 7:9] = y[src, :, 2:4]
            center_src[center_i, :, 9] = center_gt_mask[center_i].astype(np.float32)

        center_world = np.zeros((n_ctr, 10), dtype=np.float32)
        for center_i, obj_i in enumerate(scored_local):
            src = int(keep[obj_i])
            center_world[center_i, 0:2] = x[src, -1, 0:2]
            center_world[center_i, 3] = max(float(lengths[src]), 0.1)
            center_world[center_i, 4] = max(float(widths[src]), 0.1)
            center_world[center_i, 5] = self.vehicle_height
            center_world[center_i, 6] = _heading_from_velocity(x[src, -1:, 2:4])[0]
            center_world[center_i, 7:9] = x[src, -1, 2:4]
            center_world[center_i, 9] = 1.0

        map_data, map_mask, map_center = self._pseudo_map()
        return {
            "scenario_id": np.asarray([f"{self.dataset_name}_{self.split}_{scene_idx}_{i}" for i in scored_local]),
            "obj_trajs": obj_trajs,
            "obj_trajs_mask": obj_trajs_mask,
            "track_index_to_predict": scored_local.astype(np.int64),
            "obj_trajs_pos": obj_trajs_pos,
            "obj_trajs_last_pos": obj_trajs_last_pos,
            "obj_types": np.asarray([OBJECT_TYPE_VEHICLE] * n_obj),
            "obj_ids": agent_ids[keep].astype(np.int64),
            "center_objects_world": center_world,
            "center_objects_id": agent_ids[keep[scored_local]].astype(np.int64),
            "center_objects_type": np.asarray([OBJECT_TYPE_VEHICLE] * n_ctr),
            "obj_trajs_future_state": future_state,
            "obj_trajs_future_mask": future_mask,
            "center_gt_trajs": center_gt,
            "center_gt_trajs_mask": center_gt_mask,
            "center_gt_final_valid_idx": np.asarray([max(0, np.flatnonzero(m).max()) if m.any() else 0 for m in center_gt_mask], dtype=np.float32),
            "center_gt_trajs_src": center_src,
            "map_polylines": np.repeat(map_data[None], n_ctr, axis=0),
            "map_polylines_mask": np.repeat(map_mask[None], n_ctr, axis=0),
            "map_polylines_center": np.repeat(map_center[None], n_ctr, axis=0),
            "sample_index": np.full((n_ctr,), scene_idx, dtype=np.int64),
        }

    def _pack_agent_features(self, state: np.ndarray, valid: np.ndarray) -> np.ndarray:
        n = state.shape[0]
        onehot = np.zeros((n, self.history_len, 5), dtype=np.float32)
        onehot[:, :, 0] = 1.0
        time_embed = np.zeros((n, self.history_len, self.history_len + 1), dtype=np.float32)
        for t in range(self.history_len):
            time_embed[:, t, t] = 1.0
            time_embed[:, t, -1] = (t - self.history_len + 1) * self.dt
        heading_embed = np.stack([np.sin(state[:, :, 6]), np.cos(state[:, :, 6])], axis=-1).astype(np.float32)
        vel = state[:, :, 7:9]
        acc = np.zeros_like(vel)
        if self.history_len > 1:
            acc[:, 1:] = (vel[:, 1:] - vel[:, :-1]) / self.dt
            acc[:, 0] = acc[:, 1]
        out = np.concatenate([state[:, :, 0:6], onehot, time_embed, heading_embed, vel, acc], axis=-1).astype(np.float32)
        out[~valid] = 0.0
        return out

    def _pseudo_map(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = max(1, self.map_polylines)
        m = max(2, self.map_points_each_polyline)
        xs = np.linspace(-self.lane_half_length, self.lane_half_length, m, dtype=np.float32)
        offsets = (np.arange(p, dtype=np.float32) - (p - 1) * 0.5) * self.lane_width
        polylines = np.zeros((p, m, 9), dtype=np.float32)
        mask = np.ones((p, m), dtype=bool)
        for i, y in enumerate(offsets):
            polylines[i, :, 0] = xs
            polylines[i, :, 1] = y
            polylines[i, :, 3] = 1.0
            polylines[i, :, 6] = 1.0
            polylines[i, :, 7] = np.roll(xs, 1)
            polylines[i, 0, 7] = xs[0]
            polylines[i, :, 8] = y
        center = polylines[:, :, 0:3].mean(axis=1)
        return polylines, mask, center

    def collate_batch(self, batch_list: list[dict[str, Any]]) -> dict[str, Any]:
        return NeighFormerMTRDataset.collate_batch(self, batch_list)

    def describe(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "split": self.split,
            "data_dir": str(self.data_dir),
            "num_samples": len(self),
            "num_source_scenes": int(self.scene_indices.size),
            "history_len": self.history_len,
            "future_len": self.future_len,
            "agent_attr_dim": self.agent_attr_dim,
            "source": "multiagent",
            "target_agent_mode": self.target_agent_mode,
        }

    def channel_stats(self, n_samples: int = 256) -> dict[str, Any]:
        arrays = self.store.open()
        n = min(int(n_samples), len(self))
        retained = []
        scored = []
        for j in range(n):
            if self.target_agent_mode:
                assert self._target_scene_indices is not None
                scene_idx = int(self._target_scene_indices[j])
            else:
                scene_idx = int(self.scene_indices[j])
            retained.append(int(np.count_nonzero(arrays["agent_ids"][scene_idx] >= 0)))
            scored.append(int(np.count_nonzero(arrays["scored_agent_mask"][scene_idx])))
        return {
            "samples_inspected": n,
            "retained_agents_mean": float(np.mean(retained)) if retained else 0.0,
            "scored_agents_mean": float(np.mean(scored)) if scored else 0.0,
        }


def build_intention_points_from_multiagent(
    data_dir: str | Path,
    output_path: str | Path,
    num_modes: int = 6,
    max_samples: int = 50000,
) -> Path:
    """Build MTR intention anchors from scored multi-agent future endpoints."""
    data_dir = Path(data_dir)
    y = np.load(data_dir / "y_agents.npy", mmap_mode="r")
    scored = np.load(data_dir / "scored_agent_mask.npy", mmap_mode="r")
    future_valid = np.load(data_dir / "future_valid.npy", mmap_mode="r")
    n_scenes = min(int(y.shape[0]), int(max_samples))
    endpoints: list[np.ndarray] = []
    for i in range(n_scenes):
        mask = np.asarray(scored[i], dtype=bool) & np.asarray(future_valid[i, :, -1], dtype=bool)
        if np.any(mask):
            endpoints.append(np.asarray(y[i, mask, -1, 0:2], dtype=np.float32))
    if endpoints:
        pts = np.concatenate(endpoints, axis=0)
    else:
        pts = np.zeros((1, 2), dtype=np.float32)
    if pts.shape[0] >= num_modes:
        quantiles = np.linspace(0.0, 1.0, num_modes + 2, dtype=np.float32)[1:-1]
        order = np.argsort(pts[:, 0])
        anchors = pts[order[(quantiles * (pts.shape[0] - 1)).astype(np.int64)]]
    else:
        reps = int(np.ceil(num_modes / max(1, pts.shape[0])))
        anchors = np.tile(pts, (reps, 1))[:num_modes]
    payload = {OBJECT_TYPE_VEHICLE: anchors.astype(np.float32)}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(payload, f)
    return output_path
