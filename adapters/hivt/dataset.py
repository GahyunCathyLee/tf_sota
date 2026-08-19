"""NeighFormer npy -> official HiVT TemporalData conversion."""

from __future__ import annotations

from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Dataset

from adapters.common import feature_mode_indices, feature_mode_names
from adapters.simpl.lane_graph import load_cache


def _heading_from_velocity(v: np.ndarray) -> np.ndarray:
    return np.arctan2(v[:, 1], v[:, 0]).astype(np.float32)


def _step_displacements(positions: np.ndarray, valid: np.ndarray) -> np.ndarray:
    disp = np.zeros_like(positions, dtype=np.float32)
    if positions.shape[0] > 1:
        pair_valid = valid[1:] & valid[:-1]
        disp[1:] = np.where(pair_valid[:, None], positions[1:] - positions[:-1], 0.0)
    return disp


class NeighFormerHiVTDataset(Dataset):
    """HiVT-compatible dataset backed by NeighFormer mmap npy files."""

    def __init__(
        self,
        data_dir: str | Path,
        indices: np.ndarray,
        dataset_name: str,
        feature_mode: str,
        split: str,
        lane_half_length: float = 120.0,
        lane_cache_root: str | Path | None = None,
        lane_radius: float = 120.0,
        lane_max_segments: int = 192,
    ) -> None:
        super().__init__(None, None, None)
        self.data_dir = Path(data_dir)
        self.sample_indices = np.asarray(indices, dtype=np.int64)
        self.dataset_name = dataset_name
        self.feature_mode = feature_mode
        self.split = split
        self.lane_half_length = float(lane_half_length)
        self.lane_cache_root = Path(lane_cache_root) if lane_cache_root else None
        self.lane_radius = float(lane_radius)
        self.lane_max_segments = int(lane_max_segments)
        self.nb_feature_indices = np.asarray(feature_mode_indices(feature_mode), dtype=np.int64)
        self.nb_feature_names = feature_mode_names(feature_mode)
        self.node_dim = len(self.nb_feature_indices)
        self.edge_dim = 2
        self._arrays: dict[str, np.ndarray] | None = None
        self._recording_cache: dict[int, dict[str, Any] | None] = {}
        self._warned_lane_cache = False

        x_ego = np.load(self.data_dir / "x_ego.npy", mmap_mode="r")
        x_nb = np.load(self.data_dir / "x_nb.npy", mmap_mode="r")
        y = np.load(self.data_dir / "y.npy", mmap_mode="r")
        self.n_samples_total = int(x_ego.shape[0])
        self.history_len = int(x_ego.shape[1])
        self.future_len = int(y.shape[1])
        self.num_steps = self.history_len + self.future_len
        self.max_neighbors = int(x_nb.shape[2])
        if int(x_ego.shape[2]) != 6:
            raise ValueError(f"Expected x_ego[..., 6], got {x_ego.shape}")
        if int(x_nb.shape[3]) < int(self.nb_feature_indices.max()) + 1:
            raise ValueError(
                f"x_nb has {x_nb.shape[3]} channels; {feature_mode} needs index "
                f"{int(self.nb_feature_indices.max())}"
            )

    def _ensure_open(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            arrays = {
                "x_ego": np.load(self.data_dir / "x_ego.npy", mmap_mode="r"),
                "x_nb": np.load(self.data_dir / "x_nb.npy", mmap_mode="r"),
                "nb_mask": np.load(self.data_dir / "nb_mask.npy", mmap_mode="r"),
                "y": np.load(self.data_dir / "y.npy", mmap_mode="r"),
            }
            for name in ("meta_recordingId", "meta_trackId", "meta_frame", "x_last_abs"):
                path = self.data_dir / f"{name}.npy"
                if path.exists():
                    arrays[name] = np.load(path, mmap_mode="r")
            self._arrays = arrays
        return self._arrays

    def len(self) -> int:
        return int(self.sample_indices.shape[0])

    def get(self, idx: int):
        from utils import TemporalData

        arrays = self._ensure_open()
        real_idx = int(self.sample_indices[idx])
        th, tf = self.history_len, self.future_len
        ego = np.asarray(arrays["x_ego"][real_idx], dtype=np.float32)
        nb = np.asarray(arrays["x_nb"][real_idx], dtype=np.float32)
        mask = np.asarray(arrays["nb_mask"][real_idx], dtype=bool)
        fut = np.asarray(arrays["y"][real_idx], dtype=np.float32)
        slots = np.flatnonzero(mask.any(axis=0))
        num_nodes = 1 + int(slots.size)

        positions = np.zeros((num_nodes, th + tf, 2), dtype=np.float32)
        features = np.zeros((num_nodes, th, self.node_dim), dtype=np.float32)
        padding_mask = np.ones((num_nodes, th + tf), dtype=bool)

        positions[0, :th] = ego[:, 0:2]
        positions[0, th:] = fut
        features[0, :, 0:2] = _step_displacements(ego[:, 0:2], np.ones(th, dtype=bool))
        features[0, :, 2:6] = ego[:, 2:6]
        if self.node_dim > 6:
            features[0, :, 6:] = -1.0
        padding_mask[0] = False

        for offset, slot in enumerate(slots, start=1):
            nb_hist = nb[:, slot]
            slot_mask = mask[:, slot]
            hist_pos = ego[:, 0:2] + nb_hist[:, 0:2]
            positions[offset, :th] = hist_pos
            features[offset, :, 0:2] = _step_displacements(hist_pos, slot_mask)
            features[offset, :, 2:6] = np.stack(
                [
                    ego[:, 2] + nb_hist[:, 2],
                    ego[:, 3] + nb_hist[:, 3],
                    ego[:, 4] + nb_hist[:, 4],
                    ego[:, 5] + nb_hist[:, 5],
                ],
                axis=-1,
            )
            if self.node_dim > 6:
                features[offset, :, 6:] = nb_hist[:, [8, 9]]
            features[offset, ~slot_mask] = 0.0
            padding_mask[offset, :th] = ~slot_mask

        bos_mask = np.zeros((num_nodes, th), dtype=bool)
        bos_mask[:, 0] = ~padding_mask[:, 0]
        bos_mask[:, 1:th] = padding_mask[:, : th - 1] & ~padding_mask[:, 1:th]
        rotate_angles = _heading_from_velocity(features[:, -1, 2:4])

        if num_nodes > 1:
            edge_index = torch.tensor(list(permutations(range(num_nodes), 2)), dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
        lane = self._lane_features(real_idx, arrays, positions[:, th - 1])
        meta = self._meta(real_idx, arrays)

        return TemporalData(
            x=torch.from_numpy(features),
            positions=torch.from_numpy(positions),
            edge_index=edge_index,
            y=torch.from_numpy(positions[:, th:] - positions[:, th - 1 : th]),
            num_nodes=num_nodes,
            padding_mask=torch.from_numpy(padding_mask),
            bos_mask=torch.from_numpy(bos_mask),
            rotate_angles=torch.from_numpy(rotate_angles),
            lane_vectors=lane["lane_vectors"],
            is_intersections=lane["is_intersections"],
            turn_directions=lane["turn_directions"],
            traffic_controls=lane["traffic_controls"],
            lane_actor_index=lane["lane_actor_index"],
            lane_actor_vectors=lane["lane_actor_vectors"],
            seq_id=real_idx,
            agent_index=0,
            av_index=0,
            sample_index=torch.tensor([real_idx], dtype=torch.long),
            recording_id=meta["recording_id"],
            track_id=meta["track_id"],
            frame_id=meta["frame_id"],
        )

    def _meta(self, real_idx: int, arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {
            "recording_id": torch.tensor(
                [int(arrays["meta_recordingId"][real_idx])] if "meta_recordingId" in arrays else [-1],
                dtype=torch.long,
            ),
            "track_id": torch.tensor(
                [int(arrays["meta_trackId"][real_idx])] if "meta_trackId" in arrays else [-1],
                dtype=torch.long,
            ),
            "frame_id": torch.tensor(
                [int(arrays["meta_frame"][real_idx])] if "meta_frame" in arrays else [-1],
                dtype=torch.long,
            ),
        }

    def _warn_lane_cache_once(self, message: str) -> None:
        if not self._warned_lane_cache:
            print(f"[WARN] HiVT lane graph fallback: {message}")
            self._warned_lane_cache = True

    def _load_recording_cache(self, recording_id: int) -> dict[str, Any] | None:
        if recording_id in self._recording_cache:
            return self._recording_cache[recording_id]
        if self.lane_cache_root is None:
            self._recording_cache[recording_id] = None
            return None
        candidates = [
            self.lane_cache_root / f"recording_{recording_id:02d}.pkl",
            self.lane_cache_root / f"recording_{recording_id}.pkl",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            self._recording_cache[recording_id] = None
            return None
        cache = load_cache(path)
        self._recording_cache[recording_id] = cache
        return cache

    def _lane_features(
        self,
        real_idx: int,
        arrays: dict[str, np.ndarray],
        node_positions: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        if self.dataset_name == "highD" and self.lane_cache_root is not None:
            if "meta_recordingId" not in arrays or "x_last_abs" not in arrays:
                self._warn_lane_cache_once("meta_recordingId.npy or x_last_abs.npy is missing")
            else:
                cache = self._load_recording_cache(int(arrays["meta_recordingId"][real_idx]))
                if cache is not None:
                    ref = np.asarray(arrays["x_last_abs"][real_idx], dtype=np.float32)
                    segments = np.asarray(cache.get("segments", np.zeros((0, 11, 2), dtype=np.float32)), dtype=np.float32)
                    if segments.size:
                        return self._lane_features_from_segments(segments - ref.reshape(1, 1, 2), node_positions)
        return self._pseudo_lane_features(node_positions)

    def _lane_features_from_segments(
        self,
        segments: np.ndarray,
        node_positions: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        centers = segments.mean(axis=1)
        distances = np.linalg.norm(centers, axis=1)
        keep = np.flatnonzero(distances <= self.lane_radius)
        if keep.size == 0:
            return self._pseudo_lane_features(node_positions)
        keep = keep[np.argsort(distances[keep])]
        if self.lane_max_segments > 0:
            keep = keep[: self.lane_max_segments]
        seg = segments[keep]
        lane_pos = seg[:, :-1].reshape(-1, 2).astype(np.float32)
        lane_vec2 = (seg[:, 1:] - seg[:, :-1]).reshape(-1, 2).astype(np.float32)
        return self._pack_lane_features(lane_pos, lane_vec2, node_positions)

    def _pseudo_lane_features(self, node_positions: np.ndarray) -> dict[str, torch.Tensor]:
        xs = np.linspace(-self.lane_half_length, self.lane_half_length, 25, dtype=np.float32)
        lane_pos = np.stack([xs[:-1], np.zeros_like(xs[:-1])], axis=-1)
        lane_vec2 = np.stack([np.diff(xs), np.zeros(xs.shape[0] - 1, dtype=np.float32)], axis=-1)
        return self._pack_lane_features(lane_pos, lane_vec2, node_positions)

    def _pack_lane_features(
        self,
        lane_pos: np.ndarray,
        lane_vec2: np.ndarray,
        node_positions: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        lane_vectors = np.zeros((lane_vec2.shape[0], self.node_dim), dtype=np.float32)
        lane_vectors[:, :2] = lane_vec2
        lane_actor_src, lane_actor_dst, lane_actor_vec = [], [], []
        valid_nodes = np.arange(node_positions.shape[0])
        for lane_idx, pos in enumerate(lane_pos):
            rel = pos.reshape(1, 2) - node_positions[valid_nodes]
            mask = np.linalg.norm(rel, axis=1) < self.lane_radius
            for dst, vec in zip(valid_nodes[mask], rel[mask]):
                lane_actor_src.append(lane_idx)
                lane_actor_dst.append(int(dst))
                lane_actor_vec.append(vec.astype(np.float32))
        if not lane_actor_src:
            lane_actor_src = [0]
            lane_actor_dst = [0]
            lane_actor_vec = [lane_pos[0] - node_positions[0]]
        edge_index = torch.tensor([lane_actor_src, lane_actor_dst], dtype=torch.long)
        return {
            "lane_vectors": torch.from_numpy(lane_vectors),
            "is_intersections": torch.zeros(lane_vectors.shape[0], dtype=torch.uint8),
            "turn_directions": torch.zeros(lane_vectors.shape[0], dtype=torch.uint8),
            "traffic_controls": torch.zeros(lane_vectors.shape[0], dtype=torch.uint8),
            "lane_actor_index": edge_index,
            "lane_actor_vectors": torch.as_tensor(np.asarray(lane_actor_vec, dtype=np.float32)),
        }

    def channel_stats(self, n_samples: int = 256) -> dict[str, Any]:
        n = min(n_samples, self.len())
        rows: list[np.ndarray] = []
        arrays = self._ensure_open()
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

    def describe(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "data_dir": str(self.data_dir),
            "dataset": self.dataset_name,
            "feature_mode": self.feature_mode,
            "num_samples": self.len(),
            "num_samples_available": self.n_samples_total,
            "history_len": self.history_len,
            "future_len": self.future_len,
            "max_neighbors": self.max_neighbors,
            "node_dim": self.node_dim,
            "edge_dim": self.edge_dim,
            "neighbor_indices": [int(v) for v in self.nb_feature_indices],
            "neighbor_names": self.nb_feature_names,
            "lane_cache_root": str(self.lane_cache_root) if self.lane_cache_root else None,
            "lane_radius": self.lane_radius,
            "lane_max_segments": self.lane_max_segments,
            "uses_real_lane_graph": bool(
                self.dataset_name == "highD" and self.lane_cache_root and self.lane_cache_root.exists()
            ),
        }
