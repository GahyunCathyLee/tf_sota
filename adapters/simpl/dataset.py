"""NeighFormer npy -> SIMPL batch conversion."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from adapters.common import feature_mode_indices, feature_mode_names


def _get_cos(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    v1_norm = v1.norm(dim=-1)
    v2_norm = v2.norm(dim=-1)
    return (v1[..., 0] * v2[..., 0] + v1[..., 1] * v2[..., 1]) / (v1_norm * v2_norm + 1e-10)


def _get_sin(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    v1_norm = v1.norm(dim=-1)
    v2_norm = v2.norm(dim=-1)
    return (v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]) / (v1_norm * v2_norm + 1e-10)


def build_rpe(ctrs: torch.Tensor, vecs: torch.Tensor, radius: float = 100.0) -> dict[str, torch.Tensor | None]:
    d_pos = (ctrs.unsqueeze(0) - ctrs.unsqueeze(1)).norm(dim=-1)
    pos_rpe = (d_pos * 2.0 / radius).unsqueeze(0)
    cos_a1 = _get_cos(vecs.unsqueeze(0), vecs.unsqueeze(1))
    sin_a1 = _get_sin(vecs.unsqueeze(0), vecs.unsqueeze(1))
    v_pos = ctrs.unsqueeze(0) - ctrs.unsqueeze(1)
    cos_a2 = _get_cos(vecs.unsqueeze(0), v_pos)
    sin_a2 = _get_sin(vecs.unsqueeze(0), v_pos)
    return {"scene": torch.cat([torch.stack([cos_a1, sin_a1, cos_a2, sin_a2]), pos_rpe], dim=0),
            "scene_mask": None}


class NeighFormerSIMPLDataset(Dataset):
    """SIMPL-compatible dataset backed by NeighFormer mmap npy files."""

    def __init__(
        self,
        data_dir: str | Path,
        indices: np.ndarray,
        dataset_name: str,
        feature_mode: str,
        split: str,
        lane_half_length: float = 120.0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.sample_indices = np.asarray(indices, dtype=np.int64)
        self.dataset_name = dataset_name
        self.feature_mode = feature_mode
        self.split = split
        self.lane_half_length = float(lane_half_length)
        self.nb_feature_indices = np.asarray(feature_mode_indices(feature_mode), dtype=np.int64)
        self.nb_feature_names = feature_mode_names(feature_mode)
        self.actor_feature_dim = len(self.nb_feature_indices)
        self._arrays: dict[str, np.ndarray] | None = None

        x_ego = np.load(self.data_dir / "x_ego.npy", mmap_mode="r")
        x_nb = np.load(self.data_dir / "x_nb.npy", mmap_mode="r")
        y = np.load(self.data_dir / "y.npy", mmap_mode="r")
        self.n_samples_total = int(x_ego.shape[0])
        self.history_len = int(x_ego.shape[1])
        self.future_len = int(y.shape[1])
        self.max_neighbors = int(x_nb.shape[2])
        if int(x_ego.shape[2]) != 6:
            raise ValueError(f"Expected x_ego[..., 6], got {x_ego.shape}")
        if int(x_nb.shape[3]) < int(self.nb_feature_indices.max()) + 1:
            raise ValueError(
                f"x_nb has {x_nb.shape[3]} channels; {feature_mode} needs "
                f"index {int(self.nb_feature_indices.max())}"
            )

    def _ensure_open(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            arrays = {
                "x_ego": np.load(self.data_dir / "x_ego.npy", mmap_mode="r"),
                "x_nb": np.load(self.data_dir / "x_nb.npy", mmap_mode="r"),
                "nb_mask": np.load(self.data_dir / "nb_mask.npy", mmap_mode="r"),
                "y": np.load(self.data_dir / "y.npy", mmap_mode="r"),
            }
            for name in ("meta_recordingId", "meta_trackId", "meta_frame"):
                p = self.data_dir / f"{name}.npy"
                if p.exists():
                    arrays[name] = np.load(p, mmap_mode="r")
            self._arrays = arrays
        return self._arrays

    def __len__(self) -> int:
        return int(self.sample_indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        arrays = self._ensure_open()
        real_idx = int(self.sample_indices[idx])
        ego = np.array(arrays["x_ego"][real_idx], dtype=np.float32, copy=True)
        nb = np.array(arrays["x_nb"][real_idx], dtype=np.float32, copy=True)
        mask = np.array(arrays["nb_mask"][real_idx], dtype=bool, copy=True)
        fut = np.array(arrays["y"][real_idx], dtype=np.float32, copy=True)
        slots = np.flatnonzero(mask.any(axis=0))
        n_agents = 1 + int(slots.size)
        th, tf, feat_dim = self.history_len, self.future_len, self.actor_feature_dim

        trajs_obs = np.zeros((n_agents, th, feat_dim), dtype=np.float32)
        trajs_obs[0, :, :6] = ego
        if feat_dim > 6:
            trajs_obs[0, :, 6:] = -1.0
        pad_obs = np.zeros((n_agents, th), dtype=np.float32)
        pad_obs[0] = 1.0

        centers = np.zeros((n_agents, 2), dtype=np.float32)
        vecs = np.zeros((n_agents, 2), dtype=np.float32)
        centers[0] = ego[-1, 0:2]
        vecs[0] = ego[-1, 2:4]
        for offset, slot in enumerate(slots, start=1):
            nb_hist = nb[:, slot]
            slot_mask = mask[:, slot]
            trajs_obs[offset, :, :6] = np.stack(
                [
                    ego[:, 0] + nb_hist[:, 0],
                    ego[:, 1] + nb_hist[:, 1],
                    ego[:, 2] + nb_hist[:, 2],
                    ego[:, 3] + nb_hist[:, 3],
                    ego[:, 4] + nb_hist[:, 4],
                    ego[:, 5] + nb_hist[:, 5],
                ],
                axis=-1,
            )
            if feat_dim > 6:
                trajs_obs[offset, :, 6:] = nb_hist[:, [8, 9]]
            trajs_obs[offset, ~slot_mask] = 0.0
            pad_obs[offset] = slot_mask.astype(np.float32)
            centers[offset] = trajs_obs[offset, -1, 0:2]
            vecs[offset] = trajs_obs[offset, -1, 2:4]

        trajs_fut = np.zeros((n_agents, tf, 2), dtype=np.float32)
        trajs_fut[0] = fut
        pad_fut = np.zeros((n_agents, tf), dtype=np.float32)
        pad_fut[0] = 1.0

        graph = self._pseudo_lane_graph()
        scene_ctrs = torch.cat([torch.from_numpy(centers), torch.from_numpy(graph["lane_ctrs"])], dim=0)
        scene_vecs = torch.cat([torch.from_numpy(vecs), torch.from_numpy(graph["lane_vecs"])], dim=0)
        rpe = build_rpe(scene_ctrs, scene_vecs)

        meta = {
            "recordingId": int(arrays["meta_recordingId"][real_idx]) if "meta_recordingId" in arrays else -1,
            "trackId": int(arrays["meta_trackId"][real_idx]) if "meta_trackId" in arrays else -1,
            "t0_frame": int(arrays["meta_frame"][real_idx]) if "meta_frame" in arrays else -1,
        }
        return {
            "SEQ_ID": f"{self.dataset_name}_{real_idx}",
            "SAMPLE_INDEX": real_idx,
            "META": meta,
            "TRAJS_OBS": trajs_obs,
            "TRAJS_FUT": trajs_fut,
            "PAD_OBS": pad_obs,
            "PAD_FUT": pad_fut,
            "TRAJS_CTRS": centers,
            "TRAJS_VECS": vecs,
            "LANE_GRAPH": graph,
            "RPE": rpe,
        }

    def _pseudo_lane_graph(self) -> dict[str, np.ndarray | int]:
        l = self.lane_half_length
        n_lanes, n_points = 1, 10
        xs = np.linspace(-l, l, n_points, dtype=np.float32)
        node_ctrs = np.stack([xs, np.zeros_like(xs)], axis=-1)[None]
        step = np.full_like(xs, 2 * l / max(1, n_points - 1))
        node_vecs = np.stack([step, np.zeros_like(step)], axis=-1)[None]
        zeros2 = np.zeros((n_lanes, n_points, 2), dtype=np.float32)
        zeros = np.zeros((n_lanes, n_points), dtype=np.float32)
        return {
            "node_ctrs": node_ctrs,
            "node_vecs": node_vecs,
            "turn": zeros2.copy(),
            "control": zeros.copy(),
            "intersect": zeros.copy(),
            "left": zeros.copy(),
            "right": zeros.copy(),
            "lane_ctrs": np.array([[0.0, 0.0]], dtype=np.float32),
            "lane_vecs": np.array([[1.0, 0.0]], dtype=np.float32),
            "num_nodes": n_points,
            "num_lanes": n_lanes,
        }

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        data: dict[str, Any] = {"BATCH_SIZE": len(batch)}
        for key in batch[0].keys():
            data[key] = [x[key] for x in batch]
        data["TRAJS_OBS"] = [torch.from_numpy(x) for x in data["TRAJS_OBS"]]
        data["TRAJS_FUT"] = [torch.from_numpy(x) for x in data["TRAJS_FUT"]]
        data["PAD_OBS"] = [torch.from_numpy(x) for x in data["PAD_OBS"]]
        data["PAD_FUT"] = [torch.from_numpy(x) for x in data["PAD_FUT"]]
        data["LANE_GRAPH"] = [{k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in g.items()}
                              for g in data["LANE_GRAPH"]]
        data["ACTORS"], data["ACTOR_IDCS"] = self.actor_gather(data["TRAJS_OBS"])
        data["LANES"], data["LANE_IDCS"] = self.graph_gather(data["LANE_GRAPH"])
        return data

    def actor_gather(self, actors: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        actor_idcs = []
        count = 0
        for a in actors:
            actor_idcs.append(torch.arange(count, count + a.shape[0], dtype=torch.long))
            count += a.shape[0]
        return torch.cat([x.transpose(1, 2) for x in actors], dim=0), actor_idcs

    def graph_gather(self, graphs: list[dict[str, Any]]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        lane_idcs = []
        count = 0
        for g in graphs:
            lane_idcs.append(torch.arange(count, count + int(g["num_lanes"]), dtype=torch.long))
            count += int(g["num_lanes"])
        lanes = torch.cat(
            [
                torch.cat(
                    [
                        g["node_ctrs"],
                        g["node_vecs"],
                        g["turn"],
                        g["control"].unsqueeze(2),
                        g["intersect"].unsqueeze(2),
                        g["left"].unsqueeze(2),
                        g["right"].unsqueeze(2),
                    ],
                    dim=-1,
                )
                for g in graphs
            ],
            dim=0,
        )
        return lanes, lane_idcs

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
            "actor_feature_dim": self.actor_feature_dim,
            "neighbor_indices": [int(v) for v in self.nb_feature_indices],
            "neighbor_names": self.nb_feature_names,
            "pseudo_lanes_per_sample": 1,
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
