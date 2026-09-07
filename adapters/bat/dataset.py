"""NeighFormer npy -> BAT input tensors.

The official BAT model consumes NGSIM-style history/future tensors, a 13x3
social grid, and behavior graph indicators over 39 grid cells.  NeighFormer
already stores ego-centered highD/exiD windows, so this adapter keeps the BAT
model intact and supplies the missing grid/behavior features from the available
ego and neighbour kinematics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from adapters.common import feature_mode_indices, feature_mode_names

EGO_GRID_INDEX = 20


def cart_to_polar(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    out = np.zeros_like(xy, dtype=np.float32)
    out[..., 0] = np.sqrt(np.square(xy[..., 0]) + np.square(xy[..., 1]))
    out[..., 1] = np.arctan2(xy[..., 1], xy[..., 0])
    return out


def polar_to_cart(polar: Any):
    import torch

    r = polar[..., 0]
    phi = polar[..., 1]
    return torch.stack((r * torch.cos(phi), r * torch.sin(phi)), dim=-1)


def _diff(xy: np.ndarray) -> np.ndarray:
    out = np.zeros_like(xy, dtype=np.float32)
    if xy.shape[0] > 1:
        out[1:] = xy[1:] - xy[:-1]
    return out


def _diff_agents_time(xy: np.ndarray) -> np.ndarray:
    out = np.zeros_like(xy, dtype=np.float32)
    if xy.shape[1] > 1:
        out[:, 1:] = xy[:, 1:] - xy[:, :-1]
    return out


def _one_hot(index: int, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.float32)
    out[int(np.clip(index, 0, size - 1))] = 1.0
    return out


class NeighFormerBATDataset:
    """BAT-compatible dataset backed by canonical NeighFormer mmap npy files."""

    def __init__(
        self,
        data_dir: str | Path,
        indices: np.ndarray,
        dataset_name: str,
        feature_mode: str,
        split: str,
        grid_size: tuple[int, int] = (13, 3),
        enc_size: int = 64,
        polar: bool = True,
        longitudinal_cell: float = 15.0,
        lane_width: float = 3.7,
        neighbor_distance: float = 100.0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.sample_indices = np.asarray(indices, dtype=np.int64)
        self.dataset_name = dataset_name
        self.feature_mode = feature_mode
        self.split = split
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.max_vehicles = self.grid_size[0] * self.grid_size[1]
        self.enc_size = int(enc_size)
        self.polar = bool(polar)
        self.longitudinal_cell = float(longitudinal_cell)
        self.lane_width = float(lane_width)
        self.neighbor_distance = float(neighbor_distance)
        self.nb_feature_indices = np.asarray(feature_mode_indices(feature_mode), dtype=np.int64)
        self.nb_feature_names = feature_mode_names(feature_mode)
        self._arrays: dict[str, np.ndarray] | None = None

        x_ego = np.load(self.data_dir / "x_ego.npy", mmap_mode="r")
        x_nb = np.load(self.data_dir / "x_nb.npy", mmap_mode="r")
        y = np.load(self.data_dir / "y.npy", mmap_mode="r")
        self.n_samples_total = int(x_ego.shape[0])
        self.history_len = int(x_ego.shape[1])
        self.future_len = int(y.shape[1])
        self.raw_max_neighbors = int(x_nb.shape[2])
        if int(x_ego.shape[2]) != 6:
            raise ValueError(f"Expected x_ego[..., 6], got {x_ego.shape}")
        if int(x_nb.shape[3]) < int(self.nb_feature_indices.max()) + 1:
            raise ValueError(
                f"x_nb has {x_nb.shape[3]} channels; {feature_mode} needs index "
                f"{int(self.nb_feature_indices.max())}"
            )
        if self.max_vehicles != 39:
            raise ValueError("The released BAT model assumes a 13x3 social grid, i.e. 39 vehicles")

    def _ensure_open(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            arrays = {
                "x_ego": np.load(self.data_dir / "x_ego.npy", mmap_mode="r"),
                "x_nb": np.load(self.data_dir / "x_nb.npy", mmap_mode="r"),
                "nb_mask": np.load(self.data_dir / "nb_mask.npy", mmap_mode="r"),
                "y": np.load(self.data_dir / "y.npy", mmap_mode="r"),
            }
            for name in ("y_vel", "meta_recordingId", "meta_trackId", "meta_frame"):
                path = self.data_dir / f"{name}.npy"
                if path.exists():
                    arrays[name] = np.load(path, mmap_mode="r")
            self._arrays = arrays
        return self._arrays

    def __len__(self) -> int:
        return int(self.sample_indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        arrays = self._ensure_open()
        real_idx = int(self.sample_indices[idx])
        ego = np.asarray(arrays["x_ego"][real_idx], dtype=np.float32)
        nb = np.asarray(arrays["x_nb"][real_idx], dtype=np.float32)
        mask = np.asarray(arrays["nb_mask"][real_idx], dtype=bool)
        fut_cart = np.asarray(arrays["y"][real_idx], dtype=np.float32)

        hist_cart = ego[:, 0:2].astype(np.float32)
        hist = cart_to_polar(hist_cart) if self.polar else hist_cart
        fut = cart_to_polar(fut_cart) if self.polar else fut_cart
        hist_relative = _diff(hist_cart)
        va = ego[:, 2:4].astype(np.float32)
        lane = self._ego_lane_feature(hist_cart)
        cls = np.ones((self.history_len, 1), dtype=np.float32)

        grid = self._place_neighbors(hist_cart, ego[:, 2:4].astype(np.float32), nb, mask)
        feature_matrix, behavior = self._behavior_graph(grid["cart"], grid["valid"])
        lat_enc, lon_enc = self._maneuver_labels(fut_cart)

        return {
            "hist": hist,
            "fut": fut,
            "hist_relative": hist_relative,
            "lat_enc": lat_enc,
            "lon_enc": lon_enc,
            "va": va,
            "lane": lane,
            "cls": cls,
            "nbrs": cart_to_polar(grid["cart"]) if self.polar else grid["cart"],
            "nbrs_relative_self": _diff_agents_time(grid["cart"]),
            "nbrs_relative_nbrs": grid["ref_nbrs"],
            "nbrsva": grid["va"],
            "nbrslane": grid["lane"],
            "nbrscls": grid["cls"],
            "nbr_valid": grid["valid"],
            "feature_matrix": feature_matrix,
            "behavior": behavior,
            "target_cart": fut_cart,
            "sample_index": np.array(real_idx, dtype=np.int64),
        }

    def _ego_lane_feature(self, hist_cart: np.ndarray) -> np.ndarray:
        if self.feature_mode == "dimI":
            return np.zeros((self.history_len, 1), dtype=np.float32)
        lateral = np.round(hist_cart[:, 1:2] / max(self.lane_width, 1.0))
        return np.clip(lateral, -1.0, 1.0).astype(np.float32)

    def _place_neighbors(
        self,
        ego_hist: np.ndarray,
        ego_vel: np.ndarray,
        nb: np.ndarray,
        mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        gx, gy = self.grid_size
        cart = np.zeros((self.max_vehicles, self.history_len, 2), dtype=np.float32)
        ref_nbrs = np.zeros_like(cart)
        va = np.zeros_like(cart)
        lane = np.zeros((self.max_vehicles, self.history_len, 1), dtype=np.float32)
        cls = np.zeros((self.max_vehicles, self.history_len, 1), dtype=np.float32)
        valid = np.zeros(self.max_vehicles, dtype=bool)

        used = {EGO_GRID_INDEX}
        slots = np.flatnonzero(mask.any(axis=0))
        for slot in slots:
            slot_mask = mask[:, slot]
            if not slot_mask.any():
                continue
            last_valid = int(np.flatnonzero(slot_mask)[-1])
            dx, dy = float(nb[last_valid, slot, 0]), float(nb[last_valid, slot, 1])
            cell = self._grid_cell(dx, dy, used)
            if cell is None:
                continue
            used.add(cell)
            valid[cell] = True
            rel_at_each_step = nb[:, slot, 0:2].astype(np.float32)
            cart[cell] = ego_hist + rel_at_each_step
            ref_nbrs[cell] = rel_at_each_step
            va[cell] = ego_vel + nb[:, slot, 2:4].astype(np.float32)
            if self.feature_mode == "dimI":
                cls[cell, :, 0] = nb[:, slot, 8]
                lane[cell, :, 0] = nb[:, slot, 9]
            else:
                cls[cell, :, 0] = 1.0
                lane[cell, :, 0] = np.clip(np.round(rel_at_each_step[:, 1] / max(self.lane_width, 1.0)), -1.0, 1.0)
            cart[cell] = np.where(slot_mask[:, None], cart[cell], 0.0)
            ref_nbrs[cell] = np.where(slot_mask[:, None], ref_nbrs[cell], 0.0)
            va[cell] = np.where(slot_mask[:, None], va[cell], 0.0)
            lane[cell] = np.where(slot_mask[:, None], lane[cell], 0.0)
            cls[cell] = np.where(slot_mask[:, None], cls[cell], 0.0)
        return {"cart": cart, "ref_nbrs": ref_nbrs, "va": va, "lane": lane, "cls": cls, "valid": valid}

    def _grid_cell(self, dx: float, dy: float, used: set[int]) -> int | None:
        gx, gy = self.grid_size
        col = gx // 2 + int(np.clip(np.round(dx / max(self.longitudinal_cell, 1.0)), -(gx // 2), gx // 2))
        row = gy // 2 + int(np.clip(np.round(dy / max(self.lane_width, 1.0)), -(gy // 2), gy // 2))
        preferred = int(row * gx + col)
        candidates = [preferred]
        for radius in range(1, gx):
            for dc in (-radius, radius):
                c = int(np.clip(col + dc, 0, gx - 1))
                candidates.append(int(row * gx + c))
        for cell in candidates:
            if cell not in used:
                return cell
        return None

    def _behavior_graph(self, nbr_cart: np.ndarray, nbr_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = nbr_cart.copy()
        positions[EGO_GRID_INDEX] = 0.0
        valid = nbr_valid.copy()
        valid[EGO_GRID_INDEX] = True
        t_len = positions.shape[1]
        feature = np.zeros((t_len, self.max_vehicles, self.max_vehicles), dtype=np.float32)
        centrality = np.zeros((t_len, self.max_vehicles, 3), dtype=np.float32)
        for t in range(t_len):
            active = np.flatnonzero(valid)
            if active.size == 0:
                continue
            xy = positions[active, t, :]
            dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
            connected = dist < self.neighbor_distance
            weights = np.where(connected, np.exp(-dist), 0.0).astype(np.float32)
            signed = -weights
            feature[t][np.ix_(active, active)] = signed
            if active.size > 1:
                no_diag = weights.copy()
                np.fill_diagonal(no_diag, 0.0)
                degree = no_diag.sum(axis=1) / float(active.size - 1)
                denom = np.maximum(dist.sum(axis=1), 1.0e-6)
                closeness = (active.size - 1) / denom
                eigen = self._principal_vector(no_diag)
                centrality[t, active, 0] = degree
                centrality[t, active, 1] = closeness
                centrality[t, active, 2] = eigen
            else:
                centrality[t, active, :] = 1.0
        diff = np.zeros_like(centrality)
        if t_len > 1:
            diff[1:] = centrality[1:] - centrality[:-1]
        behavior = np.concatenate([diff, centrality], axis=-1).astype(np.float32)
        return feature, behavior

    @staticmethod
    def _principal_vector(weights: np.ndarray) -> np.ndarray:
        if weights.size == 0 or not np.any(weights):
            return np.ones(weights.shape[0], dtype=np.float32)
        v = np.ones(weights.shape[0], dtype=np.float32) / math.sqrt(weights.shape[0])
        for _ in range(8):
            v = weights @ v
            norm = float(np.linalg.norm(v))
            if norm <= 1.0e-6:
                break
            v = v / norm
        return np.abs(v).astype(np.float32)

    @staticmethod
    def _maneuver_labels(fut_cart: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        final = fut_cart[-1]
        lat_idx = 1
        if final[1] > 0.75:
            lat_idx = 2
        elif final[1] < -0.75:
            lat_idx = 0
        first_speed = float(np.linalg.norm(fut_cart[min(1, len(fut_cart) - 1)] - fut_cart[0]))
        last_speed = float(np.linalg.norm(fut_cart[-1] - fut_cart[max(0, len(fut_cart) - 2)]))
        lon_idx = 1
        if last_speed - first_speed > 0.5:
            lon_idx = 2
        elif first_speed - last_speed > 0.5:
            lon_idx = 0
        return _one_hot(lat_idx, 3), _one_hot(lon_idx, 3)

    def collate_fn(self, samples: list[dict[str, np.ndarray]]) -> dict[str, Any]:
        import torch

        bsz = len(samples)
        gx, gy = self.grid_size
        t_h = self.history_len
        t_f = self.future_len
        neighbor_cells: list[tuple[int, int, dict[str, np.ndarray]]] = []
        for b, sample in enumerate(samples):
            for cell in np.flatnonzero(sample["nbr_valid"]):
                neighbor_cells.append((b, int(cell), sample))
        n_nbr = max(1, len(neighbor_cells))

        def zeros(*shape: int):
            return torch.zeros(*shape, dtype=torch.float32)

        def tensor_from(array: np.ndarray):
            return torch.from_numpy(np.asarray(array).copy())

        batch = {
            "hist": zeros(t_h, bsz, 2),
            "nbrs": zeros(t_h, n_nbr, 2),
            "hist_relative": zeros(t_h, bsz, 2),
            "mask": torch.zeros(bsz, gy, gx, self.enc_size, dtype=torch.bool),
            "lat_enc": zeros(bsz, 3),
            "lon_enc": zeros(bsz, 3),
            "fut": zeros(t_f, bsz, 2),
            "op_mask": torch.ones(t_f, bsz, 2, dtype=torch.float32),
            "va": zeros(t_h, bsz, 2),
            "nbrsva": zeros(t_h, n_nbr, 2),
            "lane": zeros(t_h, bsz, 1),
            "nbrslane": zeros(t_h, n_nbr, 1),
            "dis": zeros(t_h, bsz, 1),
            "nbrsdis": zeros(t_h, n_nbr, 1),
            "cls": zeros(t_h, bsz, 1),
            "nbrscls": zeros(t_h, n_nbr, 1),
            "map_positions": zeros(max(1, len(neighbor_cells)), 2),
            "nbrs_ref_self": zeros(t_h, n_nbr, 2),
            "nbrs_ref_nbrs": zeros(t_h, n_nbr, 2),
            "feature_matrix": zeros(t_h, bsz, self.max_vehicles, self.max_vehicles),
            "behavior": zeros(t_h, bsz, self.max_vehicles, 6),
            "target": zeros(bsz, t_f, 2),
            "sample_index": torch.zeros(bsz, dtype=torch.long),
        }

        for b, sample in enumerate(samples):
            for key in ("hist", "hist_relative", "fut", "va", "lane", "cls"):
                dest = key if key != "fut" else "fut"
                batch[dest][:, b] = tensor_from(sample[key])
            batch["lat_enc"][b] = tensor_from(sample["lat_enc"])
            batch["lon_enc"][b] = tensor_from(sample["lon_enc"])
            batch["feature_matrix"][:, b] = tensor_from(sample["feature_matrix"])
            batch["behavior"][:, b] = tensor_from(sample["behavior"])
            batch["target"][b] = tensor_from(sample["target_cart"])
            batch["sample_index"][b] = int(sample["sample_index"])

        for n, (b, cell, sample) in enumerate(neighbor_cells):
            row, col = divmod(cell, gx)
            batch["mask"][b, row, col, :] = True
            batch["map_positions"][n] = torch.tensor([row, col], dtype=torch.float32)
            for src, dst in (
                ("nbrs", "nbrs"),
                ("nbrs_relative_self", "nbrs_ref_self"),
                ("nbrs_relative_nbrs", "nbrs_ref_nbrs"),
                ("nbrsva", "nbrsva"),
                ("nbrslane", "nbrslane"),
                ("nbrscls", "nbrscls"),
            ):
                batch[dst][:, n] = tensor_from(sample[src][cell])
        return batch

    def describe(self, num_samples: int | None = None) -> dict[str, Any]:
        n = len(self) if num_samples is None else min(int(num_samples), len(self))
        return {
            "split": self.split,
            "data_dir": str(self.data_dir),
            "dataset": self.dataset_name,
            "feature_mode": self.feature_mode,
            "num_samples": int(len(self)),
            "num_samples_available": self.n_samples_total,
            "history_len": self.history_len,
            "future_len": self.future_len,
            "raw_max_neighbors": self.raw_max_neighbors,
            "bat_grid_size": list(self.grid_size),
            "bat_max_vehicles": self.max_vehicles,
            "polar_coordinates": self.polar,
            "neighbor_indices": [int(v) for v in self.nb_feature_indices],
            "neighbor_names": self.nb_feature_names,
            "samples_described": n,
            "dimI_mapping": "neighbor dim -> BAT class scalar, I -> BAT lane scalar" if self.feature_mode == "dimI" else "disabled",
            "behavior_features": "degree/closeness/eigenvector centrality plus first differences from NeighFormer kinematics",
        }

    def channel_stats(self, n_samples: int = 256) -> dict[str, Any]:
        arrays = self._ensure_open()
        rows = []
        for real_idx in self.sample_indices[: min(n_samples, len(self.sample_indices))]:
            mask = np.asarray(arrays["nb_mask"][int(real_idx)], dtype=bool)
            nb = np.asarray(arrays["x_nb"][int(real_idx)], dtype=np.float32)
            if mask.any():
                rows.append(nb[mask][:, self.nb_feature_indices])
        if not rows:
            return {}
        stacked = np.concatenate(rows, axis=0)
        return {
            "samples_inspected": min(n_samples, len(self.sample_indices)),
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


def write_preprocess_manifest(
    processed_dir: str | Path,
    dataset_name: str,
    feature_mode: str,
    split_reports: dict[str, dict[str, Any]],
) -> Path:
    root = Path(processed_dir) / dataset_name / feature_mode
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    path.write_text(json.dumps(split_reports, indent=2), encoding="utf-8")
    return path
