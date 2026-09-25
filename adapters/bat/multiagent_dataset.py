"""BAT batch conversion for persistent multi-agent scene arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from adapters.bat.dataset import EGO_GRID_INDEX, NeighFormerBATDataset, _diff, _diff_agents_time, _one_hot, cart_to_polar
from adapters.multiagent_common import MultiAgentArrays, meta_dict, scene_agent_indices, scored_local_indices
from data.multiagent.interaction import InteractionConfig, build_pair_features


class MultiAgentBATDataset:
    """BAT-compatible target samples backed by full scene-level arrays.

    A dataset item is one scene; its collate function flattens all scored agents
    in the scene into BAT's usual target-vehicle batch dimension.
    """

    def __init__(
        self,
        data_dir: str | Path,
        dataset_name: str,
        feature_mode: str,
        split: str,
        indices: np.ndarray | None = None,
        grid_size: tuple[int, int] = (13, 3),
        enc_size: int = 64,
        polar: bool = True,
        longitudinal_cell: float = 15.0,
        lane_width: float = 3.7,
        neighbor_distance: float = 100.0,
        use_importance: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.feature_mode = feature_mode
        self.split = split
        self.store = MultiAgentArrays(self.data_dir)
        self.scene_indices = (
            np.arange(self.store.num_scenes, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.max_vehicles = self.grid_size[0] * self.grid_size[1]
        self.enc_size = int(enc_size)
        self.polar = bool(polar)
        self.longitudinal_cell = float(longitudinal_cell)
        self.lane_width = float(lane_width)
        self.neighbor_distance = float(neighbor_distance)
        self.history_len = self.store.history_len
        self.future_len = self.store.future_len
        self.raw_max_neighbors = self.store.max_agents
        self.use_importance_feature = bool(use_importance)
        self.behavior_feature_dim = 7 if self.use_importance_feature else 6
        self.nb_feature_names = ["x", "y", "vx", "vy", "ax", "ay"] + (["I"] if self.use_importance_feature else [])
        self.nb_feature_indices = np.arange(len(self.nb_feature_names), dtype=np.int64)
        if self.max_vehicles != 39:
            raise ValueError("The released BAT model assumes a 13x3 social grid, i.e. 39 vehicles")

    def __len__(self) -> int:
        return int(self.scene_indices.size)

    def __getitem__(self, idx: int) -> list[dict[str, np.ndarray]]:
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

        pair = self._pair_importance(arrays, scene_idx) if self.use_importance_feature else None
        samples = []
        for target_local in scored_local.tolist():
            target_orig = int(keep[int(target_local)])
            if not bool(fut_valid[target_orig].any()):
                continue
            samples.append(
                self._target_sample(
                    scene_idx,
                    target_orig,
                    keep,
                    x_all,
                    obs,
                    y_all,
                    fut_valid,
                    pair,
                    arrays,
                )
            )
        if not samples:
            target_orig = int(keep[int(scored_local[0])])
            samples.append(
                self._target_sample(scene_idx, target_orig, keep, x_all, obs, y_all, fut_valid, pair, arrays)
            )
        return samples

    def _target_sample(
        self,
        scene_idx: int,
        target_orig: int,
        keep: np.ndarray,
        x_all: np.ndarray,
        obs: np.ndarray,
        y_all: np.ndarray,
        fut_valid: np.ndarray,
        pair: tuple[np.ndarray, np.ndarray] | None,
        arrays: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        target_hist = np.asarray(x_all[target_orig], dtype=np.float32)
        ref_pos = target_hist[-1, 0:2].astype(np.float32)
        target_pos = (target_hist[:, 0:2] - ref_pos).astype(np.float32)
        hist = cart_to_polar(target_pos) if self.polar else target_pos.astype(np.float32)
        fut_cart = np.asarray(y_all[target_orig, :, 0:2] - ref_pos, dtype=np.float32)
        fut = cart_to_polar(fut_cart) if self.polar else fut_cart
        lane = self._ego_lane_feature(target_pos)
        cls = np.ones((self.history_len, 1), dtype=np.float32)
        valid_future = np.asarray(fut_valid[target_orig], dtype=bool)

        grid = self._place_neighbors(target_orig, keep, target_hist, ref_pos, x_all, obs, pair)
        feature_matrix, behavior = self._behavior_graph(grid["cart"], grid["valid"], grid["importance"])
        lat_enc, lon_enc = self._maneuver_labels(fut_cart)

        return {
            "hist": hist,
            "fut": fut,
            "hist_relative": _diff(target_pos),
            "lat_enc": lat_enc,
            "lon_enc": lon_enc,
            "va": target_hist[:, 2:4].astype(np.float32),
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
            "target_valid": valid_future,
            "sample_index": np.array(scene_idx, dtype=np.int64),
            "target_agent_index": np.array(target_orig, dtype=np.int64),
            "meta": meta_dict(arrays, scene_idx),
        }

    def _place_neighbors(
        self,
        target_orig: int,
        keep: np.ndarray,
        target_hist: np.ndarray,
        ref_pos: np.ndarray,
        x_all: np.ndarray,
        obs: np.ndarray,
        pair: tuple[np.ndarray, np.ndarray] | None,
    ) -> dict[str, np.ndarray]:
        cart = np.zeros((self.max_vehicles, self.history_len, 2), dtype=np.float32)
        ref_nbrs = np.zeros_like(cart)
        va = np.zeros_like(cart)
        lane = np.zeros((self.max_vehicles, self.history_len, 1), dtype=np.float32)
        cls = np.zeros((self.max_vehicles, self.history_len, 1), dtype=np.float32)
        importance = np.zeros((self.max_vehicles, self.history_len, 1), dtype=np.float32)
        valid = np.zeros(self.max_vehicles, dtype=bool)
        used = {EGO_GRID_INDEX}

        target_pos = (target_hist[:, 0:2] - ref_pos).astype(np.float32)
        target_vel = target_hist[:, 2:4]
        for src_orig in keep.tolist():
            src_orig = int(src_orig)
            if src_orig == int(target_orig):
                continue
            slot_mask = np.asarray(obs[src_orig], dtype=bool)
            if not bool(slot_mask.any()):
                continue
            last_valid = int(np.flatnonzero(slot_mask)[-1])
            src_pos = np.asarray(x_all[src_orig, :, 0:2] - ref_pos, dtype=np.float32)
            rel_hist = np.asarray(src_pos - target_pos, dtype=np.float32)
            dx, dy = float(rel_hist[last_valid, 0]), float(rel_hist[last_valid, 1])
            cell = self._grid_cell(dx, dy, used)
            if cell is None:
                continue
            used.add(cell)
            valid[cell] = True
            cart[cell] = src_pos
            ref_nbrs[cell] = rel_hist
            va[cell] = x_all[src_orig, :, 2:4]
            cls[cell, :, 0] = 1.0
            lane[cell, :, 0] = np.clip(np.round(rel_hist[:, 1] / max(self.lane_width, 1.0)), -1.0, 1.0)
            if pair is not None:
                pair_features, pair_valid = pair
                pvalid = pair_valid[target_orig, src_orig]
                importance[cell, pvalid, 0] = pair_features[target_orig, src_orig, pvalid, 9]

            cart[cell] = np.where(slot_mask[:, None], cart[cell], 0.0)
            ref_nbrs[cell] = np.where(slot_mask[:, None], ref_nbrs[cell], 0.0)
            va[cell] = np.where(slot_mask[:, None], va[cell], 0.0)
            lane[cell] = np.where(slot_mask[:, None], lane[cell], 0.0)
            cls[cell] = np.where(slot_mask[:, None], cls[cell], 0.0)
            importance[cell] = np.where(slot_mask[:, None], importance[cell], 0.0)
        return {
            "cart": cart,
            "ref_nbrs": ref_nbrs,
            "va": va,
            "lane": lane,
            "cls": cls,
            "importance": importance,
            "valid": valid,
        }

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

    def collate_fn(self, scene_samples: list[list[dict[str, np.ndarray]]]) -> dict[str, Any]:
        flat = [sample for scene in scene_samples for sample in scene]
        return NeighFormerBATDataset.collate_fn(self, flat)

    _ego_lane_feature = NeighFormerBATDataset._ego_lane_feature
    _grid_cell = NeighFormerBATDataset._grid_cell
    _behavior_graph = NeighFormerBATDataset._behavior_graph
    _principal_vector = staticmethod(NeighFormerBATDataset._principal_vector)
    _maneuver_labels = staticmethod(NeighFormerBATDataset._maneuver_labels)

    def describe(self, num_samples: int | None = None) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "split": self.split,
            "data_dir": str(self.data_dir),
            "source": "multiagent",
            "num_scenes": len(self),
            "num_scenes_available": self.store.num_scenes,
            "history_len": self.history_len,
            "future_len": self.future_len,
            "raw_max_agents": self.raw_max_neighbors,
            "bat_grid_size": list(self.grid_size),
            "bat_max_vehicles": self.max_vehicles,
            "polar_coordinates": self.polar,
            "feature_mode": self.feature_mode,
            "use_importance": self.use_importance_feature,
            "behavior_feature_dim": self.behavior_feature_dim,
            "neighbor_names": self.nb_feature_names,
            "target_scope": "scored agents",
            "I_mapping": "pairwise I[target, source, t] -> appended BAT behavior scalar"
            if self.use_importance_feature
            else "disabled",
        }

    def channel_stats(self, n_samples: int = 256) -> dict[str, Any]:
        arrays = self.store.open()
        n = min(int(n_samples), len(self))
        retained = []
        scored = []
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
        }
