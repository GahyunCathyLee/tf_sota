"""NeighFormer npy -> MTP-GO scene-graph conversion.

Feature mapping (see ``adapters/common_schema.md`` and the upstream MTP-GO
preprocessing convention this mirrors):

    node 0        = ego (target vehicle)
        [x, y, xVelocity, yVelocity, xAcceleration, yAcceleration]  from x_ego
        positions are relative to the ego position at the last history step
    node j+1      = neighbour slot k (only slots present at some history step)
        [dx, dy, dvx, dvy, dax, day]                                from x_nb
        ego-relative, exactly the channels upstream ``highD-imp`` uses

    all feature modes -> 6 native node channels only

Edges are rebuilt per history step exactly like upstream graph preprocessing:
fully connected (self-loops included) over the nodes present at that step, with
a single scalar edge feature = Euclidean distance in the ego-relative frame.

The canonical NeighFormer arrays only store ego-neighbour interaction values,
so ``use_importance`` is intentionally unsupported here: MTP-GO graph edges also
include neighbour-neighbour pairs. Use the persistent multi-agent arrays, where
pairwise ``I_ij`` can be computed from physical agent state.

Targets: canonical NeighFormer arrays contain only the ego future. If
``y_nb.npy``/``y_nb_mask.npy`` are present, fixed-ID neighbour futures are also
loaded and ``tar_real_mask`` marks every valid target agent step. Future graph
topology is still frozen at the last observed history graph.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Node target layout: [x, y, vx, vy, ax, ay]. Motion models only consume the
# first ``n_states`` of these (4 for the default 2Xnode model).
TARGET_CHANNELS = 6
NATIVE_NODE_FEATURES = 6


class MTPGoData(Data):
    """PyG graph data with global sample ids kept unchanged while batching."""

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "sample_index":
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def build_edges(pos: np.ndarray, present: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Fully connected edges (with self-loops) over present nodes.

    pos     : (n_nodes, 2) ego-relative positions
    present : (n_nodes,) bool
    """
    idx = np.flatnonzero(present)
    if idx.size == 0:  # ego is always present, but stay defensive
        idx = np.zeros(1, dtype=np.int64)
    p = pos[idx]
    src = np.repeat(idx, idx.size)
    dst = np.tile(idx, idx.size)
    dist = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1).reshape(-1)
    edge_index = torch.from_numpy(np.stack([src, dst]).astype(np.int64))
    edge_attr = torch.from_numpy(dist.astype(np.float32)).unsqueeze(1)
    return edge_index, edge_attr


class NeighFormerGraphDataset(Dataset):
    """PyG dataset producing MTP-GO scene graphs from the NeighFormer npy files."""

    def __init__(
        self,
        data_dir: Path,
        indices: np.ndarray,
        feature_mode: str,
        split: str = "train",
        use_importance: bool = False,
    ) -> None:
        super().__init__(None, None, None)
        self.data_dir = Path(data_dir)
        self.sample_indices = np.asarray(indices, dtype=np.int64)
        self.feature_mode = feature_mode
        self.use_importance = bool(use_importance)
        if self.use_importance:
            raise ValueError(
                "MTP-GO use_importance=true requires multi-agent arrays because canonical "
                "x_nb only stores ego-neighbour I, not arbitrary graph-edge I_ij."
            )
        self.split = split
        self.n_node_features = NATIVE_NODE_FEATURES
        self.edge_feature_dim = 1
        self._arrays: dict[str, np.ndarray] | None = None

        # Read shapes once up front (cheap with mmap) so callers can validate.
        head = np.load(self.data_dir / "x_ego.npy", mmap_mode="r")
        nb = np.load(self.data_dir / "x_nb.npy", mmap_mode="r")
        fut = np.load(self.data_dir / "y.npy", mmap_mode="r")
        self.n_samples_total = int(head.shape[0])
        self.history_len = int(head.shape[1])
        self.ego_channels = int(head.shape[2])
        self.max_neighbors = int(nb.shape[2])
        self.future_len = int(fut.shape[1])
        self.has_y_vel = (self.data_dir / "y_vel.npy").exists()
        self.has_y_acc = (self.data_dir / "y_acc.npy").exists()
        self.has_neighbor_future = (self.data_dir / "y_nb.npy").exists() and (self.data_dir / "y_nb_mask.npy").exists()

        if self.ego_channels != 6:
            raise ValueError(f"Expected 6 ego channels, got {self.ego_channels}")

    # ---------------------------------------------------------------- loading
    def _ensure_open(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            arrays = {
                "x_ego": np.load(self.data_dir / "x_ego.npy", mmap_mode="r"),
                "x_nb": np.load(self.data_dir / "x_nb.npy", mmap_mode="r"),
                "nb_mask": np.load(self.data_dir / "nb_mask.npy", mmap_mode="r"),
                "y": np.load(self.data_dir / "y.npy", mmap_mode="r"),
            }
            if self.has_y_vel:
                arrays["y_vel"] = np.load(self.data_dir / "y_vel.npy", mmap_mode="r")
            if self.has_y_acc:
                arrays["y_acc"] = np.load(self.data_dir / "y_acc.npy", mmap_mode="r")
            if self.has_neighbor_future:
                arrays["y_nb"] = np.load(self.data_dir / "y_nb.npy", mmap_mode="r")
                arrays["y_nb_mask"] = np.load(self.data_dir / "y_nb_mask.npy", mmap_mode="r")
            self._arrays = arrays
        return self._arrays

    def len(self) -> int:
        return int(self.sample_indices.shape[0])

    # ------------------------------------------------------------------ items
    def get(self, idx: int) -> Data:
        arrays = self._ensure_open()
        i = int(self.sample_indices[idx])

        ego = np.asarray(arrays["x_ego"][i], dtype=np.float32)        # (T_h, 6)
        nb = np.asarray(arrays["x_nb"][i], dtype=np.float32)          # (T_h, K, 13)
        mask = np.asarray(arrays["nb_mask"][i], dtype=bool)           # (T_h, K)

        slots = np.flatnonzero(mask.any(axis=0))                      # kept neighbour slots
        n_nodes = 1 + int(slots.size)
        t_h, t_f = self.history_len, self.future_len

        # ---- node history features
        x = np.zeros((n_nodes, t_h, NATIVE_NODE_FEATURES), dtype=np.float32)
        x[0, :, :NATIVE_NODE_FEATURES] = ego[:, :NATIVE_NODE_FEATURES]
        if slots.size:
            nb_sel = nb[:, slots, :NATIVE_NODE_FEATURES]              # (T_h, n_slots, 6)
            nb_sel = np.where(mask[:, slots][..., None], nb_sel, 0.0)
            x[1:] = np.transpose(nb_sel, (1, 0, 2))

        # ---- per-step graphs
        pos = np.zeros((t_h, n_nodes, 2), dtype=np.float32)
        present = np.ones((t_h, n_nodes), dtype=bool)
        if slots.size:
            pos[:, 1:, :] = nb[:, slots, 0:2]
            present[:, 1:] = mask[:, slots]

        hist_ei: list[torch.Tensor] = []
        hist_ef: list[torch.Tensor] = []
        for t in range(t_h):
            ei, ef = build_edges(pos[t], present[t])
            hist_ei.append(ei)
            hist_ef.append(ef)

        # Future topology is unknown -> freeze the last observed graph.
        fut_ei = [hist_ei[-1]] * t_f
        fut_ef = [hist_ef[-1]] * t_f

        # ---- targets
        y = np.zeros((n_nodes, t_f, TARGET_CHANNELS), dtype=np.float32)
        y[0, :, 0:2] = arrays["y"][i]
        if self.has_y_vel:
            y[0, :, 2:4] = arrays["y_vel"][i]
        if self.has_y_acc:
            y[0, :, 4:6] = arrays["y_acc"][i]
        real_mask = np.zeros((n_nodes, t_f, TARGET_CHANNELS), dtype=bool)
        real_mask[0] = True
        if self.has_neighbor_future and "y_nb" in arrays and "y_nb_mask" in arrays:
            y_nb = np.asarray(arrays["y_nb"][i], dtype=np.float32)
            y_nb_mask = np.asarray(arrays["y_nb_mask"][i], dtype=bool)
            n_target_channels = min(TARGET_CHANNELS, int(y_nb.shape[-1]))
            for offset, slot in enumerate(slots, start=1):
                if slot >= y_nb.shape[1]:
                    continue
                valid = y_nb_mask[:, slot]
                y[offset, valid, :n_target_channels] = y_nb[valid, slot, :n_target_channels]
                real_mask[offset, valid, :n_target_channels] = True

        # ---- static per-node attributes expected by upstream modules.
        # Vehicle type / length / width are not part of the NeighFormer schema,
        # so they are constant placeholders. They only reach the model when
        # init_static or n_ode_static is enabled (both default to False).
        v_type = torch.zeros(n_nodes, 2, dtype=torch.float32)
        v_type[:, 0] = 1.0
        dim = torch.zeros(n_nodes, 2, dtype=torch.float32)
        cf = torch.full((n_nodes,), 3, dtype=torch.long)

        return MTPGoData(
            x=torch.from_numpy(x),
            edge_index=hist_ei,
            edge_features=hist_ef,
            y=torch.from_numpy(y),
            tar_edge_index=fut_ei,
            tar_edge_features=fut_ef,
            tar_real_mask=torch.from_numpy(real_mask),
            cf=cf,
            dim=dim,
            v_type=v_type,
            sample_index=torch.tensor([i], dtype=torch.long),
        )

    # ------------------------------------------------------------------ misc
    def channel_stats(self, n_samples: int = 256) -> dict[str, Any]:
        """Per-channel stats over the neighbour node rows of a few scene graphs.

        Used to show that only native MTP-GO node features reach the model.
        """
        n = min(n_samples, self.len())
        rows: list[np.ndarray] = []
        for j in range(n):
            data = self.get(j)
            x = data.x.numpy()
            if x.shape[0] > 1:
                rows.append(x[1:].reshape(-1, x.shape[-1]))
        if not rows:
            return {}
        stacked = np.concatenate(rows, axis=0)
        names = ["x|dx", "y|dy", "vx|dvx", "vy|dvy", "ax|dax", "ay|day"]
        return {
            "samples_inspected": n,
            "neighbor_rows": int(stacked.shape[0]),
            "consumed_node_features": names,
            "consumed_edge_features": ["distance"],
            "excluded_proposed_node_channels": ["lc_state", "lit", "lis", "gate", "I_x", "I_y", "dim", "I"],
            "per_channel": {
                names[c]: {
                    "min": float(stacked[:, c].min()),
                    "max": float(stacked[:, c].max()),
                    "mean": float(stacked[:, c].mean()),
                    "nonzero_fraction": float((stacked[:, c] != 0).mean()),
                }
                for c in range(stacked.shape[1])
            },
        }

    def describe(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "data_dir": str(self.data_dir),
            "feature_mode": self.feature_mode,
            "num_samples": self.len(),
            "num_samples_available": self.n_samples_total,
            "history_len": self.history_len,
            "future_len": self.future_len,
            "max_neighbors": self.max_neighbors,
            "ego_channels": self.ego_channels,
            "node_feature_channels": self.n_node_features,
            "node_feature_names": ["x", "y", "xV", "yV", "xA", "yA"],
            "edge_feature_channels": self.edge_feature_dim,
            "edge_feature_names": ["distance"],
            "use_importance": self.use_importance,
            "excluded_proposed_node_channels": ["lc_state", "lit", "lis", "gate", "I_x", "I_y", "dim", "I"],
            "y_vel_used": self.has_y_vel,
            "y_acc_used": self.has_y_acc,
            "neighbor_future_used": self.has_neighbor_future,
        }


def estimate_dt(data_dir: Path, n_samples: int = 512) -> float | None:
    """Estimate the sampling interval from ego displacement / velocity.

    Used only as a sanity check against the configured ``dt``.
    """
    ego = np.load(Path(data_dir) / "x_ego.npy", mmap_mode="r")
    n = min(n_samples, int(ego.shape[0]))
    if n == 0 or ego.shape[1] < 2:
        return None
    sample = np.asarray(ego[:n], dtype=np.float64)
    disp = np.linalg.norm(np.diff(sample[..., 0:2], axis=1), axis=-1)   # (n, T_h-1)
    speed = np.linalg.norm(sample[:, :-1, 2:4], axis=-1)                # (n, T_h-1)
    ok = speed > 1.0
    if not ok.any():
        return None
    return float(np.median(disp[ok] / speed[ok]))
