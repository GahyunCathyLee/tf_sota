"""Lane graph cache utilities for the SIMPL adapter.

The cache stores map/lane geometry in the same global coordinate frame used by
the NeighFormer mmap files. Dataset items only crop and rotate/translate that
geometry into the sample-centric frame.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np


SEG_LENGTH = 10.0
SEG_N_NODE = 10
N_CTRL = SEG_N_NODE + 1


def empty_graph(lane_half_length: float = 120.0) -> dict[str, np.ndarray | int]:
    """Fallback straight lane used when no real map segment can be loaded."""
    n_lanes, n_points = 1, SEG_N_NODE
    xs = np.linspace(-lane_half_length, lane_half_length, n_points, dtype=np.float32)
    node_ctrs = np.stack([xs, np.zeros_like(xs)], axis=-1)[None]
    step = np.full_like(xs, 2 * lane_half_length / max(1, n_points - 1))
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


def save_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_cache(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def polyline_length(points: np.ndarray) -> tuple[np.ndarray, float]:
    if len(points) < 2:
        return np.zeros(len(points), dtype=np.float32), 0.0
    deltas = points[1:] - points[:-1]
    seg_lens = np.linalg.norm(deltas, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)]).astype(np.float32)
    return cum, float(cum[-1])


def interp_polyline(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    cum, total = polyline_length(points)
    if total <= 1e-6:
        return np.repeat(points[:1], len(distances), axis=0)
    distances = np.clip(distances.astype(np.float32), 0.0, total)
    out = np.zeros((len(distances), 2), dtype=np.float32)
    seg_idx = np.searchsorted(cum, distances, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(points) - 2)
    seg_start = cum[seg_idx]
    seg_end = cum[seg_idx + 1]
    alpha = (distances - seg_start) / np.maximum(seg_end - seg_start, 1e-6)
    out = points[seg_idx] * (1.0 - alpha[:, None]) + points[seg_idx + 1] * alpha[:, None]
    return out.astype(np.float32)


def segment_centerline(centerline: np.ndarray, seg_length: float = SEG_LENGTH) -> list[np.ndarray]:
    centerline = np.asarray(centerline, dtype=np.float32)
    cum, total = polyline_length(centerline)
    if total < 1.0:
        return []
    starts = np.arange(0.0, max(total - 1e-3, 0.0), seg_length, dtype=np.float32)
    segments = []
    for start in starts:
        end = min(float(start + seg_length), total)
        if end - float(start) < 1.0:
            continue
        d = np.linspace(float(start), end, N_CTRL, dtype=np.float32)
        segments.append(interp_polyline(centerline, d))
    return segments


def build_segment_arrays(
    centerlines: list[np.ndarray],
    left_flags: list[float] | None = None,
    right_flags: list[float] | None = None,
) -> dict[str, np.ndarray]:
    segments: list[np.ndarray] = []
    left: list[float] = []
    right: list[float] = []
    for idx, line in enumerate(centerlines):
        line_segments = segment_centerline(line)
        segments.extend(line_segments)
        left.extend([float(left_flags[idx]) if left_flags is not None else 0.0] * len(line_segments))
        right.extend([float(right_flags[idx]) if right_flags is not None else 0.0] * len(line_segments))
    if not segments:
        return {
            "segments": np.zeros((0, N_CTRL, 2), dtype=np.float32),
            "left": np.zeros((0,), dtype=np.float32),
            "right": np.zeros((0,), dtype=np.float32),
        }
    return {
        "segments": np.stack(segments, axis=0).astype(np.float32),
        "left": np.asarray(left, dtype=np.float32),
        "right": np.asarray(right, dtype=np.float32),
    }


def _anchor_graph_from_points(points: np.ndarray, left_flag: float, right_flag: float) -> tuple[np.ndarray, ...]:
    anchor = points.mean(axis=0)
    direction = points[-1] - points[0]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.array([1.0, 0.0], dtype=np.float32)
    else:
        direction = (direction / norm).astype(np.float32)
    cos_v, sin_v = float(direction[0]), float(direction[1])
    rot = np.array([[cos_v, -sin_v], [sin_v, cos_v]], dtype=np.float32)
    local = (points - anchor).dot(rot).astype(np.float32)
    node_ctrs = ((local[:-1] + local[1:]) * 0.5).astype(np.float32)
    node_vecs = (local[1:] - local[:-1]).astype(np.float32)
    left = np.full((SEG_N_NODE,), float(left_flag), dtype=np.float32)
    right = np.full((SEG_N_NODE,), float(right_flag), dtype=np.float32)
    return node_ctrs, node_vecs, anchor.astype(np.float32), direction.astype(np.float32), left, right


def graph_from_segments(
    segments: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray],
    left_flags: np.ndarray,
    right_flags: np.ndarray,
    radius: float = 120.0,
    max_segments: int = 192,
    fallback_half_length: float = 120.0,
) -> dict[str, np.ndarray | int]:
    if segments.size == 0:
        return empty_graph(fallback_half_length)
    scene_segments = transform(segments.reshape(-1, 2)).reshape(segments.shape)
    centers = scene_segments.mean(axis=1)
    distances = np.linalg.norm(centers, axis=1)
    keep = np.flatnonzero(distances <= float(radius))
    if keep.size == 0:
        return empty_graph(fallback_half_length)
    keep = keep[np.argsort(distances[keep])]
    if max_segments > 0:
        keep = keep[: int(max_segments)]

    node_ctrs, node_vecs, lane_ctrs, lane_vecs, left, right = [], [], [], [], [], []
    for idx in keep:
        nc, nv, lc, lv, lf, rf = _anchor_graph_from_points(
            scene_segments[idx], float(left_flags[idx]), float(right_flags[idx])
        )
        node_ctrs.append(nc)
        node_vecs.append(nv)
        lane_ctrs.append(lc)
        lane_vecs.append(lv)
        left.append(lf)
        right.append(rf)

    n_lanes = len(lane_ctrs)
    zeros2 = np.zeros((n_lanes, SEG_N_NODE, 2), dtype=np.float32)
    zeros = np.zeros((n_lanes, SEG_N_NODE), dtype=np.float32)
    return {
        "node_ctrs": np.stack(node_ctrs, axis=0).astype(np.float32),
        "node_vecs": np.stack(node_vecs, axis=0).astype(np.float32),
        "turn": zeros2.copy(),
        "control": zeros.copy(),
        "intersect": zeros.copy(),
        "left": np.stack(left, axis=0).astype(np.float32),
        "right": np.stack(right, axis=0).astype(np.float32),
        "lane_ctrs": np.stack(lane_ctrs, axis=0).astype(np.float32),
        "lane_vecs": np.stack(lane_vecs, axis=0).astype(np.float32),
        "num_nodes": SEG_N_NODE,
        "num_lanes": n_lanes,
    }


def rotate_to_heading(points: np.ndarray, ref_x: float, ref_y: float, heading_rad: float) -> np.ndarray:
    shifted = points - np.array([ref_x, ref_y], dtype=np.float32)
    c, s = math.cos(float(heading_rad)), math.sin(float(heading_rad))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return shifted.dot(rot).astype(np.float32)


def translate_to_origin(points: np.ndarray, ref_x: float, ref_y: float) -> np.ndarray:
    return (points - np.array([ref_x, ref_y], dtype=np.float32)).astype(np.float32)
