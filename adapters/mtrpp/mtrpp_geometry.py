"""Geometry helpers for the highD/exiD MTR++ adapter."""

from __future__ import annotations

import math

import numpy as np


def wrap_angle_np(angle: np.ndarray) -> np.ndarray:
    return ((angle + math.pi) % (2.0 * math.pi) - math.pi).astype(np.float32)


def rotate_points_np(points: np.ndarray, heading: np.ndarray | float) -> np.ndarray:
    """Rotate xy vectors by heading radians.

    ``points`` may have arbitrary leading dimensions ending in 2. ``heading`` is
    broadcast against those leading dimensions.
    """

    pts = np.asarray(points, dtype=np.float32)
    h = np.asarray(heading, dtype=np.float32)
    c = np.cos(h)
    s = np.sin(h)
    x = pts[..., 0]
    y = pts[..., 1]
    return np.stack([x * c - y * s, x * s + y * c], axis=-1).astype(np.float32)


def latest_token_pose(
    positions: np.ndarray,
    velocities: np.ndarray,
    valid: np.ndarray,
    fallback_heading: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return np.zeros(2, dtype=np.float32), np.float32(fallback_heading), 0
    t = int(valid_idx[-1])
    pos = np.asarray(positions[t], dtype=np.float32)
    vel = np.asarray(velocities[t], dtype=np.float32)
    if float(np.linalg.norm(vel)) > 1.0e-4:
        heading = np.float32(np.arctan2(vel[1], vel[0]))
    else:
        heading = np.float32(fallback_heading)
    return pos, heading, t


def localize_agent_state(
    state: np.ndarray,
    valid: np.ndarray,
    token_pos: np.ndarray,
    token_heading: float,
) -> np.ndarray:
    """Convert an MTR 10D agent state polyline into its own token frame."""

    local = np.asarray(state, dtype=np.float32).copy()
    local[..., 0:2] = rotate_points_np(local[..., 0:2] - token_pos[None, :], -token_heading)
    local[..., 6] = wrap_angle_np(local[..., 6] - token_heading)
    local[..., 7:9] = rotate_points_np(local[..., 7:9], -token_heading)
    local[~valid] = 0.0
    return local


def localize_map_polylines(polylines: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Represent each map polyline in a map-token local frame.

    Returns local polylines, token global centers ``[P,3]``, and token headings
    ``[P]``. highD/exiD only provide pseudo-lane geometry here; no HD-map
    semantics are synthesized.
    """

    local = np.asarray(polylines, dtype=np.float32).copy()
    centers = np.zeros((polylines.shape[0], 3), dtype=np.float32)
    headings = np.zeros((polylines.shape[0],), dtype=np.float32)
    for i in range(polylines.shape[0]):
        valid_idx = np.flatnonzero(mask[i])
        if valid_idx.size == 0:
            continue
        xy = polylines[i, valid_idx, 0:2]
        center_xy = xy.mean(axis=0).astype(np.float32)
        tangent = xy[-1] - xy[0]
        heading = np.float32(np.arctan2(tangent[1], tangent[0])) if np.linalg.norm(tangent) > 1.0e-4 else np.float32(0.0)
        centers[i, 0:2] = center_xy
        headings[i] = heading
        local[i, :, 0:2] = rotate_points_np(polylines[i, :, 0:2] - center_xy[None, :], -heading)
        if local.shape[-1] >= 9:
            local[i, :, 7:9] = rotate_points_np(polylines[i, :, 7:9] - center_xy[None, :], -heading)
        local[i, ~mask[i]] = 0.0
    return local.astype(np.float32), centers, headings
