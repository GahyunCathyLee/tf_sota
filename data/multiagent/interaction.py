#!/usr/bin/env python3
"""Actual-experiment directed interaction builder for persistent physical agents."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


DatasetName = Literal["highD", "exiD"]

NB_DIM = 10
HIGHD_ACTUAL_LIT_DENOM_EPS = 0.0
DEFAULT_LIS_MODE = "7"
DEFAULT_GAMMA = 0.5
DEFAULT_TOP_N = 3

# Slot order: P, F, LP, LA, LF, RP, RA, RF.
SLOT_NAMES = ("P", "F", "LP", "LA", "LF", "RP", "RA", "RF")
TOPN_SLOT_PRIORITY = {slot: rank for rank, slot in enumerate((0, 2, 5, 1, 4, 7, 3, 6))}

LIS_BINS = {
    "3": {"cuts": [-5.8639, 4.9525], "vals": [-1.0, 0.0, 1.0]},
    "5": {"cuts": [-13.7033, -3.0238, 2.2735, 13.0957], "vals": [-2.0, -1.0, 0.0, 1.0, 2.0]},
    "7": {
        "cuts": [-18.7902, -8.2922, -1.9963, 1.3381, 7.3744, 18.5267],
        "vals": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
    },
    "9": {
        "cuts": [-22.7661, -12.1209, -5.8639, -1.4829, 0.9127, 4.9525, 11.4115, 22.7702],
        "vals": [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
    },
}

SLOT_WEIGHTS_BY_LANE_LEVEL = np.asarray(
    [
        [0.4255, 0.0336, 0.0000, 0.0000, 0.0000, 0.4574, 0.0119, 0.1190],
        [0.4805, 0.0002, 0.0000, 0.0000, 0.0000, 0.3803, 0.0234, 0.1839],
        [0.4784, 0.0373, 0.3344, 0.0343, 0.2050, 0.0000, 0.0000, 0.0000],
    ],
    dtype=np.float32,
)

SLOT_WEIGHTS_PRE_LC = np.asarray(
    [
        [0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.6253, 0.2663, 0.3117],
        [0.0072, 0.0263, 0.0006, 0.0000, 0.0000, 0.3970, 0.3776, 0.5494],
        [0.0183, 0.1326, 0.6745, 0.5179, 0.2365, 0.0000, 0.0000, 0.0000],
        [0.0381, 0.0233, 0.5755, 0.3548, 0.4799, 0.0000, 0.0000, 0.0000],
    ],
    dtype=np.float32,
)

SLOT_WEIGHTS_POST_LC = np.asarray(
    [
        [0.0460, 0.3983, 0.0000, 0.0023, 0.0762, 0.2338, 0.2022, 0.3281],
        [0.1036, 0.0851, 0.4832, 0.0540, 0.3810, 0.0013, 0.0000, 0.0002],
        [0.6018, 0.3591, 0.0115, 0.0013, 0.0099, 0.1709, 0.0069, 0.0014],
        [0.2618, 0.0000, 0.0036, 0.0000, 0.0000, 0.6545, 0.2032, 0.1449],
    ],
    dtype=np.float32,
)

GLOBAL_SLOT_WEIGHTS = np.asarray([0.4944, 0.0411, 0.0935, 0.0074, 0.0002, 0.5559, 0.0000, 0.1179], dtype=np.float32)

LC_LEVEL_TO_GROUP = {
    (0, 1): 0,
    (0, 2): 0,
    (1, 2): 1,
    (1, 0): 2,
    (2, 0): 3,
    (2, 1): 3,
}


@dataclass(frozen=True)
class InteractionConfig:
    dataset: DatasetName
    lis_mode: str = DEFAULT_LIS_MODE
    lambda_x: float = 0.1
    lambda_y: float = 0.1
    alpha: float = 1.5
    beta: float = 2.0
    gamma: float = DEFAULT_GAMMA
    top_n: int = DEFAULT_TOP_N
    apply_slot_weight: bool = True
    apply_topn: bool = True
    highd_lit_denom_eps: float = HIGHD_ACTUAL_LIT_DENOM_EPS
    slot_weight_mode: Literal["auto", "conditional", "global"] = "auto"


def lit_to_lis(lit: float, lis_mode: str = DEFAULT_LIS_MODE) -> float:
    cfg = LIS_BINS[lis_mode]
    return float(cfg["vals"][bisect.bisect_right(cfg["cuts"], lit)])


def compute_importance(
    sigma_x: float,
    sigma_y: float,
    lambda_x: float = 0.1,
    lambda_y: float = 0.1,
    alpha: float = 1.5,
    beta: float = 2.0,
) -> float:
    return float(np.exp(-lambda_x * (abs(sigma_x) ** alpha) - lambda_y * (sigma_y ** beta)))


def amplify_importance(i_base: float, weight: float, gamma: float = DEFAULT_GAMMA) -> float:
    return float(min(i_base * (1.0 + gamma * weight), 1.0))


def volume_bin(phys_length: float, phys_width: float, vehicle_type: int, dataset: DatasetName) -> float:
    car_like = int(vehicle_type) == 1
    if car_like:
        height = 1.45 if phys_length < 4.5 else 1.70 if phys_length < 5.0 else 1.90
    else:
        height = 2.75 if phys_length < 12.0 else 3.75
    volume = phys_width * phys_length * height
    for i, edge in enumerate((12.0, 20.0, 90.0, 150.0)):
        if volume < edge:
            return float(i)
    return 4.0


def lane_level_from_lane_id(lane_id: int, sorted_lane_ids: list[int]) -> int:
    if lane_id not in sorted_lane_ids:
        return -1
    if len(sorted_lane_ids) == 1:
        return 1
    idx = sorted_lane_ids.index(int(lane_id))
    if idx == 0:
        return 0
    if idx == len(sorted_lane_ids) - 1:
        return 2
    return 1


def target_context(lane_levels: np.ndarray, t: int, lane_ids: np.ndarray | None = None) -> tuple[str, int]:
    """Return actual-experiment context kind and index from observed history.

    Historical preprocessing computes the LC context once from the complete
    observed history window available at prediction time. It does not inspect
    future trajectory labels.
    """
    hist = np.asarray(lane_levels, dtype=np.int16)
    valid = hist >= 0
    if not bool(valid.any()):
        return "global", -1
    first_change = None
    if lane_ids is not None:
        lanes = np.asarray(lane_ids, dtype=np.int32)
        for idx in range(1, min(hist.size, lanes.size)):
            if lanes[idx] != lanes[idx - 1]:
                first_change = idx
                break
    else:
        for idx in range(1, hist.size):
            if hist[idx] >= 0 and hist[idx - 1] >= 0 and hist[idx] != hist[idx - 1]:
                first_change = idx
                break
    if first_change is None:
        level = int(hist[np.flatnonzero(valid)[-1]])
        if 0 <= level <= 2:
            return "lane_following", level
        return "global", -1
    group = LC_LEVEL_TO_GROUP.get((int(hist[first_change - 1]), int(hist[first_change])), -1)
    if group < 0:
        return "global", -1
    return ("pre_lc" if t < first_change else "post_lc"), group


def slot_weight(slot: int, context_kind: str, context_index: int) -> float:
    if context_kind == "lane_following" and 0 <= context_index <= 2:
        return float(SLOT_WEIGHTS_BY_LANE_LEVEL[context_index, slot])
    if context_kind == "pre_lc" and 0 <= context_index <= 3:
        return float(SLOT_WEIGHTS_PRE_LC[context_index, slot])
    if context_kind == "post_lc" and 0 <= context_index <= 3:
        return float(SLOT_WEIGHTS_POST_LC[context_index, slot])
    return float(GLOBAL_SLOT_WEIGHTS[slot])


def topn_filter_scores(scores: np.ndarray, valid: np.ndarray, top_n: int = DEFAULT_TOP_N) -> np.ndarray:
    filtered = np.zeros_like(scores, dtype=np.float32)
    slots = [int(i) for i in np.flatnonzero(valid)]
    slots.sort(key=lambda k: (-float(scores[k]), TOPN_SLOT_PRIORITY.get(k, scores.size)))
    for k in slots[: int(top_n)]:
        filtered[k] = float(scores[k])
    return filtered


def lateral_motion_state(slot: int, nb_lat_v: float, nb_lane_offset: float, nb_lane_width: float) -> float:
    norm = nb_lane_offset / (nb_lane_width * 0.5) if nb_lane_width > 0.5 else 0.0
    if abs(norm) <= 0.5:
        return 1.0
    if slot < 2:
        return 0.0 if norm * nb_lat_v < 0.0 else 2.0
    if slot < 5:
        return 0.0 if nb_lat_v < 0.0 else 2.0
    return 0.0 if nb_lat_v > 0.0 else 2.0


def _rot2d(x: float, y: float, theta: float) -> tuple[float, float]:
    c = math.cos(theta)
    s = math.sin(theta)
    return c * x + s * y, -s * x + c * y


def _vehicle_front_rear_pts(
    cx: float, cy: float, hdg: float, width: float, length: float
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    ux, uy = math.cos(hdg), math.sin(hdg)
    lx, ly = -math.sin(hdg), math.cos(hdg)
    hl, hw = length / 2.0, width / 2.0
    fm = (cx + hl * ux, cy + hl * uy)
    rm = (cx - hl * ux, cy - hl * uy)
    front3 = [(fm[0] + hw * lx, fm[1] + hw * ly), fm, (fm[0] - hw * lx, fm[1] - hw * ly)]
    rear3 = [(rm[0] + hw * lx, rm[1] + hw * ly), rm, (rm[0] - hw * lx, rm[1] - hw * ly)]
    return front3, rear3


def infer_slot_code(dx: float, dy: float, target_lane: int, neighbor_lane: int) -> int:
    lead = dx >= 0.0
    if target_lane >= 0 and neighbor_lane >= 0:
        lane_delta = int(neighbor_lane) - int(target_lane)
    else:
        lane_delta = -1 if dy > 1.0 else 1 if dy < -1.0 else 0
    if lane_delta == 0:
        return 0 if lead else 1
    left_side = lane_delta < 0 if target_lane >= 0 and neighbor_lane >= 0 else dy > 0.0
    alongside = abs(dx) < 0.5
    if left_side:
        return 3 if alongside else 2 if lead else 4
    return 6 if alongside else 5 if lead else 7


def _exid_keypoint_dxdy(
    slot: int,
    target: np.ndarray,
    neighbor: np.ndarray,
    target_heading: float,
    neighbor_heading: float,
    target_width: float,
    target_length: float,
    neighbor_width: float,
    neighbor_length: float,
) -> tuple[float, float]:
    ego_front, ego_rear = _vehicle_front_rear_pts(
        float(target[0]), float(target[1]), target_heading, target_width, target_length
    )
    nb_front, nb_rear = _vehicle_front_rear_pts(
        float(neighbor[0]), float(neighbor[1]), neighbor_heading, neighbor_width, neighbor_length
    )
    if slot in (3, 6):
        dx, dy = _rot2d(float(neighbor[0] - target[0]), float(neighbor[1] - target[1]), target_heading)
        dy -= (1.0 if dy >= 0.0 else -1.0) * 0.5 * (target_width + neighbor_width)
        return dx, dy
    ego_pts, nb_pts = (ego_front, nb_rear) if slot in (0, 2, 5) else (ego_rear, nb_front)
    best_dist = math.inf
    best_ep = ego_pts[1]
    best_np = nb_pts[1]
    for ep in ego_pts:
        for np_ in nb_pts:
            dist = math.hypot(ep[0] - np_[0], ep[1] - np_[1])
            if dist < best_dist:
                best_dist = dist
                best_ep = ep
                best_np = np_
    return _rot2d(best_np[0] - best_ep[0], best_np[1] - best_ep[1], target_heading)


def _compute_raw_pair(
    target: np.ndarray,
    neigh: np.ndarray,
    *,
    slot: int,
    dataset: DatasetName,
    target_length: float,
    target_width: float,
    neigh_length: float,
    neigh_width: float,
    target_heading: float,
    neigh_heading: float,
    highd_lit_denom_eps: float,
) -> tuple[float, float, float, float, float, float, float]:
    rel = neigh - target
    if dataset == "exiD":
        dx, dy = _exid_keypoint_dxdy(
            slot,
            target,
            neigh,
            target_heading,
            neigh_heading,
            target_width,
            target_length,
            neigh_width,
            neigh_length,
        )
        dvx, dvy = _rot2d(float(rel[2]), float(rel[3]), target_heading)
        dax, day = _rot2d(float(rel[4]), float(rel[5]), target_heading)
        gap = abs(dx)
        denom_base = dvx if dx >= 0.0 else -dvx
        denom = denom_base if abs(denom_base) >= 1e-6 else (1e-6 if denom_base >= 0.0 else -1e-6)
    else:
        dx, dy, dvx, dvy, dax, day = (float(v) for v in rel[:6])
        half_sum = 0.5 * (target_length + neigh_length)
        if dx >= 0.0:
            gap = abs(dx - half_sum)
            denom_base = dvx
        else:
            gap = abs(-dx - half_sum)
            denom_base = -dvx
        eps = float(highd_lit_denom_eps)
        denom = denom_base + (eps if denom_base >= 0.0 else -eps)
        if abs(denom) < 1e-6:
            denom = 1e-6 if denom >= 0.0 else -1e-6
    lit = gap / denom
    return dx, dy, dvx, dvy, dax, day, lit


def build_pair_features(
    x_agents: np.ndarray,
    obs_valid: np.ndarray,
    agent_length: np.ndarray,
    agent_width: np.ndarray,
    agent_type: np.ndarray,
    lane_id: np.ndarray,
    lane_offset: np.ndarray,
    lane_width: np.ndarray,
    *,
    lane_level: np.ndarray | None = None,
    heading: np.ndarray | None = None,
    slot_override: np.ndarray | None = None,
    context_override: np.ndarray | None = None,
    lateral_velocity: np.ndarray | None = None,
    config: InteractionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build directed pair features ``[A,A,T,10]`` or ``[B,A,A,T,10]``."""
    squeeze = False
    if x_agents.ndim == 3:
        x_agents = x_agents[None]
        obs_valid = obs_valid[None]
        agent_length = agent_length[None]
        agent_width = agent_width[None]
        agent_type = agent_type[None]
        lane_id = lane_id[None]
        lane_offset = lane_offset[None]
        lane_width = lane_width[None]
        lane_level = None if lane_level is None else lane_level[None]
        heading = None if heading is None else heading[None]
        slot_override = None if slot_override is None else slot_override[None]
        context_override = None if context_override is None else context_override[None]
        lateral_velocity = None if lateral_velocity is None else lateral_velocity[None]
        squeeze = True

    bsz, agents, hist, _ = x_agents.shape
    pair = np.zeros((bsz, agents, agents, hist, NB_DIM), dtype=np.float32)
    valid = np.zeros((bsz, agents, agents, hist), dtype=bool)
    boosted = np.zeros((bsz, agents, agents, hist), dtype=np.float32)
    heading_arr = np.zeros((bsz, agents, hist), dtype=np.float32) if heading is None else np.asarray(heading, dtype=np.float32)
    lane_level_arr = None if lane_level is None else np.asarray(lane_level, dtype=np.int16)
    slot_arr = None if slot_override is None else np.asarray(slot_override, dtype=np.int16)
    context_arr = None if context_override is None else np.asarray(context_override, dtype=np.int16)
    lat_vel_arr = None if lateral_velocity is None else np.asarray(lateral_velocity, dtype=np.float32)

    for b in range(bsz):
        for i in range(agents):
            for j in range(agents):
                if i == j:
                    continue
                for t in range(hist):
                    if not (obs_valid[b, i, t] and obs_valid[b, j, t]):
                        continue
                    target = x_agents[b, i, t]
                    neigh = x_agents[b, j, t]
                    rel = neigh - target
                    slot = -1 if slot_arr is None else int(slot_arr[b, i, j, t])
                    if slot < 0:
                        slot = infer_slot_code(float(rel[0]), float(rel[1]), int(lane_id[b, i, t]), int(lane_id[b, j, t]))

                    dx, dy, dvx, dvy, dax, day, lit = _compute_raw_pair(
                        target,
                        neigh,
                        slot=slot,
                        dataset=config.dataset,
                        target_length=float(agent_length[b, i]),
                        target_width=float(agent_width[b, i]),
                        neigh_length=float(agent_length[b, j]),
                        neigh_width=float(agent_width[b, j]),
                        target_heading=float(heading_arr[b, i, t]),
                        neigh_heading=float(heading_arr[b, j, t]),
                        highd_lit_denom_eps=float(config.highd_lit_denom_eps),
                    )
                    nb_lat_v = float(lat_vel_arr[b, j, t]) if lat_vel_arr is not None else float(neigh[3])
                    sigma_x = lit_to_lis(lit, config.lis_mode)
                    sigma_y_motion = lateral_motion_state(
                        slot, nb_lat_v, float(lane_offset[b, j, t]), float(lane_width[b, j, t])
                    )
                    ti_lane = int(lane_id[b, i, t])
                    nb_lane = int(lane_id[b, j, t])
                    delta_lane = float(abs(nb_lane - ti_lane)) if ti_lane >= 0 and nb_lane >= 0 else 0.0
                    sigma_y = float(math.sqrt(sigma_y_motion * sigma_y_motion + delta_lane * delta_lane))
                    dim = volume_bin(
                        float(agent_length[b, j]),
                        float(agent_width[b, j]),
                        int(agent_type[b, j]),
                        config.dataset,
                    )
                    i_base = compute_importance(
                        sigma_x, sigma_y, config.lambda_x, config.lambda_y, config.alpha, config.beta
                    )
                    if config.apply_slot_weight:
                        weight_mode = config.slot_weight_mode
                        if weight_mode == "auto":
                            weight_mode = "conditional"
                        if weight_mode == "global":
                            kind, idx = "global", -1
                        elif context_arr is not None:
                            context_code = int(context_arr[b, i, t])
                            if 0 <= context_code <= 2:
                                kind, idx = "lane_following", context_code
                            elif 10 <= context_code <= 13:
                                kind, idx = "pre_lc", context_code - 10
                            elif 20 <= context_code <= 23:
                                kind, idx = "post_lc", context_code - 20
                            else:
                                kind, idx = "global", -1
                        elif lane_level_arr is not None:
                            kind, idx = target_context(lane_level_arr[b, i], t, lane_ids=lane_id[b, i])
                        else:
                            kind, idx = "global", -1
                        weight = slot_weight(slot, kind, idx)
                        imp = amplify_importance(i_base, weight, config.gamma)
                    else:
                        imp = i_base
                    pair[b, i, j, t] = np.asarray(
                        [dx, dy, dvx, dvy, dax, day, sigma_x, sigma_y, dim, imp], dtype=np.float32
                    )
                    boosted[b, i, j, t] = imp
                    valid[b, i, j, t] = True

            if config.apply_topn and config.top_n > 0:
                for t in range(hist):
                    keep_scores = topn_filter_scores(boosted[b, i, :, t], valid[b, i, :, t], config.top_n)
                    drop = valid[b, i, :, t] & (keep_scores == 0.0)
                    pair[b, i, drop, t, 9] = 0.0
                    pair[b, i, keep_scores > 0.0, t, 9] = keep_scores[keep_scores > 0.0]

    if not np.isfinite(pair).all():
        pair[~np.isfinite(pair)] = 0.0
    return (pair[0], valid[0]) if squeeze else (pair, valid)
