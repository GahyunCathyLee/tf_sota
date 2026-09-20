#!/usr/bin/env python3
"""Common per-agent multimodal metrics for persistent-agent prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EvaluationConfig:
    num_modes: int = 6
    future_steps: int = 15
    miss_threshold_m: float = 2.0


def _as_bool(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=bool)


def evaluate_multimodal_predictions(
    pred_xy: np.ndarray,
    y_agents: np.ndarray,
    future_valid: np.ndarray,
    scored_agent_mask: np.ndarray,
    *,
    config: EvaluationConfig = EvaluationConfig(),
    return_diagnostics: bool = True,
) -> dict[str, Any]:
    """Evaluate K=6 multimodal trajectories with per-agent mode selection.

    Parameters
    ----------
    pred_xy:
        Array shaped ``[B, A, K, T, 2]``.
    y_agents:
        Array shaped ``[B, A, T, >=2]``. Only xy columns are used.
    future_valid:
        Boolean array shaped ``[B, A, T]``.
    scored_agent_mask:
        Boolean array shaped ``[B, A]``.

    The headline aggregation is agent-averaged:

    ``mean_i min_k metric(i, k)``.
    """
    pred = np.asarray(pred_xy, dtype=np.float64)
    gt = np.asarray(y_agents, dtype=np.float64)[..., :2]
    fut_valid = _as_bool(future_valid)
    scored = _as_bool(scored_agent_mask)

    if pred.ndim != 5:
        raise ValueError(f"pred_xy must be [B,A,K,T,2], got {pred.shape}")
    if pred.shape[2] != config.num_modes:
        raise ValueError(f"Expected K={config.num_modes}, got {pred.shape[2]}")
    if pred.shape[3] != config.future_steps:
        raise ValueError(f"Expected T={config.future_steps}, got {pred.shape[3]}")
    if pred.shape[-1] != 2:
        raise ValueError(f"Expected xy size 2, got {pred.shape[-1]}")
    if gt.shape != pred.shape[:2] + pred.shape[3:4] + (2,):
        raise ValueError(f"y_agents xy shape mismatch: {gt.shape} vs {pred.shape}")
    if fut_valid.shape != pred.shape[:2] + pred.shape[3:4]:
        raise ValueError(f"future_valid shape mismatch: {fut_valid.shape} vs {pred.shape}")
    if scored.shape != pred.shape[:2]:
        raise ValueError(f"scored_agent_mask shape mismatch: {scored.shape} vs {pred.shape}")

    scored_complete = scored & fut_valid.all(axis=-1)
    if not np.array_equal(scored, scored_complete):
        bad = int(np.count_nonzero(scored & ~fut_valid.all(axis=-1)))
        raise ValueError(f"Headline Rule-B evaluation requires complete futures; bad scored agents={bad}")

    if not np.isfinite(pred).all():
        raise ValueError("pred_xy contains NaN or Inf")
    if not np.isfinite(gt[scored]).all():
        raise ValueError("scored y_agents contain NaN or Inf")

    num_scored = int(scored.sum())
    if num_scored == 0:
        raise ValueError("No scored agents")

    err = np.linalg.norm(pred - gt[:, :, None, :, :], axis=-1)  # [B,A,K,T]
    ade_modes = err.mean(axis=-1)
    fde_modes = err[..., -1]
    min_ade = ade_modes.min(axis=-1)
    min_fde = fde_modes.min(axis=-1)
    miss = min_fde > config.miss_threshold_m

    out: dict[str, Any] = {
        "minADE_6": float(min_ade[scored].mean()),
        "minFDE_6": float(min_fde[scored].mean()),
        "MR_2m@6": float(miss[scored].mean()),
        "num_scenes": int(pred.shape[0]),
        "num_scored_agents": num_scored,
        "mean_scored_agents_per_scene": float(scored.sum(axis=1).mean()),
        "aggregation": "agent_averaged",
        "num_modes": config.num_modes,
        "future_steps": config.future_steps,
        "miss_threshold_m": config.miss_threshold_m,
    }

    if return_diagnostics:
        scene_has_scored = scored.any(axis=1)
        scene_minade = np.zeros(pred.shape[0], dtype=np.float64)
        scene_minfde = np.zeros(pred.shape[0], dtype=np.float64)
        scene_mr = np.zeros(pred.shape[0], dtype=np.float64)
        for b in np.flatnonzero(scene_has_scored):
            m = scored[b]
            scene_minade[b] = float(min_ade[b, m].mean())
            scene_minfde[b] = float(min_fde[b, m].mean())
            scene_mr[b] = float(miss[b, m].mean())
        out["diagnostic_scene_averaged"] = {
            "minADE_6": float(scene_minade[scene_has_scored].mean()),
            "minFDE_6": float(scene_minfde[scene_has_scored].mean()),
            "MR_2m@6": float(scene_mr[scene_has_scored].mean()),
            "num_scenes_with_scored_agents": int(scene_has_scored.sum()),
        }
        ego_mask = np.zeros_like(scored)
        ego_mask[:, 0] = scored[:, 0]
        if bool(ego_mask.any()):
            out["diagnostic_ego_only"] = {
                "minADE_6": float(min_ade[ego_mask].mean()),
                "minFDE_6": float(min_fde[ego_mask].mean()),
                "MR_2m@6": float(miss[ego_mask].mean()),
                "num_scored_agents": int(ego_mask.sum()),
            }
    return out

