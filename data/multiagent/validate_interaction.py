#!/usr/bin/env python3
"""Validate paper-spec interaction features against canonical ego-slot x_nb."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from data.multiagent import preprocess as ma_pre  # noqa: E402
from data.multiagent.interaction import (  # noqa: E402
    DEFAULT_TOP_N,
    HIGHD_ACTUAL_LIT_DENOM_EPS,
    InteractionConfig,
    amplify_importance,
    build_pair_features,
    compute_importance,
    lateral_motion_state,
    lit_to_lis,
    slot_weight,
    target_context,
    topn_filter_scores,
    volume_bin,
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HIGH_D = _load_module("ma_val_highd_preprocess", EXPERIMENT_ROOT / "data" / "highD" / "preprocess.py")
EXI_D = _load_module("ma_val_exid_preprocess", EXPERIMENT_ROOT / "data" / "exiD" / "preprocess.py")


@dataclass(frozen=True)
class ValidationResult:
    report: dict[str, Any]
    examples: Any


def _sample_indices(paths: ma_pre.DatasetPaths, split: str, max_samples: int, max_recordings: int | None) -> np.ndarray:
    return ma_pre._sample_indices(paths, split, max_samples, max_recordings)


def _context(paths: ma_pre.DatasetPaths, rec_id: int, target_hz: float) -> dict[str, Any]:
    return (
        ma_pre._build_highd_context(paths.raw_dir, rec_id, target_hz)
        if paths.dataset == "highD"
        else ma_pre._build_exid_context(paths.raw_dir, rec_id, target_hz)
    )


def _slot_ids(ctx: dict[str, Any], ego_tid: int, frame: int) -> np.ndarray | None:
    row = ctx["frame_to_row"].get(int(ego_tid), {}).get(int(frame))
    if row is None:
        return None
    return np.asarray(ctx["nb_ids_all"][row], dtype=np.int32)


def _state_row(
    dataset: str,
    ctx: dict[str, Any],
    tid: int,
    frame: int,
    ref_x: float,
    ref_y: float,
    ref_hdg: float,
) -> tuple[np.ndarray, int, int, float, float, float] | None:
    return ma_pre._state(dataset, ctx, tid, frame, ref_x, ref_y, ref_hdg)


def _ego_history_arrays(
    dataset: str,
    ctx: dict[str, Any],
    ego_tid: int,
    hist_frames: list[int],
    ref_x: float,
    ref_y: float,
    ref_hdg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = []
    lane_ids = []
    lane_levels = []
    lane_offsets = []
    lane_widths = []
    headings = []
    for frame in hist_frames:
        row = _state_row(dataset, ctx, ego_tid, frame, ref_x, ref_y, ref_hdg)
        if row is None:
            states.append(np.zeros(6, dtype=np.float32))
            lane_ids.append(-1)
            lane_levels.append(-1)
            lane_offsets.append(0.0)
            lane_widths.append(0.0)
            headings.append(0.0)
        else:
            state, lid, lvl, lco, lw, hdg = row
            states.append(state)
            lane_ids.append(lid)
            lane_levels.append(lvl)
            lane_offsets.append(lco)
            lane_widths.append(lw)
            headings.append(hdg)
    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(lane_ids, dtype=np.int32),
        np.asarray(lane_levels, dtype=np.int16),
        np.asarray(lane_offsets, dtype=np.float32),
        np.asarray(lane_widths, dtype=np.float32),
        np.asarray(headings, dtype=np.float32),
    )


def _agent_type(ctx: dict[str, Any], tid: int) -> int:
    return int(ctx["type"].get(int(tid), 1))


def _tau_x(
    dataset: str,
    dx: float,
    dvx: float,
    target_length: float,
    neigh_length: float,
) -> float:
    if dataset == "exiD":
        denom = dvx if dx >= 0.0 else -dvx
        if abs(denom) < 1e-6:
            denom = 1e-6 if denom >= 0.0 else -1e-6
        return float(abs(dx) / denom)
    half_sum = 0.5 * (target_length + neigh_length)
    if dx >= 0.0:
        gap = abs(dx - half_sum)
        denom_base = dvx
    else:
        gap = abs(-dx - half_sum)
        denom_base = -dvx
    denom = denom_base + (HIGHD_ACTUAL_LIT_DENOM_EPS if denom_base >= 0.0 else -HIGHD_ACTUAL_LIT_DENOM_EPS)
    if abs(denom) < 1e-6:
        denom = 1e-6 if denom >= 0.0 else -1e-6
    return float(gap / denom)


def validate_dataset(
    dataset: str,
    source_root: Path,
    split: str,
    max_samples: int,
    max_recordings: int | None,
    target_hz: float,
    examples: int,
) -> ValidationResult:
    paths = ma_pre.DatasetPaths(dataset, source_root)
    arrays = ma_pre._load_canonical_arrays(paths.canonical_dir)
    x_nb = np.load(paths.canonical_dir / "x_nb.npy", mmap_mode="r")
    nb_mask = np.load(paths.canonical_dir / "nb_mask.npy", mmap_mode="r")
    sample_indices = _sample_indices(paths, split, max_samples, max_recordings)
    hist = int(arrays["x_ego"].shape[1])

    ctx_cache: dict[int, dict[str, Any]] = {}
    total = 0
    sx_match = sy_match = dim_match = i_match = 0
    i_abs = []
    sx_abs = []
    sy_abs = []
    dim_abs = []
    example_rows: list[dict[str, Any]] = []
    sx_mismatches: list[dict[str, Any]] = []
    sy_mismatches: list[dict[str, Any]] = []
    i_mismatches: list[dict[str, Any]] = []
    first_divergence_counts = {"sigma_x": 0, "sigma_y": 0, "dim": 0, "I_filtered": 0}

    cfg = InteractionConfig(dataset=dataset)

    for source_idx in sample_indices:
        idx = int(source_idx)
        rec_id = int(arrays["meta_recordingId"][idx])
        ego_tid = int(arrays["meta_trackId"][idx])
        t0 = int(arrays["meta_frame"][idx])
        if rec_id not in ctx_cache:
            ctx_cache[rec_id] = _context(paths, rec_id, target_hz)
        ctx = ctx_cache[rec_id]
        step = int(ctx["step"])
        ref_x = float(arrays["x_last_abs"][idx, 0])
        ref_y = float(arrays["x_last_abs"][idx, 1])
        ref_hdg = 0.0
        if dataset == "exiD":
            ego_row = ctx["frame_to_row"].get(ego_tid, {}).get(t0)
            ref_hdg = float(ctx["heading"][ego_row]) if ego_row is not None else 0.0
        hist_frames = [t0 - (hist - 1 - t) * step for t in range(hist)]
        ego_states, ego_lane, ego_level, ego_lco, ego_lw, ego_hdg = _ego_history_arrays(
            dataset, ctx, ego_tid, hist_frames, ref_x, ref_y, ref_hdg
        )
        ego_length = float(ctx["length"].get(ego_tid, 0.0))
        ego_width = float(ctx["width"].get(ego_tid, 0.0))
        ego_type = _agent_type(ctx, ego_tid)

        for ti, frame in enumerate(hist_frames):
            ids8 = _slot_ids(ctx, ego_tid, frame)
            if ids8 is None:
                continue
            valid_slots = [slot for slot in range(8) if bool(nb_mask[idx, ti, slot]) and int(ids8[slot]) > 0]
            if not valid_slots:
                continue
            # Build only ego + the eight canonical slot occupants for this timestep.
            agent_states = [ego_states[ti]]
            obs_valid = [True]
            lengths = [ego_length]
            widths = [ego_width]
            types = [ego_type]
            lane_ids = [int(ego_lane[ti])]
            lane_levels = [int(ego_level[ti])]
            lane_offsets = [float(ego_lco[ti])]
            lane_widths = [float(ego_lw[ti])]
            headings = [float(ego_hdg[ti])]
            lat_velocities = [float(ctx["yv"][ctx["frame_to_row"][ego_tid][frame]])]
            slot_to_agent = {}
            for slot in range(8):
                tid = int(ids8[slot])
                state_row_idx = ctx["frame_to_row"].get(int(tid), {}).get(int(frame)) if tid > 0 else None
                row = _state_row(dataset, ctx, tid, frame, ref_x, ref_y, ref_hdg) if state_row_idx is not None else None
                if row is None or state_row_idx is None:
                    continue
                state, lid, lvl, lco, lw, hdg = row
                slot_to_agent[slot] = len(agent_states)
                agent_states.append(state)
                obs_valid.append(True)
                lengths.append(float(ctx["length"].get(tid, 0.0)))
                widths.append(float(ctx["width"].get(tid, 0.0)))
                types.append(_agent_type(ctx, tid))
                lane_ids.append(lid)
                lane_levels.append(lvl)
                lane_offsets.append(lco)
                lane_widths.append(lw)
                headings.append(hdg)
                lat_velocities.append(float(ctx["yv"][state_row_idx]))
            a = len(agent_states)
            x_agents = np.zeros((a, hist, 6), dtype=np.float32)
            ov = np.zeros((a, hist), dtype=bool)
            lane_arr = np.full((a, hist), -1, dtype=np.int32)
            level_arr = np.full((a, hist), -1, dtype=np.int16)
            lco_arr = np.zeros((a, hist), dtype=np.float32)
            lw_arr = np.zeros((a, hist), dtype=np.float32)
            hdg_arr = np.zeros((a, hist), dtype=np.float32)
            latv_arr = np.zeros((a, hist), dtype=np.float32)
            x_agents[:, ti] = np.asarray(agent_states, dtype=np.float32)
            ov[:, ti] = np.asarray(obs_valid, dtype=bool)
            lane_arr[:, ti] = np.asarray(lane_ids, dtype=np.int32)
            lane_arr[0] = ego_lane
            level_arr[0] = ego_level
            level_arr[1:, ti] = np.asarray(lane_levels[1:], dtype=np.int16)
            lco_arr[:, ti] = np.asarray(lane_offsets, dtype=np.float32)
            lw_arr[:, ti] = np.asarray(lane_widths, dtype=np.float32)
            hdg_arr[:, ti] = np.asarray(headings, dtype=np.float32)
            latv_arr[:, ti] = np.asarray(lat_velocities, dtype=np.float32)
            override = np.full((a, a, hist), -1, dtype=np.int16)
            for slot, ai in slot_to_agent.items():
                override[0, ai, ti] = slot
            pair, pair_valid = build_pair_features(
                x_agents,
                ov,
                np.asarray(lengths, dtype=np.float32),
                np.asarray(widths, dtype=np.float32),
                np.asarray(types, dtype=np.int8),
                lane_arr,
                lco_arr,
                lw_arr,
                lane_level=level_arr,
                heading=hdg_arr,
                slot_override=override,
                lateral_velocity=latv_arr,
                config=cfg,
            )

            # Stage examples: reconstruct pre-filter rank over occupied slots.
            boosted = np.zeros(8, dtype=np.float32)
            stages = {}
            kind, cidx = target_context(ego_level, ti, lane_ids=ego_lane)
            weight_mode = cfg.slot_weight_mode
            if weight_mode == "auto":
                weight_mode = "conditional"
            for slot in valid_slots:
                ai = slot_to_agent.get(slot)
                if ai is None or not pair_valid[0, ai, ti]:
                    continue
                p = pair[0, ai, ti]
                sigma_x, sigma_y = float(p[6]), float(p[7])
                i_base = compute_importance(sigma_x, sigma_y)
                w = slot_weight(slot, "global", -1) if weight_mode == "global" else slot_weight(slot, kind, cidx)
                boosted[slot] = amplify_importance(i_base, w)
                stages[slot] = (i_base, w, boosted[slot])
            filtered = topn_filter_scores(boosted, np.asarray([s in valid_slots for s in range(8)], dtype=bool), DEFAULT_TOP_N)
            ranked = sorted(valid_slots, key=lambda s: (-float(boosted[s]), {0: 0, 2: 1, 5: 2, 1: 3, 4: 4, 7: 5, 3: 6, 6: 7}.get(s, 8)))

            for slot in valid_slots:
                ai = slot_to_agent.get(slot)
                if ai is None or not pair_valid[0, ai, ti]:
                    continue
                calc = pair[0, ai, ti].copy()
                canon = np.asarray(x_nb[idx, ti, slot], dtype=np.float32)
                total += 1
                sx_abs.append(abs(float(calc[6] - canon[6])))
                sy_abs.append(abs(float(calc[7] - canon[7])))
                dim_abs.append(abs(float(calc[8] - canon[8])))
                i_abs.append(abs(float(calc[9] - canon[9])))
                sx_ok = abs(float(calc[6] - canon[6])) <= 1e-6
                sy_ok = abs(float(calc[7] - canon[7])) <= 1e-5
                dim_ok = abs(float(calc[8] - canon[8])) <= 1e-6
                i_ok = abs(float(calc[9] - canon[9])) <= 1e-5
                sx_match += int(sx_ok)
                sy_match += int(sy_ok)
                dim_match += int(dim_ok)
                i_match += int(i_ok)
                if not sx_ok:
                    first_divergence_counts["sigma_x"] += 1
                elif not sy_ok:
                    first_divergence_counts["sigma_y"] += 1
                elif not dim_ok:
                    first_divergence_counts["dim"] += 1
                elif not i_ok:
                    first_divergence_counts["I_filtered"] += 1
                tau_x = _tau_x(
                    dataset,
                    float(calc[0]),
                    float(calc[2]),
                    float(ego_length),
                    float(lengths[ai]),
                )
                mismatch_row = {
                    "source_index": idx,
                    "recordingId": rec_id,
                    "ego_trackId": ego_tid,
                    "t0_frame": t0,
                    "history_timestep": ti,
                    "frame": int(frame),
                    "slot": int(slot),
                    "neighbor_trackId": int(ids8[slot]),
                    "canonical_sigma_x": float(canon[6]),
                    "reconstructed_sigma_x": float(calc[6]),
                    "canonical_sigma_y": float(canon[7]),
                    "reconstructed_sigma_y": float(calc[7]),
                    "canonical_dim": float(canon[8]),
                    "reconstructed_dim": float(calc[8]),
                    "canonical_I_filtered": float(canon[9]),
                    "reconstructed_I_filtered": float(calc[9]),
                    "tau_x": tau_x,
                    "dx": float(calc[0]),
                    "dy": float(calc[1]),
                    "dvx": float(calc[2]),
                    "dvy": float(calc[3]),
                    "target_lane_id": int(lane_ids[0]),
                    "neighbor_lane_id": int(lane_ids[ai]),
                    "neighbor_lane_offset": float(lane_offsets[ai]),
                    "neighbor_lane_width": float(lane_widths[ai]),
                    "context": f"{kind}:{cidx}",
                    "rank": int(ranked.index(slot) + 1) if slot in ranked else None,
                }
                if (not sx_ok) and len(sx_mismatches) < examples:
                    sx_mismatches.append(dict(mismatch_row))
                if (not sy_ok) and len(sy_mismatches) < examples:
                    sy_mismatches.append(dict(mismatch_row))
                if (not i_ok) and len(i_mismatches) < examples:
                    i_mismatches.append(dict(mismatch_row))
                if len(example_rows) < examples:
                    i_base, w, i_boosted = stages.get(slot, (None, None, None))
                    example_rows.append(
                        {
                            "source_index": idx,
                            "recordingId": rec_id,
                            "ego_trackId": ego_tid,
                            "t0_frame": t0,
                            "history_timestep": ti,
                            "frame": int(frame),
                            "slot": int(slot),
                            "context": f"{kind}:{cidx}",
                            "rank": int(ranked.index(slot) + 1) if slot in ranked else None,
                            "sigma_x": float(calc[6]),
                            "sigma_y": float(calc[7]),
                            "dim": float(calc[8]),
                            "I_base": None if i_base is None else float(i_base),
                            "w_slot_c": None if w is None else float(w),
                            "I_boosted": None if i_boosted is None else float(i_boosted),
                            "I_filtered": float(calc[9]),
                            "canonical_I": float(canon[9]),
                            "canonical": canon[[6, 7, 8, 9]].tolist(),
                        }
                    )

    den = max(1, total)
    report = {
        "dataset": dataset,
        "num_compared_slots": int(total),
        "sigma_x_match_rate": float(sx_match / den),
        "sigma_y_match_rate": float(sy_match / den),
        "dim_match_rate": float(dim_match / den),
        "I_filtered_match_rate": float(i_match / den),
        "sigma_x_mae": float(np.mean(sx_abs)) if sx_abs else None,
        "sigma_x_max_abs_error": float(np.max(sx_abs)) if sx_abs else None,
        "sigma_y_mae": float(np.mean(sy_abs)) if sy_abs else None,
        "sigma_y_max_abs_error": float(np.max(sy_abs)) if sy_abs else None,
        "dim_mae": float(np.mean(dim_abs)) if dim_abs else None,
        "dim_max_abs_error": float(np.max(dim_abs)) if dim_abs else None,
        "I_filtered_mae": float(np.mean(i_abs)) if i_abs else None,
        "I_filtered_max_abs_error": float(np.max(i_abs)) if i_abs else None,
        "first_divergence_counts": first_divergence_counts,
        "max_samples": int(max_samples),
        "max_recordings": max_recordings,
    }
    examples_out = {
        "first_valid_slots": example_rows,
        "sigma_x_mismatches": sx_mismatches,
        "sigma_y_mismatches": sy_mismatches,
        "I_filtered_mismatches": i_mismatches,
    }
    return ValidationResult(report, examples_out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=EXPERIMENT_ROOT.parent / "neighformer" / "data")
    parser.add_argument("--dataset", choices=["highD", "exiD", "both"], default="both")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--max-recordings", type=int, default=6)
    parser.add_argument("--target-hz", type=float, default=3.0)
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--output", type=Path, default=EXPERIMENT_ROOT / "data" / "multiagent" / "interaction_validation_v3.json")
    args = parser.parse_args(argv)

    datasets = ["highD", "exiD"] if args.dataset == "both" else [args.dataset]
    out = {}
    for dataset in datasets:
        result = validate_dataset(
            dataset,
            args.source_root.expanduser().resolve(),
            args.split,
            args.max_samples,
            args.max_recordings,
            args.target_hz,
            args.examples,
        )
        out[dataset] = {"report": result.report, "examples": result.examples}
        print(json.dumps({dataset: result.report}, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
