#!/usr/bin/env python3
"""Production-format persistent-agent highD/exiD multi-agent writer.

The writer reads canonical NeighFormer sample keys and raw CSV tracks, then
creates a separate model-agnostic multi-agent dataset. Canonical highD/exiD
outputs are read-only inputs and are never modified.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
VRU_CLASSES = {"motorcycle", "bicycle", "pedestrian"}
ROLE_NAMES = (
    "ego",
    "front",
    "rear",
    "left_front",
    "left_alongside",
    "left_rear",
    "right_front",
    "right_alongside",
    "right_rear",
)
ROLE_TO_CODE = {name: i for i, name in enumerate(ROLE_NAMES)}
ROLE_UNASSIGNED = 9
ROLE_PAD = -1


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HIGH_D = _load_module("ma_highd_preprocess", EXPERIMENT_ROOT / "data" / "highD" / "preprocess.py")
EXI_D = _load_module("ma_exid_preprocess", EXPERIMENT_ROOT / "data" / "exiD" / "preprocess.py")


@dataclass(frozen=True)
class DatasetPaths:
    dataset: str
    source_root: Path

    @property
    def canonical_dir(self) -> Path:
        return self.source_root / self.dataset / "dimI"

    @property
    def raw_dir(self) -> Path:
        return self.source_root / self.dataset / "raw"

    @property
    def split_dir(self) -> Path:
        return self.source_root / self.dataset / "splits"


def _rec_file_id(rec_id: int) -> str:
    return f"{int(rec_id):02d}"


def _safe_numeric_series(s: pd.Series, default: int = -1) -> np.ndarray:
    return (
        pd.to_numeric(s.astype(str).str.strip().str.split(";").str[0], errors="coerce")
        .fillna(default)
        .astype(np.int32)
        .to_numpy()
    )


def _type_code(cls: str, dataset: str) -> int:
    c = str(cls or "").strip().lower()
    if dataset == "highD":
        return 1 if c == "car" else 2
    return 1 if c in {"car", "van"} else 2


def _load_canonical_arrays(canonical_dir: Path) -> dict[str, np.ndarray]:
    names = (
        "x_ego",
        "y",
        "y_vel",
        "y_acc",
        "x_last_abs",
        "meta_recordingId",
        "meta_trackId",
        "meta_frame",
    )
    return {name: np.load(canonical_dir / f"{name}.npy", mmap_mode="r") for name in names}


def _sample_indices(
    paths: DatasetPaths,
    split: str,
    max_samples: int | None,
    max_recordings: int | None,
) -> np.ndarray:
    split_path = paths.split_dir / f"{split}_indices.npy"
    if split_path.exists():
        indices = np.load(split_path).astype(np.int64)
    else:
        total = int(np.load(paths.canonical_dir / "x_ego.npy", mmap_mode="r").shape[0])
        indices = np.arange(total, dtype=np.int64)
    if max_recordings is not None and max_recordings > 0:
        meta_rec = np.load(paths.canonical_dir / "meta_recordingId.npy", mmap_mode="r")
        recs = np.sort(np.unique(meta_rec[indices]))[: int(max_recordings)]
        indices = indices[np.isin(meta_rec[indices], recs)]
    if max_samples is not None and int(max_samples) > 0:
        indices = indices[: int(max_samples)]
    return np.sort(indices.astype(np.int64))


def _build_highd_context(raw_dir: Path, rec_id: int, target_hz: float) -> dict[str, Any]:
    rid = _rec_file_id(rec_id)
    rec_meta = pd.read_csv(raw_dir / f"{rid}_recordingMeta.csv")
    trk_meta = pd.read_csv(raw_dir / f"{rid}_tracksMeta.csv")
    tracks = pd.read_csv(raw_dir / f"{rid}_tracks.csv")

    c_y, frame_rate, upper_mark, lower_mark = HIGH_D.flip_constants(rec_meta)
    step = max(1, int(round(frame_rate / target_hz)))
    for c in HIGH_D.NEIGHBOR_COLS_8:
        if c not in tracks.columns:
            tracks[c] = 0
    for c in ("xVelocity", "yVelocity", "xAcceleration", "yAcceleration"):
        if c not in tracks.columns:
            tracks[c] = 0.0
    if "laneId" not in tracks.columns:
        tracks["laneId"] = 0

    vid_to_dd = dict(zip(trk_meta["id"].astype(int), trk_meta["drivingDirection"].astype(int)))
    vid_to_length = dict(zip(trk_meta["id"].astype(int), trk_meta["width"].astype(float)))
    vid_to_width = dict(zip(trk_meta["id"].astype(int), trk_meta["height"].astype(float)))
    vid_to_type = {
        int(tid): _type_code(cls, "highD")
        for tid, cls in zip(trk_meta["id"].astype(int).tolist(), trk_meta["class"].astype(str).tolist())
    }

    upper_for_calc = upper_mark.copy()
    if len(upper_for_calc):
        upper_for_calc = np.sort((c_y - upper_for_calc).astype(np.float32))
    upper_center, _ = HIGH_D.build_lane_tables(upper_for_calc)
    lower_center, _ = HIGH_D.build_lane_tables(lower_mark)
    upper_mm = (1, int(len(upper_center))) if len(upper_center) else None

    frame = tracks["frame"].astype(np.int32).to_numpy()
    vid = tracks["id"].astype(np.int32).to_numpy()
    x = tracks["x"].astype(np.float32).to_numpy().copy()
    y = tracks["y"].astype(np.float32).to_numpy().copy()
    row_length = np.asarray([vid_to_length.get(int(v), 0.0) for v in vid], dtype=np.float32)
    row_width = np.asarray([vid_to_width.get(int(v), 0.0) for v in vid], dtype=np.float32)
    x += 0.5 * row_length
    y += 0.5 * row_width
    xv = tracks["xVelocity"].astype(np.float32).to_numpy()
    yv = tracks["yVelocity"].astype(np.float32).to_numpy()
    xa = tracks["xAcceleration"].astype(np.float32).to_numpy()
    ya = tracks["yAcceleration"].astype(np.float32).to_numpy()
    lane_id = tracks["laneId"].astype(np.int32).to_numpy()
    dd = np.asarray([vid_to_dd.get(int(v), 0) for v in vid], dtype=np.int8)
    x_max = float(np.nanmax(x)) if x.size else 0.0

    n_upper = len(upper_mark)
    lane_offset = np.zeros(len(y), dtype=np.float32)
    lane_width = np.full(len(y), 3.75, dtype=np.float32)
    lid_arr = lane_id.astype(np.int32)
    mask_lo = dd == 2
    j_lo = lid_arr - n_upper - 2
    ok_lo = mask_lo & (j_lo >= 0) & (j_lo < len(lower_mark) - 1)
    lane_offset[ok_lo] = y[ok_lo] - 0.5 * (lower_mark[j_lo[ok_lo]] + lower_mark[j_lo[ok_lo] + 1])
    mask_up = dd == 1
    j_up = lid_arr - 2
    ok_up = mask_up & (j_up >= 0) & (j_up < len(upper_mark) - 1)
    lane_offset[ok_up] = y[ok_up] - 0.5 * (upper_mark[j_up[ok_up]] + upper_mark[j_up[ok_up] + 1])
    lane_offset[dd == 1] *= -1.0
    lane_width[ok_lo] = np.abs(lower_mark[j_lo[ok_lo] + 1] - lower_mark[j_lo[ok_lo]])
    lane_width[ok_up] = np.abs(upper_mark[j_up[ok_up] + 1] - upper_mark[j_up[ok_up]])

    x, y, xv, yv, xa, ya, lane_id = HIGH_D.maybe_flip(x, y, xv, yv, xa, ya, lane_id, dd, c_y, x_max, upper_mm)
    x_min = float(np.nanmin(x)) if x.size else 0.0
    y_min = float(np.nanmin(y)) if y.size else 0.0
    x = (x - x_min).astype(np.float32)
    y = (y - y_min).astype(np.float32)
    lane_ids_per_dd = {
        dd_val: sorted(set(int(lid) for lid in lane_id[dd == dd_val] if int(lid) > 0))
        for dd_val in (1, 2)
    }
    lane_level = np.full(len(lane_id), -1, dtype=np.int16)
    for row, (lid, dd_val) in enumerate(zip(lane_id, dd)):
        lane_level[row] = int(HIGH_D._lane_id_to_level(int(lid), int(dd_val), lane_ids_per_dd.get(int(dd_val), []), True))

    frame_to_row: dict[int, dict[int, int]] = {}
    for v, idxs in tracks.groupby("id").indices.items():
        rows = np.asarray(idxs, dtype=np.int32)
        rows = rows[np.argsort(frame[rows])]
        frame_to_row[int(v)] = {int(fr): int(r) for fr, r in zip(frame[rows], rows)}
    frame_to_tids: dict[int, list[int]] = {}
    for tid, fmap in frame_to_row.items():
        for fr in fmap:
            frame_to_tids.setdefault(int(fr), []).append(int(tid))
    nb_ids_all = np.stack([tracks[c].astype(np.int32).to_numpy() for c in HIGH_D.NEIGHBOR_COLS_8], axis=1)
    return {
        "step": step,
        "frame_to_row": frame_to_row,
        "frame_to_tids": frame_to_tids,
        "nb_ids_all": nb_ids_all,
        "x": x,
        "y": y,
        "xv": xv,
        "yv": yv,
        "xa": xa,
        "ya": ya,
        "lane_id": lane_id.astype(np.int32),
        "lane_level": lane_level,
        "lane_offset": lane_offset.astype(np.float32),
        "lane_width": lane_width.astype(np.float32),
        "heading": np.zeros(len(x), dtype=np.float32),
        "length": vid_to_length,
        "width": vid_to_width,
        "type": vid_to_type,
        "class_map": {},
    }


def _build_exid_context(raw_dir: Path, rec_id: int, target_hz: float) -> dict[str, Any]:
    rid = _rec_file_id(rec_id)
    rec_meta = pd.read_csv(raw_dir / f"{rid}_recordingMeta.csv")
    trk_meta = pd.read_csv(raw_dir / f"{rid}_tracksMeta.csv")
    tracks = pd.read_csv(raw_dir / f"{rid}_tracks.csv", low_memory=False)
    frame_rate = EXI_D.get_frame_rate(rec_meta)
    step = max(1, int(round(frame_rate / target_hz)))
    for c in EXI_D.NEIGHBOR_COLS_8:
        if c not in tracks.columns:
            tracks[c] = -1
    for c in ("lonVelocity", "latVelocity", "lonAcceleration", "latAcceleration"):
        if c not in tracks.columns:
            tracks[c] = 0.0
    if "laneletId" not in tracks.columns:
        tracks["laneletId"] = -1

    class_map = EXI_D.get_class_map(trk_meta)
    tracks = tracks.sort_values(["trackId", "frame"], kind="mergesort").reset_index(drop=True)
    frame = tracks["frame"].astype(np.int32).to_numpy()
    vid = tracks["trackId"].astype(np.int32).to_numpy()
    x = tracks["xCenter"].astype(np.float32).to_numpy().copy()
    y = tracks["yCenter"].astype(np.float32).to_numpy().copy()
    xv = tracks["lonVelocity"].astype(np.float32).to_numpy()
    yv = tracks["latVelocity"].astype(np.float32).to_numpy()
    xa = tracks["lonAcceleration"].astype(np.float32).to_numpy()
    ya = tracks["latAcceleration"].astype(np.float32).to_numpy()
    lane_id = tracks["laneletId"].fillna(-1).astype(np.int32).to_numpy()
    if "latLaneCenterOffset" in tracks.columns:
        lane_offset = pd.to_numeric(
            tracks["latLaneCenterOffset"].astype(str).str.strip().str.split(";").str[0],
            errors="coerce",
        ).fillna(0.0).astype(np.float32).to_numpy()
    else:
        lane_offset = np.zeros(len(tracks), dtype=np.float32)
    if "laneWidth" in tracks.columns:
        lane_width = pd.to_numeric(
            tracks["laneWidth"].astype(str).str.strip().str.split(";").str[0],
            errors="coerce",
        ).fillna(3.5).astype(np.float32).to_numpy()
    else:
        lane_width = np.full(len(tracks), 3.5, dtype=np.float32)
    heading = np.deg2rad(
        tracks["heading"].astype(np.float32).to_numpy()
        if "heading" in tracks.columns
        else np.zeros(len(tracks), dtype=np.float32)
    ).astype(np.float32)
    width_arr = tracks["width"].astype(np.float32).to_numpy() if "width" in tracks.columns else np.zeros(len(tracks), dtype=np.float32)
    length_arr = tracks["length"].astype(np.float32).to_numpy() if "length" in tracks.columns else np.zeros(len(tracks), dtype=np.float32)

    x_min = float(np.nanmin(x)) if x.size else 0.0
    y_min = float(np.nanmin(y)) if y.size else 0.0
    x = (x - x_min).astype(np.float32)
    y = (y - y_min).astype(np.float32)

    frame_to_row: dict[int, dict[int, int]] = {}
    vid_to_length: dict[int, float] = {}
    vid_to_width: dict[int, float] = {}
    vid_to_type: dict[int, int] = {}
    for v, idxs in tracks.groupby("trackId").indices.items():
        rows = np.asarray(idxs, dtype=np.int32)
        rows = rows[np.argsort(frame[rows])]
        frame_to_row[int(v)] = {int(fr): int(r) for fr, r in zip(frame[rows], rows)}
        r0 = int(rows[0])
        vid_to_length[int(v)] = float(length_arr[r0])
        vid_to_width[int(v)] = float(width_arr[r0])
        vid_to_type[int(v)] = _type_code(class_map.get(int(v), "other"), "exiD")
    frame_to_tids: dict[int, list[int]] = {}
    for tid, fmap in frame_to_row.items():
        for fr in fmap:
            frame_to_tids.setdefault(int(fr), []).append(int(tid))
    nb_ids_all = np.stack([_safe_numeric_series(tracks[c]) for c in EXI_D.NEIGHBOR_COLS_8], axis=1)
    lane_ids_rec = sorted(set(int(lid) for lid in lane_id if int(lid) >= 0))
    lane_level = np.asarray([EXI_D._lane_id_to_level(int(lid), lane_ids_rec) for lid in lane_id], dtype=np.int16)
    return {
        "step": step,
        "frame_to_row": frame_to_row,
        "frame_to_tids": frame_to_tids,
        "nb_ids_all": nb_ids_all,
        "x": x,
        "y": y,
        "xv": xv,
        "yv": yv,
        "xa": xa,
        "ya": ya,
        "lane_id": lane_id,
        "lane_level": lane_level,
        "lane_offset": lane_offset.astype(np.float32),
        "lane_width": lane_width.astype(np.float32),
        "heading": heading,
        "length": vid_to_length,
        "width": vid_to_width,
        "type": vid_to_type,
        "class_map": class_map,
    }


def _state(
    dataset: str,
    ctx: dict[str, Any],
    tid: int,
    frame: int,
    ref_x: float,
    ref_y: float,
    ref_hdg: float,
) -> tuple[np.ndarray, int, int, float, float, float] | None:
    row = ctx["frame_to_row"].get(int(tid), {}).get(int(frame))
    if row is None:
        return None
    if dataset == "exiD":
        hdg = float(ctx["heading"][row])
        px, py = EXI_D._norm_pos(float(ctx["x"][row]), float(ctx["y"][row]), ref_x, ref_y, ref_hdg)
        vx, vy = EXI_D._local_to_norm_frame(float(ctx["xv"][row]), float(ctx["yv"][row]), hdg, ref_hdg)
        ax, ay = EXI_D._local_to_norm_frame(float(ctx["xa"][row]), float(ctx["ya"][row]), hdg, ref_hdg)
        heading = np.float32(hdg - ref_hdg)
        state = np.asarray([px, py, vx, vy, ax, ay], dtype=np.float32)
    else:
        state = np.asarray(
            [
                ctx["x"][row] - ref_x,
                ctx["y"][row] - ref_y,
                ctx["xv"][row],
                ctx["yv"][row],
                ctx["xa"][row],
                ctx["ya"][row],
            ],
            dtype=np.float32,
        )
        heading = np.float32(0.0)
    return (
        state,
        int(ctx["lane_id"][row]),
        int(ctx["lane_level"][row]),
        float(ctx["lane_offset"][row]),
        float(ctx["lane_width"][row]),
        float(heading),
    )


def _role_map_at_t0(ctx: dict[str, Any], ego_tid: int, t0: int) -> dict[int, int]:
    row = ctx["frame_to_row"].get(int(ego_tid), {}).get(int(t0))
    if row is None:
        return {}
    ids = np.asarray(ctx["nb_ids_all"][row], dtype=np.int64)
    return {int(tid): slot + 1 for slot, tid in enumerate(ids) if int(tid) > 0}


def _summ(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def _array_specs(n: int, amax: int, hist: int, fut: int) -> dict[str, tuple[str, tuple[int, ...], Any]]:
    return {
        "agent_ids": ("int32", (n, amax), -1),
        "x_agents": ("float32", (n, amax, hist, 6), 0),
        "obs_valid": ("bool", (n, amax, hist), False),
        "y_agents": ("float32", (n, amax, fut, 6), 0),
        "future_valid": ("bool", (n, amax, fut), False),
        "scored_agent_mask": ("bool", (n, amax), False),
        "distances_t0": ("float32", (n, amax), np.nan),
        "agent_length": ("float32", (n, amax), 0),
        "agent_width": ("float32", (n, amax), 0),
        "agent_type": ("int8", (n, amax), 0),
        "heading": ("float32", (n, amax, hist), 0),
        "lateral_velocity": ("float32", (n, amax, hist), 0),
        "lane_id": ("int32", (n, amax, hist), -1),
        "lane_level": ("int16", (n, amax, hist), -1),
        "lane_offset": ("float32", (n, amax, hist), 0),
        "lane_width": ("float32", (n, amax, hist), 0),
        "semantic_role_t0": ("int8", (n, amax), ROLE_PAD),
        "candidate_count": ("int16", (n,), 0),
        "retained_count": ("int16", (n,), 0),
        "truncated_count": ("int16", (n,), 0),
        "recordingId": ("int32", (n,), 0),
        "ego_trackId": ("int32", (n,), 0),
        "t0_frame": ("int32", (n,), 0),
        "source_index": ("int64", (n,), 0),
        "ego_index": ("int16", (n,), 0),
    }


def _allocate(out_dir: Path, n: int, amax: int, hist: int, fut: int) -> dict[str, np.ndarray]:
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = _array_specs(n, amax, hist, fut)
    arrays = {}
    for name, (dtype, shape, fill) in specs.items():
        arr = open_memmap(out_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        arr[...] = fill
        arrays[name] = arr
    return arrays


def _allocate_chunk(n: int, amax: int, hist: int, fut: int) -> dict[str, np.ndarray]:
    arrays = {}
    for name, (dtype, shape, fill) in _array_specs(n, amax, hist, fut).items():
        arr = np.empty(shape, dtype=np.dtype(dtype))
        arr[...] = fill
        arrays[name] = arr
    return arrays


def _fill_scenes(
    paths: DatasetPaths,
    out: dict[str, np.ndarray],
    sample_indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    *,
    target_hz: float,
    context_radius: float,
    amax: int,
    include_vru: bool,
) -> dict[str, Any]:
    hist = int(arrays["x_ego"].shape[1])
    fut = int(arrays["y"].shape[1])
    out["source_index"][...] = sample_indices.astype(np.int64)

    context_cache: dict[int, dict[str, Any]] = {}
    candidate_counts: list[int] = []
    retained_counts: list[int] = []
    scored_counts: list[int] = []
    trunc_counts: list[int] = []
    ego_errors: list[float] = []
    duplicate_scenes = 0
    missing_ego = 0
    nan_inf = 0

    for out_i, source_idx in enumerate(sample_indices):
        idx = int(source_idx)
        rec_id = int(arrays["meta_recordingId"][idx])
        ego_tid = int(arrays["meta_trackId"][idx])
        t0 = int(arrays["meta_frame"][idx])
        if rec_id not in context_cache:
            context_cache[rec_id] = (
                _build_highd_context(paths.raw_dir, rec_id, target_hz)
                if paths.dataset == "highD"
                else _build_exid_context(paths.raw_dir, rec_id, target_hz)
            )
        ctx = context_cache[rec_id]
        step = int(ctx["step"])
        hist_frames = [t0 - (hist - 1 - t) * step for t in range(hist)]
        fut_frames = [t0 + (t + 1) * step for t in range(fut)]
        ref_x = float(arrays["x_last_abs"][idx, 0])
        ref_y = float(arrays["x_last_abs"][idx, 1])
        ref_hdg = 0.0
        if paths.dataset == "exiD":
            ego_row = ctx["frame_to_row"].get(ego_tid, {}).get(t0)
            ref_hdg = float(ctx["heading"][ego_row]) if ego_row is not None else 0.0

        visible = sorted(ctx["frame_to_tids"].get(t0, []))
        if paths.dataset == "exiD" and not include_vru:
            visible = [tid for tid in visible if ctx["class_map"].get(int(tid), "other") not in VRU_CLASSES]

        candidates: list[tuple[int, float]] = []
        for tid in visible:
            row_state = _state(paths.dataset, ctx, tid, t0, ref_x, ref_y, ref_hdg)
            if row_state is None:
                continue
            state = row_state[0]
            dist = float(np.linalg.norm(state[:2]))
            if tid == ego_tid or dist <= context_radius:
                candidates.append((int(tid), dist))
        candidates.sort(key=lambda item: (item[0] != ego_tid, item[1], item[0]))
        selected = candidates[:amax]

        out["recordingId"][out_i] = rec_id
        out["ego_trackId"][out_i] = ego_tid
        out["t0_frame"][out_i] = t0
        out["ego_index"][out_i] = 0
        out["candidate_count"][out_i] = len(candidates)
        out["retained_count"][out_i] = len(selected)
        out["truncated_count"][out_i] = max(0, len(candidates) - len(selected))
        candidate_counts.append(len(candidates))
        retained_counts.append(len(selected))
        trunc_counts.append(max(0, len(candidates) - len(selected)))
        roles = _role_map_at_t0(ctx, ego_tid, t0)

        seen_ids: set[int] = set()
        for ai, (tid, dist) in enumerate(selected):
            if tid in seen_ids:
                duplicate_scenes += 1
            seen_ids.add(tid)
            out["agent_ids"][out_i, ai] = tid
            out["distances_t0"][out_i, ai] = dist
            out["semantic_role_t0"][out_i, ai] = ROLE_TO_CODE["ego"] if tid == ego_tid else roles.get(tid, ROLE_UNASSIGNED)
            out["agent_length"][out_i, ai] = float(ctx["length"].get(tid, 0.0))
            out["agent_width"][out_i, ai] = float(ctx["width"].get(tid, 0.0))
            out["agent_type"][out_i, ai] = int(ctx["type"].get(tid, 1))

            for ti, frame in enumerate(hist_frames):
                row_state = _state(paths.dataset, ctx, tid, frame, ref_x, ref_y, ref_hdg)
                if row_state is None:
                    continue
                state, lid, lvl, lco, lw, hdg = row_state
                out["x_agents"][out_i, ai, ti] = state
                out["obs_valid"][out_i, ai, ti] = True
                out["lane_id"][out_i, ai, ti] = lid
                out["lane_level"][out_i, ai, ti] = lvl
                out["lane_offset"][out_i, ai, ti] = lco
                out["lane_width"][out_i, ai, ti] = lw
                out["heading"][out_i, ai, ti] = hdg
                raw_row = ctx["frame_to_row"].get(int(tid), {}).get(int(frame))
                if raw_row is not None:
                    out["lateral_velocity"][out_i, ai, ti] = float(ctx["yv"][raw_row])
            for fi, frame in enumerate(fut_frames):
                row_state = _state(paths.dataset, ctx, tid, frame, ref_x, ref_y, ref_hdg)
                if row_state is None:
                    continue
                out["y_agents"][out_i, ai, fi] = row_state[0]
                out["future_valid"][out_i, ai, fi] = True

        if not selected or selected[0][0] != ego_tid:
            missing_ego += 1
        future_counts = out["future_valid"][out_i].sum(axis=1)
        valid_agent = out["agent_ids"][out_i] >= 0
        out["scored_agent_mask"][out_i] = valid_agent & (future_counts == fut)
        scored_counts.append(int(out["scored_agent_mask"][out_i].sum()))

        if selected and selected[0][0] == ego_tid:
            err = float(np.max(np.abs(out["y_agents"][out_i, 0, :, :2] - np.asarray(arrays["y"][idx], dtype=np.float32))))
            ego_errors.append(err)
        scene_arrays = (
            out["x_agents"][out_i],
            out["y_agents"][out_i],
            out["distances_t0"][out_i, valid_agent],
            out["heading"][out_i],
            out["lane_offset"][out_i],
            out["lane_width"][out_i],
        )
        nan_inf += sum(int(np.count_nonzero(~np.isfinite(a))) for a in scene_arrays)

    return {
        "candidate_counts": candidate_counts,
        "retained_counts": retained_counts,
        "scored_counts": scored_counts,
        "trunc_counts": trunc_counts,
        "ego_errors": ego_errors,
        "duplicate_scenes": duplicate_scenes,
        "missing_ego": missing_ego,
        "nan_inf": nan_inf,
    }


def _build_chunk(
    paths: DatasetPaths,
    sample_indices: np.ndarray,
    *,
    target_hz: float,
    context_radius: float,
    amax: int,
    include_vru: bool,
    hist: int,
    fut: int,
    lo: int,
) -> dict[str, Any]:
    arrays = _load_canonical_arrays(paths.canonical_dir)
    out = _allocate_chunk(int(sample_indices.size), amax, hist, fut)
    stats = _fill_scenes(
        paths,
        out,
        sample_indices,
        arrays,
        target_hz=target_hz,
        context_radius=context_radius,
        amax=amax,
        include_vru=include_vru,
    )
    return {"lo": int(lo), "arrays": out, "stats": stats}


def _merge_stats(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key in ("candidate_counts", "retained_counts", "scored_counts", "trunc_counts", "ego_errors"):
        dst[key].extend(src[key])
    for key in ("duplicate_scenes", "missing_ego", "nan_inf"):
        dst[key] += int(src[key])


def build_dataset(
    paths: DatasetPaths,
    out_dir: Path,
    sample_indices: np.ndarray,
    *,
    target_hz: float,
    context_radius: float,
    amax: int,
    include_vru: bool,
    num_workers: int = 1,
    chunk_size: int = 1024,
) -> dict[str, Any]:
    start = time.perf_counter()
    arrays = _load_canonical_arrays(paths.canonical_dir)
    hist = int(arrays["x_ego"].shape[1])
    fut = int(arrays["y"].shape[1])
    n = int(sample_indices.size)
    out = _allocate(out_dir, n, amax, hist, fut)
    stats = {
        "candidate_counts": [],
        "retained_counts": [],
        "scored_counts": [],
        "trunc_counts": [],
        "ego_errors": [],
        "duplicate_scenes": 0,
        "missing_ego": 0,
        "nan_inf": 0,
    }
    n_workers = int(num_workers)
    if n_workers == 0:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, n_workers)
    chunk_size = max(1, int(chunk_size))

    if n_workers == 1 or n == 0:
        _merge_stats(
            stats,
            _fill_scenes(
                paths,
                out,
                sample_indices,
                arrays,
                target_hz=target_hz,
                context_radius=context_radius,
                amax=amax,
                include_vru=include_vru,
            ),
        )
    else:
        tasks = []
        for lo in range(0, n, chunk_size):
            hi = min(n, lo + chunk_size)
            tasks.append((lo, hi, np.asarray(sample_indices[lo:hi], dtype=np.int64)))
        results: dict[int, dict[str, Any]] = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as exe:
            futs = [
                exe.submit(
                    _build_chunk,
                    paths,
                    chunk_indices,
                    target_hz=target_hz,
                    context_radius=context_radius,
                    amax=amax,
                    include_vru=include_vru,
                    hist=hist,
                    fut=fut,
                    lo=lo,
                )
                for lo, _hi, chunk_indices in tasks
            ]
            for fut_obj in concurrent.futures.as_completed(futs):
                result = fut_obj.result()
                results[int(result["lo"])] = result
        for lo, hi, _chunk_indices in tasks:
            result = results[lo]
            chunk_arrays = result["arrays"]
            for name, arr in out.items():
                arr[lo:hi] = chunk_arrays[name]
            _merge_stats(stats, result["stats"])

    for arr in out.values():
        if hasattr(arr, "flush"):
            arr.flush()

    elapsed = time.perf_counter() - start
    file_size = sum(p.stat().st_size for p in out_dir.glob("*.npy"))
    report = {
        "dataset": paths.dataset,
        "source_root": str(paths.source_root),
        "canonical_dir": str(paths.canonical_dir),
        "raw_dir": str(paths.raw_dir),
        "output_dir": str(out_dir),
        "num_scenes": n,
        "target_hz": target_hz,
        "history_steps": hist,
        "future_steps": fut,
        "context_radius_m": context_radius,
        "amax": amax,
        "include_vru": include_vru,
        "scored_agent_rule": "Rule B: visible at t0 and complete 15-step future",
        "arrays": {name: list(arr.shape) for name, arr in out.items()},
        "candidate_count": _summ(stats["candidate_counts"]),
        "retained_count": _summ(stats["retained_counts"]),
        "truncated_count": _summ(stats["trunc_counts"]),
        "num_truncated_scenes": int(sum(v > 0 for v in stats["trunc_counts"])),
        "pct_truncated_scenes": float(np.mean(np.asarray(stats["trunc_counts"]) > 0)) if n else 0.0,
        "scored_agents": _summ(stats["scored_counts"]),
        "mean_scored_agents_per_scene": float(np.mean(stats["scored_counts"])) if stats["scored_counts"] else 0.0,
        "pct_scenes_scored_ge2": float(np.mean(np.asarray(stats["scored_counts"]) >= 2)) if stats["scored_counts"] else 0.0,
        "ego_y_max_abs_error": {
            "max": float(max(stats["ego_errors"])) if stats["ego_errors"] else None,
            "mean": float(np.mean(stats["ego_errors"])) if stats["ego_errors"] else None,
        },
        "missing_ego_scenes": int(stats["missing_ego"]),
        "duplicate_agent_id_scenes": int(stats["duplicate_scenes"]),
        "nan_inf_count": int(stats["nan_inf"]),
        "file_size_bytes": int(file_size),
        "preprocess_seconds": float(elapsed),
        "throughput_scenes_per_second": float(n / elapsed) if elapsed > 0 else 0.0,
        "num_workers": int(n_workers),
        "chunk_size": int(chunk_size),
        "schema_version": "persistent_multiagent_v1",
    }
    (out_dir / "preprocess_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=EXPERIMENT_ROOT.parent / "neighformer" / "data")
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT / "data")
    parser.add_argument("--dataset", choices=["highD", "exiD", "both"], default="both")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--subset-name", default="train_subset")
    parser.add_argument("--max-samples", type=int, default=5000, help="Maximum canonical windows to convert; <=0 means full split.")
    parser.add_argument("--max-recordings", type=int)
    parser.add_argument("--target-hz", type=float, default=3.0)
    parser.add_argument("--context-radius", type=float, default=120.0)
    parser.add_argument("--amax", type=int, default=48)
    parser.add_argument("--include-vru", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1, help="Worker processes (0 = os.cpu_count())")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Scenes per worker task when num-workers > 1")
    args = parser.parse_args(argv)

    datasets = ["highD", "exiD"] if args.dataset == "both" else [args.dataset]
    combined = {}
    for ds in datasets:
        paths = DatasetPaths(ds, args.source_root.expanduser().resolve())
        indices = _sample_indices(paths, args.split, args.max_samples, args.max_recordings)
        out_dir = args.output_root.expanduser().resolve() / f"{ds}_multiagent" / args.subset_name
        report = build_dataset(
            paths,
            out_dir,
            indices,
            target_hz=args.target_hz,
            context_radius=args.context_radius,
            amax=args.amax,
            include_vru=args.include_vru,
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,
        )
        combined[ds] = report
        print(json.dumps({
            "dataset": ds,
            "output_dir": str(out_dir),
            "num_scenes": report["num_scenes"],
            "ego_y_max_abs_error": report["ego_y_max_abs_error"],
            "candidate_count": report["candidate_count"],
            "pct_truncated_scenes": report["pct_truncated_scenes"],
            "mean_scored_agents_per_scene": report["mean_scored_agents_per_scene"],
            "nan_inf_count": report["nan_inf_count"],
            "throughput_scenes_per_second": report["throughput_scenes_per_second"],
        }, indent=2))
    combined_path = args.output_root.expanduser().resolve() / "multiagent_preprocess_report.json"
    combined_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
