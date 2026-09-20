#!/usr/bin/env python3
"""Prototype persistent-agent highD/exiD multi-agent datasets.

This script is intentionally small-scope: it samples existing canonical
NeighFormer scene keys, reconstructs physical-agent histories/futures from raw
CSV, writes bounded prototype arrays, and reports statistics needed before full
multi-agent generation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
ROLE_NAMES = (
    "front",
    "rear",
    "left_front",
    "left_alongside",
    "left_rear",
    "right_front",
    "right_alongside",
    "right_rear",
)
VRU_CLASSES = {"motorcycle", "bicycle", "pedestrian"}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PAR_PRE = _load_module("ma_par_preprocess", EXPERIMENT_ROOT / "data" / "par" / "preprocess.py")


@dataclass(frozen=True)
class DatasetPaths:
    dataset: str
    data_root: Path

    @property
    def canonical_dir(self) -> Path:
        return self.data_root / self.dataset / "dimI"

    @property
    def raw_dir(self) -> Path:
        return self.data_root / self.dataset / "raw"

    @property
    def split_dir(self) -> Path:
        return self.data_root / self.dataset / "splits"


def _load_arrays(data_dir: Path) -> dict[str, np.ndarray]:
    names = ("x_ego", "x_nb", "nb_mask", "y", "y_vel", "y_acc", "x_last_abs",
             "meta_recordingId", "meta_trackId", "meta_frame")
    return {name: np.load(data_dir / f"{name}.npy", mmap_mode="r") for name in names}


def _sample_indices(paths: DatasetPaths, split: str, n: int, seed: int, max_recordings: int | None) -> np.ndarray:
    split_path = paths.split_dir / f"{split}_indices.npy"
    if split_path.exists():
        base = np.load(split_path).astype(np.int64)
    else:
        total = int(np.load(paths.canonical_dir / "x_ego.npy", mmap_mode="r").shape[0])
        base = np.arange(total, dtype=np.int64)
    if max_recordings is not None and max_recordings > 0:
        meta_rec = np.load(paths.canonical_dir / "meta_recordingId.npy", mmap_mode="r")
        recs = np.unique(meta_rec[base])
        recs = np.sort(recs)[: int(max_recordings)]
        base = base[np.isin(meta_rec[base], recs)]
    if base.size <= n:
        return np.sort(base)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(base, size=n, replace=False))


def _rec_file_id(rec_id: int) -> str:
    return f"{int(rec_id):02d}"


def _read_class_map(paths: DatasetPaths, rec_id: int) -> dict[int, str]:
    meta = paths.raw_dir / f"{_rec_file_id(rec_id)}_tracksMeta.csv"
    if not meta.exists():
        return {}
    df = pd.read_csv(meta)
    if paths.dataset == "highD":
        id_col = "id" if "id" in df.columns else "trackId"
        class_col = "class" if "class" in df.columns else None
    else:
        id_col = "trackId"
        class_col = "class" if "class" in df.columns else None
    if id_col not in df.columns or class_col is None:
        return {}
    return {
        int(tid): str(cls).strip().lower()
        for tid, cls in zip(df[id_col].astype(int).tolist(), df[class_col].astype(str).tolist())
    }


def _build_context(paths: DatasetPaths, rec_id: int, target_hz: float) -> dict[str, Any]:
    if paths.dataset == "highD":
        ctx = PAR_PRE._build_highd_context(paths.raw_dir, rec_id, target_hz)
    else:
        ctx = PAR_PRE._build_exid_context(paths.raw_dir, rec_id, target_hz)
    frame_to_tids: dict[int, list[int]] = {}
    for tid, fmap in ctx["frame_to_row"].items():
        for frame in fmap:
            frame_to_tids.setdefault(int(frame), []).append(int(tid))
    ctx["frame_to_tids"] = frame_to_tids
    ctx["class_map"] = _read_class_map(paths, rec_id)
    return ctx


def _state(paths: DatasetPaths, ctx: dict[str, Any], tid: int, frame: int,
           ref_x: float, ref_y: float, ref_hdg: float) -> np.ndarray | None:
    if paths.dataset == "highD":
        return PAR_PRE._highd_state(ctx, tid, frame, ref_x, ref_y)
    return PAR_PRE._exid_state(ctx, tid, frame, ref_x, ref_y, ref_hdg)


def _role_map_at_t0(ctx: dict[str, Any], ego_tid: int, t0: int) -> dict[int, str]:
    ego_row = ctx["frame_to_row"].get(int(ego_tid), {}).get(int(t0))
    if ego_row is None:
        return {}
    ids = np.asarray(ctx["nb_ids_all"][ego_row], dtype=np.int64)
    return {int(tid): ROLE_NAMES[k] for k, tid in enumerate(ids) if int(tid) > 0}


def _old_slot_switch_stats(arrays: dict[str, np.ndarray], ctx: dict[str, Any],
                           idx: int, hist_frames: list[int]) -> tuple[int, int, bool]:
    ego_tid = int(arrays["meta_trackId"][idx])
    active_slots = 0
    switched_slots = 0
    for slot in range(8):
        ids = []
        for ti, frame in enumerate(hist_frames):
            if not bool(arrays["nb_mask"][idx, ti, slot]):
                continue
            ego_row = ctx["frame_to_row"].get(ego_tid, {}).get(int(frame))
            if ego_row is None:
                continue
            tid = int(ctx["nb_ids_all"][ego_row, slot])
            if tid > 0:
                ids.append(tid)
        if ids:
            active_slots += 1
            if len(set(ids)) > 1:
                switched_slots += 1
    return active_slots, switched_slots, switched_slots > 0


def _summ(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _ratio(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def build_dataset(
    paths: DatasetPaths,
    out_dir: Path,
    sample_indices: np.ndarray,
    target_hz: float,
    context_radius: float,
    amax: int,
    min_future_steps: int,
    include_vru: bool,
    examples: int,
) -> dict[str, Any]:
    arrays = _load_arrays(paths.canonical_dir)
    hist = int(arrays["x_ego"].shape[1])
    fut = int(arrays["y"].shape[1])
    n = int(sample_indices.size)

    agent_ids = np.full((n, amax), -1, dtype=np.int32)
    x_agents = np.zeros((n, amax, hist, 6), dtype=np.float32)
    obs_valid = np.zeros((n, amax, hist), dtype=bool)
    y_agents = np.zeros((n, amax, fut, 6), dtype=np.float32)
    future_valid = np.zeros((n, amax, fut), dtype=bool)
    scored_a = np.zeros((n, amax), dtype=bool)
    scored_b = np.zeros((n, amax), dtype=bool)
    scored_c = np.zeros((n, amax), dtype=bool)
    distances = np.full((n, amax), np.nan, dtype=np.float32)
    role_t0 = np.full((n, amax), "", dtype=object)
    meta = {
        "recordingId": np.zeros(n, dtype=np.int32),
        "ego_trackId": np.zeros(n, dtype=np.int32),
        "t0_frame": np.zeros(n, dtype=np.int32),
        "source_index": sample_indices.astype(np.int64),
    }

    context_cache: dict[int, dict[str, Any]] = {}
    stats: dict[str, Any] = {
        "visible_all_t0": [],
        "visible_relevant_t0": [],
        "selected_agents": [],
        "scored_rule_a": [],
        "scored_rule_b": [],
        "scored_rule_c": [],
        "complete_future_agents": [],
        "obs_valid_ratio": [],
        "future_valid_ratio": [],
        "candidate_discard_rule_a": [],
        "candidate_discard_rule_b": [],
        "candidate_discard_rule_c": [],
        "truncated": 0,
        "ego_y_max_abs_error": [],
        "old_active_slots": 0,
        "old_switched_slots": 0,
        "old_scenes_with_switch": 0,
        "examples": [],
    }

    for out_i, idx in enumerate(sample_indices):
        idx = int(idx)
        rec_id = int(arrays["meta_recordingId"][idx])
        ego_tid = int(arrays["meta_trackId"][idx])
        t0 = int(arrays["meta_frame"][idx])
        if rec_id not in context_cache:
            context_cache[rec_id] = _build_context(paths, rec_id, target_hz)
        ctx = context_cache[rec_id]
        step = int(ctx["step"])
        hist_frames = [t0 - (hist - 1 - t) * step for t in range(hist)]
        fut_frames = [t0 + (t + 1) * step for t in range(fut)]
        ref_x = float(arrays["x_last_abs"][idx, 0])
        ref_y = float(arrays["x_last_abs"][idx, 1])
        ref_hdg = 0.0
        if paths.dataset == "exiD":
            ref_hdg = float(ctx["frame_to_hdg"].get(ego_tid, {}).get(t0, 0.0))

        roles = _role_map_at_t0(ctx, ego_tid, t0)
        visible = sorted(ctx["frame_to_tids"].get(t0, []))
        if not include_vru and paths.dataset == "exiD":
            visible = [tid for tid in visible if ctx["class_map"].get(tid, "car") not in VRU_CLASSES]
        candidates = []
        for tid in visible:
            s0 = _state(paths, ctx, tid, t0, ref_x, ref_y, ref_hdg)
            if s0 is None:
                continue
            dist = float(np.linalg.norm(s0[0:2]))
            if tid == ego_tid or dist <= context_radius:
                candidates.append((tid, dist))
        candidates.sort(key=lambda x: (x[1] != 0.0, x[1], x[0]))
        if len(candidates) > amax:
            stats["truncated"] += 1
        selected = candidates[:amax]

        meta["recordingId"][out_i] = rec_id
        meta["ego_trackId"][out_i] = ego_tid
        meta["t0_frame"][out_i] = t0
        active_slots, switched_slots, scene_switch = _old_slot_switch_stats(arrays, ctx, idx, hist_frames)
        stats["old_active_slots"] += active_slots
        stats["old_switched_slots"] += switched_slots
        stats["old_scenes_with_switch"] += int(scene_switch)

        for ai, (tid, dist) in enumerate(selected):
            agent_ids[out_i, ai] = int(tid)
            distances[out_i, ai] = float(dist)
            role_t0[out_i, ai] = "ego" if tid == ego_tid else roles.get(tid, "unassigned")
            for ti, frame in enumerate(hist_frames):
                s = _state(paths, ctx, tid, frame, ref_x, ref_y, ref_hdg)
                if s is not None:
                    x_agents[out_i, ai, ti] = s
                    obs_valid[out_i, ai, ti] = True
            for fi, frame in enumerate(fut_frames):
                s = _state(paths, ctx, tid, frame, ref_x, ref_y, ref_hdg)
                if s is not None:
                    y_agents[out_i, ai, fi] = s
                    future_valid[out_i, ai, fi] = True

        future_counts = future_valid[out_i].sum(axis=1)
        valid_agent = agent_ids[out_i] >= 0
        scored_a[out_i] = valid_agent & (future_counts >= 1)
        scored_b[out_i] = valid_agent & (future_counts == fut)
        scored_c[out_i] = valid_agent & (future_counts >= min_future_steps)

        ego_pos = np.flatnonzero(agent_ids[out_i] == ego_tid)
        if ego_pos.size:
            ego_ai = int(ego_pos[0])
            err = np.max(np.abs(y_agents[out_i, ego_ai, :, 0:2] - np.asarray(arrays["y"][idx], dtype=np.float32)))
            stats["ego_y_max_abs_error"].append(float(err))

        stats["visible_all_t0"].append(len(visible))
        stats["visible_relevant_t0"].append(len(candidates))
        stats["selected_agents"].append(len(selected))
        stats["scored_rule_a"].append(int(scored_a[out_i].sum()))
        stats["scored_rule_b"].append(int(scored_b[out_i].sum()))
        stats["scored_rule_c"].append(int(scored_c[out_i].sum()))
        stats["complete_future_agents"].append(int(((future_counts == fut) & valid_agent).sum()))
        stats["obs_valid_ratio"].append(float(obs_valid[out_i, valid_agent].mean()) if valid_agent.any() else 0.0)
        stats["future_valid_ratio"].append(float(future_valid[out_i, valid_agent].mean()) if valid_agent.any() else 0.0)
        for rule_name, mask in (("a", scored_a[out_i]), ("b", scored_b[out_i]), ("c", scored_c[out_i])):
            stats[f"candidate_discard_rule_{rule_name}"].append(1.0 - _ratio(int(mask.sum()), len(selected)))

        if len(stats["examples"]) < examples:
            rows = []
            for ai, (tid, dist) in enumerate(selected[: min(12, len(selected))]):
                rows.append({
                    "agent_index": ai,
                    "trackId": int(tid),
                    "distance_t0": round(float(dist), 3),
                    "obs_count": int(obs_valid[out_i, ai].sum()),
                    "future_count": int(future_valid[out_i, ai].sum()),
                    "scored_A": bool(scored_a[out_i, ai]),
                    "scored_B": bool(scored_b[out_i, ai]),
                    "scored_C": bool(scored_c[out_i, ai]),
                    "role_t0": str(role_t0[out_i, ai]),
                })
            stats["examples"].append({
                "recordingId": rec_id,
                "ego_trackId": ego_tid,
                "t0_frame": t0,
                "agents": rows,
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{paths.dataset}_prototype.npz",
        agent_ids=agent_ids,
        x_agents=x_agents,
        obs_valid=obs_valid,
        y_agents=y_agents,
        future_valid=future_valid,
        scored_agent_mask_rule_a=scored_a,
        scored_agent_mask_rule_b=scored_b,
        scored_agent_mask_rule_c=scored_c,
        distances_t0=distances,
        role_t0=role_t0,
        **meta,
    )

    n_scenes = max(1, n)
    report = {
        "dataset": paths.dataset,
        "canonical_dir": str(paths.canonical_dir),
        "raw_dir": str(paths.raw_dir),
        "num_scenes": n,
        "target_hz": target_hz,
        "history_steps": hist,
        "future_steps": fut,
        "context_radius_m": context_radius,
        "amax": amax,
        "min_future_steps_rule_c": min_future_steps,
        "arrays": {
            "agent_ids": list(agent_ids.shape),
            "x_agents": list(x_agents.shape),
            "obs_valid": list(obs_valid.shape),
            "y_agents": list(y_agents.shape),
            "future_valid": list(future_valid.shape),
            "scored_agent_mask": list(scored_b.shape),
        },
        "visible_all_t0": _summ(stats["visible_all_t0"]),
        "visible_relevant_t0": _summ(stats["visible_relevant_t0"]),
        "selected_agents": _summ(stats["selected_agents"]),
        "scored_rule_a_visible_t0_any_future": _summ(stats["scored_rule_a"]),
        "scored_rule_b_visible_t0_complete_future": _summ(stats["scored_rule_b"]),
        "scored_rule_c_visible_t0_min_future": _summ(stats["scored_rule_c"]),
        "complete_future_agents": _summ(stats["complete_future_agents"]),
        "pct_scenes_rule_a_ge2": _ratio(sum(v >= 2 for v in stats["scored_rule_a"]), n_scenes),
        "pct_scenes_rule_a_ge3": _ratio(sum(v >= 3 for v in stats["scored_rule_a"]), n_scenes),
        "pct_scenes_rule_b_ge2": _ratio(sum(v >= 2 for v in stats["scored_rule_b"]), n_scenes),
        "pct_scenes_rule_b_ge3": _ratio(sum(v >= 3 for v in stats["scored_rule_b"]), n_scenes),
        "pct_scenes_rule_c_ge2": _ratio(sum(v >= 2 for v in stats["scored_rule_c"]), n_scenes),
        "pct_scenes_rule_c_ge3": _ratio(sum(v >= 3 for v in stats["scored_rule_c"]), n_scenes),
        "mean_candidate_discard_rule_a": float(np.mean(stats["candidate_discard_rule_a"])) if n else 0.0,
        "mean_candidate_discard_rule_b": float(np.mean(stats["candidate_discard_rule_b"])) if n else 0.0,
        "mean_candidate_discard_rule_c": float(np.mean(stats["candidate_discard_rule_c"])) if n else 0.0,
        "obs_valid_ratio": _summ(stats["obs_valid_ratio"]),
        "future_valid_ratio": _summ(stats["future_valid_ratio"]),
        "old_semantic_slot_identity_switch_rate": _ratio(stats["old_switched_slots"], stats["old_active_slots"]),
        "old_semantic_slot_scenes_with_any_switch": _ratio(stats["old_scenes_with_switch"], n_scenes),
        "old_active_slot_histories": int(stats["old_active_slots"]),
        "old_switched_slot_histories": int(stats["old_switched_slots"]),
        "amax_truncation_rate": _ratio(stats["truncated"], n_scenes),
        "ego_y_max_abs_error": {
            "max": float(max(stats["ego_y_max_abs_error"])) if stats["ego_y_max_abs_error"] else None,
            "mean": float(np.mean(stats["ego_y_max_abs_error"])) if stats["ego_y_max_abs_error"] else None,
        },
        "examples": stats["examples"],
    }
    (out_dir / f"{paths.dataset}_prototype_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=EXPERIMENT_ROOT.parent / "neighformer" / "data")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT / "data" / "multiagent" / "prototype")
    parser.add_argument("--dataset", choices=["highD", "exiD", "both"], default="both")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--target-hz", type=float, default=3.0)
    parser.add_argument("--context-radius", type=float, default=120.0)
    parser.add_argument("--amax", type=int, default=16)
    parser.add_argument("--min-future-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--max-recordings", type=int, default=6,
                        help="Limit prototype samples to the first N recordings to avoid full-dataset raw CSV loads")
    parser.add_argument("--include-vru", action="store_true")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args(argv)

    datasets = ["highD", "exiD"] if args.dataset == "both" else [args.dataset]
    reports = {}
    for ds_name in datasets:
        paths = DatasetPaths(ds_name, args.data_root.expanduser().resolve())
        indices = _sample_indices(
            paths,
            args.split,
            args.samples,
            args.seed + (17 if ds_name == "exiD" else 0),
            args.max_recordings,
        )
        report = build_dataset(
            paths,
            args.output_dir,
            indices,
            args.target_hz,
            args.context_radius,
            args.amax,
            args.min_future_steps,
            args.include_vru,
            args.examples,
        )
        reports[ds_name] = report
        print(json.dumps({
            "dataset": ds_name,
            "num_scenes": report["num_scenes"],
            "selected_agents": report["selected_agents"],
            "scored_rule_b": report["scored_rule_b_visible_t0_complete_future"],
            "pct_rule_b_ge2": report["pct_scenes_rule_b_ge2"],
            "old_slot_switch_rate": report["old_semantic_slot_identity_switch_rate"],
            "ego_y_max_abs_error": report["ego_y_max_abs_error"],
        }, indent=2))
    (args.output_dir / "combined_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
