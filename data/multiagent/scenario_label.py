#!/usr/bin/env python3
"""Raw trajectory event labels for persistent multi-agent scenes.

This script labels each retained/scored agent window directly from raw highD or
exiD tracks. It intentionally does not reuse existing single-agent
``scenario_labels.csv`` files and does not emit traffic-state labels.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HIGH_D_LABEL = _load_module("ma_label_highd", EXPERIMENT_ROOT / "data" / "highD" / "scenario_label.py")
EXI_D_LABEL = _load_module("ma_label_exid", EXPERIMENT_ROOT / "data" / "exiD" / "scenario_label.py")


def _dataset_module(dataset: str):
    return HIGH_D_LABEL if dataset == "highD" else EXI_D_LABEL


def _rec_file_id(rec_id: int) -> str:
    return f"{int(rec_id):02d}"


def _load_scene_arrays(root: Path) -> dict[str, np.ndarray]:
    names = ("agent_ids", "scored_agent_mask", "recordingId", "ego_trackId", "t0_frame", "source_index")
    return {name: np.load(root / f"{name}.npy", mmap_mode="r") for name in names}


def _keys_by_recording(arrays: dict[str, np.ndarray], *, scored_only: bool) -> dict[int, set[tuple[int, int]]]:
    out: dict[int, set[tuple[int, int]]] = {}
    agent_ids = arrays["agent_ids"]
    mask = np.asarray(arrays["scored_agent_mask"], dtype=bool) if scored_only else np.asarray(agent_ids >= 0)
    for scene_i in range(agent_ids.shape[0]):
        rec_id = int(arrays["recordingId"][scene_i])
        t0 = int(arrays["t0_frame"][scene_i])
        for agent_i in np.flatnonzero(mask[scene_i]):
            tid = int(agent_ids[scene_i, agent_i])
            if tid >= 0:
                out.setdefault(rec_id, set()).add((tid, t0))
    return out


def _process_recording(
    dataset: str,
    raw_dir: Path,
    rec_id: int,
    keys: list[tuple[int, int]],
    history_sec: float,
    future_sec: float,
    target_hz: float,
    w_adj: int,
) -> tuple[int, dict[tuple[int, int], dict[str, Any]]]:
    module = _dataset_module(dataset)
    xx = _rec_file_id(rec_id)
    tracks_path = raw_dir / f"{xx}_tracks.csv"
    recmeta_path = raw_dir / f"{xx}_recordingMeta.csv"
    if not tracks_path.exists() or not recmeta_path.exists():
        return int(rec_id), {}

    tracks = module.smart_read_csv(tracks_path)
    recmeta = module.smart_read_csv(recmeta_path)
    tracks.columns = [c.strip() for c in tracks.columns]
    recmeta.columns = [c.strip() for c in recmeta.columns]
    tracks_n = module.normalize_tracks(tracks, xx)
    recmeta_n = module.normalize_recmeta(recmeta, xx)

    fr = float(recmeta_n["frameRate"].iloc[0])
    if not np.isfinite(fr) or fr <= 0:
        return int(rec_id), {}

    tracks_n = tracks_n.merge(recmeta_n, on="recordingId", how="left")
    tracks_n["frame"] = pd.to_numeric(tracks_n["frame"], errors="coerce").astype("Int64")
    tracks_n = tracks_n.dropna(subset=["frame"]).sort_values(["trackId", "frame"])
    tracks_n["_frame_int"] = tracks_n["frame"].astype(int)

    ds_step = max(1, int(round(fr / target_hz)))
    hist_steps = int(round(history_sec * target_hz))
    fut_steps = int(round(future_sec * target_hz))
    win_native = (hist_steps + fut_steps - 1) * ds_step
    lookup = module.build_lane_lookup(tracks_n)

    grouped: dict[int, dict[str, Any]] = {}
    for tid, g in tracks_n.groupby("trackId", sort=False):
        gg = g.sort_values("_frame_int")
        frames = gg["_frame_int"].to_numpy(dtype=np.int64)
        lane = pd.to_numeric(gg["laneId"], errors="coerce").to_numpy()
        if dataset == "highD":
            valid = np.isfinite(lane[1:]) & np.isfinite(lane[:-1]) if lane.size > 1 else np.zeros(0, dtype=bool)
            changed = (lane[1:] != lane[:-1]) & valid if lane.size > 1 else np.zeros(0, dtype=bool)
            lc_frames = frames[1:][changed] if lane.size > 1 else np.zeros(0, dtype=np.int64)
            lc_prev_frames = frames[:-1][changed] if lane.size > 1 else np.zeros(0, dtype=np.int64)
        else:
            if "laneChange" in gg.columns:
                lc_col = pd.to_numeric(gg["laneChange"], errors="coerce").fillna(0).to_numpy()
            else:
                lc_col = np.zeros(frames.shape[0], dtype=np.int64)
            lc_frames = frames[lc_col == 1]
            lc_prev_frames = lc_frames
        vcol = "yVelocity" if "yVelocity" in gg.columns else "latVelocity"
        grouped[int(tid)] = {
            "df": gg,
            "frames": frames,
            "lc_frames": lc_frames.astype(np.int64),
            "lc_prev_frames": lc_prev_frames.astype(np.int64),
            "velocity": pd.to_numeric(gg[vcol], errors="coerce").to_numpy() if vcol in gg.columns else None,
        }

    def label_key(track: dict[str, Any], t0_frame: int) -> dict[str, Any]:
        frames = track["frames"]
        t1_frame = int(t0_frame) + win_native
        lc_frames = track["lc_frames"]
        if lc_frames.size == 0:
            return {
                "lc_count": 0,
                "lc_frame": -1,
                "event_label": "lane_following",
                "lc_direction": "none",
                "has_adj_rear_or_alongside": False,
            }

        lo = int(np.searchsorted(lc_frames, t0_frame, side="left"))
        hi = int(np.searchsorted(lc_frames, t1_frame, side="right"))
        cand = lc_frames[lo:hi]
        if dataset == "highD" and cand.size:
            prev = track["lc_prev_frames"][lo:hi]
            cand = cand[prev >= t0_frame]
        if cand.size == 0:
            return {
                "lc_count": 0,
                "lc_frame": -1,
                "event_label": "lane_following",
                "lc_direction": "none",
                "has_adj_rear_or_alongside": False,
            }

        lc_frame = int(cand[0])
        lc_count = int(cand.size)
        velocity = track["velocity"]
        direction = None
        if velocity is not None:
            v_lo = int(np.searchsorted(frames, max(t0_frame, lc_frame - 5), side="left"))
            v_hi = int(np.searchsorted(frames, min(t1_frame, lc_frame + 5), side="right"))
            vals = velocity[v_lo:v_hi]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                mean_v = float(vals.mean())
                if mean_v > 0:
                    direction = "right"
                elif mean_v < 0:
                    direction = "left"

        has_adj = False
        if direction is not None:
            df = track["df"]
            rear_col = "leftRearId" if direction == "left" else "rightRearId"
            alongside_col = "leftAlongsideId" if direction == "left" else "rightAlongsideId"
            p_lo = int(np.searchsorted(frames, max(t0_frame, lc_frame - w_adj), side="left"))
            p_hi = int(np.searchsorted(frames, lc_frame, side="left"))
            pre = df.iloc[p_lo:p_hi]
            for _, row in pre.iterrows():
                ego_lane = row.get("laneId", np.nan)
                if pd.isna(ego_lane):
                    continue
                for col in (rear_col, alongside_col):
                    if col not in pre.columns:
                        continue
                    nb_id = module._to_int_id(row.get(col, 0))
                    if nb_id is None:
                        continue
                    nb_lane = module.get_lane_at(lookup, nb_id, int(row["_frame_int"]))
                    if nb_lane is not None and module.is_adjacent_lane(int(ego_lane), nb_lane):
                        has_adj = True
                        break
                if has_adj:
                    break

        return {
            "lc_count": lc_count,
            "lc_frame": lc_frame,
            "event_label": "lane_change",
            "lc_direction": direction if direction is not None else "unknown",
            "has_adj_rear_or_alongside": bool(has_adj),
        }

    rows = []
    for tid, t0_frame in sorted(set(keys)):
        tid = int(tid)
        t0_frame = int(t0_frame)
        base = {
            "recordingId": int(rec_id),
            "trackId": tid,
            "t0_frame": t0_frame,
            "frameRate": fr,
            "ds_step": ds_step,
            "history_sec": history_sec,
            "future_sec": future_sec,
            "target_hz": target_hz,
        }
        track = grouped.get(tid)
        if track is None:
            rows.append(
                {
                    **base,
                    "event_label": "unknown",
                    "lc_frame": -1,
                    "lc_count": 0,
                    "lc_direction": "unknown",
                    "has_adj_rear_or_alongside": False,
                }
            )
            continue
        frames = track["frames"]
        t1_frame = t0_frame + win_native
        lo = int(np.searchsorted(frames, t0_frame, side="left"))
        if lo >= frames.size or int(frames[lo]) > t1_frame:
            rows.append(
                {
                    **base,
                    "event_label": "unknown",
                    "lc_frame": -1,
                    "lc_count": 0,
                    "lc_direction": "unknown",
                    "has_adj_rear_or_alongside": False,
                }
            )
            continue

        result = label_key(track, t0_frame)
        rows.append({**base, **result})

    by_key = {}
    keep_cols = (
        "recordingId",
        "trackId",
        "t0_frame",
        "frameRate",
        "ds_step",
        "history_sec",
        "future_sec",
        "target_hz",
        "lc_count",
        "lc_frame",
        "event_label",
        "lc_direction",
        "has_adj_rear_or_alongside",
    )
    for row in rows:
        slim = {k: row.get(k) for k in keep_cols}
        by_key[(int(row["trackId"]), int(row["t0_frame"]))] = slim
    return int(rec_id), by_key


def build_labels(
    dataset: str,
    split_root: Path,
    raw_dir: Path,
    *,
    scored_only: bool,
    history_sec: float,
    future_sec: float,
    target_hz: float,
    w_adj: int,
    num_workers: int,
) -> dict[str, Any]:
    arrays = _load_scene_arrays(split_root)
    keys_by_rec = _keys_by_recording(arrays, scored_only=scored_only)
    n_workers = int(num_workers) if int(num_workers) > 0 else (os.cpu_count() or 1)
    label_maps: dict[int, dict[tuple[int, int], dict[str, Any]]] = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as exe:
        futs = [
            exe.submit(
                _process_recording,
                dataset,
                raw_dir,
                rec_id,
                sorted(keys),
                history_sec,
                future_sec,
                target_hz,
                w_adj,
            )
            for rec_id, keys in sorted(keys_by_rec.items())
        ]
        for fut in concurrent.futures.as_completed(futs):
            rec_id, label_map = fut.result()
            label_maps[rec_id] = label_map

    agent_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    agent_ids = arrays["agent_ids"]
    scored = np.asarray(arrays["scored_agent_mask"], dtype=bool)
    include_mask = scored if scored_only else np.asarray(agent_ids >= 0)
    found = 0

    for scene_i in range(agent_ids.shape[0]):
        rec_id = int(arrays["recordingId"][scene_i])
        ego_tid = int(arrays["ego_trackId"][scene_i])
        t0 = int(arrays["t0_frame"][scene_i])
        scene_events = []
        ego_event = "unknown"
        ego_direction = "unknown"
        for agent_i in np.flatnonzero(include_mask[scene_i]):
            tid = int(agent_ids[scene_i, agent_i])
            label = label_maps.get(rec_id, {}).get((tid, t0))
            label_found = label is not None
            found += int(label_found)
            event = str(label.get("event_label", "unknown")) if label else "unknown"
            direction = str(label.get("lc_direction", "unknown")) if label else "unknown"
            if bool(scored[scene_i, agent_i]):
                scene_events.append(event)
            if tid == ego_tid:
                ego_event = event
                ego_direction = direction
            agent_rows.append(
                {
                    "scene_index": scene_i,
                    "source_index": int(arrays["source_index"][scene_i]),
                    "recordingId": rec_id,
                    "ego_trackId": ego_tid,
                    "t0_frame": t0,
                    "agent_index": int(agent_i),
                    "agentId": tid,
                    "is_ego": bool(tid == ego_tid),
                    "is_scored": bool(scored[scene_i, agent_i]),
                    "event_label": event,
                    "lc_direction": direction,
                    "lc_count": int(label.get("lc_count", 0)) if label else 0,
                    "lc_frame": int(label.get("lc_frame", -1)) if label else -1,
                    "has_adj_rear_or_alongside": bool(label.get("has_adj_rear_or_alongside", False)) if label else False,
                    "label_found": label_found,
                }
            )

        known = [e for e in scene_events if e != "unknown"]
        if any(e == "lane_change" for e in known):
            scene_event = "lane_change"
        elif known and all(e == "lane_following" for e in known):
            scene_event = "lane_following"
        else:
            scene_event = "unknown"
        scene_rows.append(
            {
                "scene_index": scene_i,
                "source_index": int(arrays["source_index"][scene_i]),
                "recordingId": rec_id,
                "ego_trackId": ego_tid,
                "t0_frame": t0,
                "num_scored_agents": int(scored[scene_i].sum()),
                "num_scored_lane_change": int(sum(e == "lane_change" for e in scene_events)),
                "scene_event_label": scene_event,
                "ego_event_label": ego_event,
                "ego_lc_direction": ego_direction,
            }
        )

    agent_df = pd.DataFrame(agent_rows)
    scene_df = pd.DataFrame(scene_rows)
    agent_path = split_root / "scenario_labels_agent.csv"
    scene_path = split_root / "scenario_labels_scene.csv"
    agent_df.to_csv(agent_path, index=False)
    scene_df.to_csv(scene_path, index=False)
    report = {
        "dataset": dataset,
        "split_root": str(split_root),
        "raw_dir": str(raw_dir),
        "scored_only": bool(scored_only),
        "num_scenes": int(agent_ids.shape[0]),
        "num_agent_rows": int(len(agent_df)),
        "label_found_rows": int(found),
        "label_found_fraction": float(found / max(1, len(agent_df))),
        "agent_event_counts": agent_df["event_label"].value_counts(dropna=False).to_dict() if len(agent_df) else {},
        "scene_event_counts": scene_df["scene_event_label"].value_counts(dropna=False).to_dict() if len(scene_df) else {},
        "agent_csv": str(agent_path),
        "scene_csv": str(scene_path),
    }
    (split_root / "scenario_label_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=EXPERIMENT_ROOT.parent / "neighformer" / "data")
    parser.add_argument("--multiagent-root", type=Path, default=EXPERIMENT_ROOT / "data")
    parser.add_argument("--dataset", choices=["highD", "exiD", "both"], default="both")
    parser.add_argument("--splits", nargs="+", default=["train_full", "val_full", "test_full"])
    parser.add_argument("--all-agents", action="store_true", help="Label all retained agents instead of scored agents only")
    parser.add_argument("--history-sec", type=float, default=2.0)
    parser.add_argument("--future-sec", type=float, default=5.0)
    parser.add_argument("--target-hz", type=float, default=3.0)
    parser.add_argument("--w-adj", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args(argv)

    datasets = ["highD", "exiD"] if args.dataset == "both" else [args.dataset]
    reports = {}
    for dataset in datasets:
        reports[dataset] = {}
        raw_dir = args.source_root.expanduser().resolve() / dataset / "raw"
        for split in args.splits:
            split_root = args.multiagent_root.expanduser().resolve() / f"{dataset}_multiagent" / split
            reports[dataset][split] = build_labels(
                dataset,
                split_root,
                raw_dir,
                scored_only=not args.all_agents,
                history_sec=args.history_sec,
                future_sec=args.future_sec,
                target_hz=args.target_hz,
                w_adj=args.w_adj,
                num_workers=args.num_workers,
            )
            print(json.dumps(reports[dataset][split], indent=2), flush=True)
    out = args.multiagent_root.expanduser().resolve() / "multiagent_scenario_label_report.json"
    out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
