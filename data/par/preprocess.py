#!/usr/bin/env python3
"""Build PAR-specific highD/exiD arrays without modifying NeighFormer outputs.

The canonical NeighFormer preprocessing remains the source of sample order,
splits, baseline/dimI neighbour features, and ego targets.  This script creates
a repo-local PAR data root:

    data/par/<dataset>/dimI/
    data/par/<dataset>/splits/

The common arrays are symlinked from an existing canonical data root, while PAR
extra arrays are generated from the same raw CSV files:

    nb_ids.npy          int32   [N, K]
    nb_attr.npy         float32 [N, K, 2]       # [dim, I] from t0 slot
    nb_attr_mask.npy    bool    [N, K]
    x_nb_abs.npy        float32 [N, T, K, 6]    # fixed-ID neighbour history
    x_nb_abs_mask.npy   bool    [N, T, K]
    y_nb.npy            float32 [N, Tf, K, 6]   # fixed-ID neighbour future
    y_nb_mask.npy       bool    [N, Tf, K]

The fixed neighbour identity for each slot is taken from the ego row at
``t0_frame`` (the last history frame) and only if the canonical ``nb_mask`` says
that slot is active at t0.  This keeps the existing baseline/dimI contract
intact while giving PAR labelled neighbour futures for multi-agent token loss.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from tqdm import tqdm


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = EXPERIMENT_ROOT.parent / "neighformer" / "data"

COMMON_DIMI_FILES = (
    "x_ego.npy",
    "x_nb.npy",
    "nb_mask.npy",
    "y.npy",
    "y_vel.npy",
    "y_acc.npy",
    "x_last_abs.npy",
    "meta_recordingId.npy",
    "meta_trackId.npy",
    "meta_frame.npy",
    "scenario_labels.csv",
)
SPLIT_FILES = ("train_indices.npy", "val_indices.npy", "test_indices.npy")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HIGH_D = _load_module("sota_highd_preprocess", EXPERIMENT_ROOT / "data" / "highD" / "preprocess.py")
EXI_D = _load_module("sota_exid_preprocess", EXPERIMENT_ROOT / "data" / "exiD" / "preprocess.py")


@dataclass(frozen=True)
class BuildPaths:
    dataset: str
    source_root: Path
    output_root: Path
    raw_root: Path | None

    @property
    def source_dimI(self) -> Path:
        return self.source_root / self.dataset / "dimI"

    @property
    def source_splits(self) -> Path:
        return self.source_root / self.dataset / "splits"

    @property
    def raw_dir(self) -> Path:
        return self.raw_root if self.raw_root is not None else self.source_root / self.dataset / "raw"

    @property
    def output_dimI(self) -> Path:
        return self.output_root / self.dataset / "dimI"

    @property
    def output_splits(self) -> Path:
        return self.output_root / self.dataset / "splits"


def _link_or_skip(src: Path, dst: Path, overwrite_links: bool = True) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and overwrite_links:
            dst.unlink()
        else:
            return
    rel = os.path.relpath(src.resolve(), dst.parent.resolve())
    dst.symlink_to(rel)


def prepare_output_tree(paths: BuildPaths) -> None:
    paths.output_dimI.mkdir(parents=True, exist_ok=True)
    paths.output_splits.mkdir(parents=True, exist_ok=True)
    for name in COMMON_DIMI_FILES:
        _link_or_skip(paths.source_dimI / name, paths.output_dimI / name)
    for name in SPLIT_FILES:
        _link_or_skip(paths.source_splits / name, paths.output_splits / name)


def _read_source_arrays(dimI: Path) -> dict[str, np.ndarray]:
    required = ["x_ego", "x_nb", "nb_mask", "y", "x_last_abs", "meta_recordingId", "meta_trackId", "meta_frame"]
    arrays = {}
    for name in required:
        path = dimI / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        arrays[name] = np.load(path, mmap_mode="r")
    return arrays


def _rec_file_id(rec_id: int) -> str:
    return f"{int(rec_id):02d}"


def _safe_numeric_series(s: pd.Series, default: int = -1) -> np.ndarray:
    return pd.to_numeric(s.astype(str).str.strip().str.split(";").str[0], errors="coerce").fillna(default).astype(np.int32).to_numpy()


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
    vid_to_w = dict(zip(trk_meta["id"].astype(int), trk_meta["width"].astype(float)))
    vid_to_l = dict(zip(trk_meta["id"].astype(int), trk_meta["height"].astype(float)))

    frame = tracks["frame"].astype(np.int32).to_numpy()
    vid = tracks["id"].astype(np.int32).to_numpy()
    x = tracks["x"].astype(np.float32).to_numpy().copy()
    y = tracks["y"].astype(np.float32).to_numpy().copy()
    w_row = np.array([vid_to_w.get(int(v), 0.0) for v in vid], np.float32)
    h_row = np.array([vid_to_l.get(int(v), 0.0) for v in vid], np.float32)
    x += 0.5 * w_row
    y += 0.5 * h_row
    xv = tracks["xVelocity"].astype(np.float32).to_numpy()
    yv = tracks["yVelocity"].astype(np.float32).to_numpy()
    xa = tracks["xAcceleration"].astype(np.float32).to_numpy()
    ya = tracks["yAcceleration"].astype(np.float32).to_numpy()
    lane_id = tracks["laneId"].astype(np.int16).to_numpy()
    dd = np.array([vid_to_dd.get(int(v), 0) for v in vid], np.int8)
    x_max = float(np.nanmax(x)) if len(x) else 0.0

    upper_for_calc = upper_mark.copy()
    upper_center, _ = HIGH_D.build_lane_tables(upper_for_calc)
    upper_mm = (1, int(len(upper_center))) if len(upper_center) else None
    x, y, xv, yv, xa, ya, lane_id = HIGH_D.maybe_flip(x, y, xv, yv, xa, ya, lane_id, dd, c_y, x_max, upper_mm)
    x_min = float(np.nanmin(x)) if x.size else 0.0
    y_min = float(np.nanmin(y)) if y.size else 0.0
    x = (x - x_min).astype(np.float32)
    y = (y - y_min).astype(np.float32)

    per_vid_frame_to_row: dict[int, dict[int, int]] = {}
    for v, idxs in tracks.groupby("id").indices.items():
        idxs = np.array(idxs, np.int32)
        idxs = idxs[np.argsort(frame[idxs])]
        per_vid_frame_to_row[int(v)] = {int(fr): int(r) for fr, r in zip(frame[idxs], idxs)}
    nb_ids_all = np.stack([tracks[c].astype(np.int32).to_numpy() for c in HIGH_D.NEIGHBOR_COLS_8], axis=1)
    return {
        "step": step,
        "frame_to_row": per_vid_frame_to_row,
        "nb_ids_all": nb_ids_all,
        "x": x,
        "y": y,
        "xv": xv,
        "yv": yv,
        "xa": xa,
        "ya": ya,
    }


def _highd_state(ctx: dict[str, Any], tid: int, frame: int, ref_x: float, ref_y: float) -> np.ndarray | None:
    row = ctx["frame_to_row"].get(int(tid), {}).get(int(frame))
    if row is None:
        return None
    return np.array(
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

    tracks = tracks.sort_values(["trackId", "frame"], kind="mergesort").reset_index(drop=True)
    frame = tracks["frame"].astype(np.int32).to_numpy()
    vid = tracks["trackId"].astype(np.int32).to_numpy()
    x = tracks["xCenter"].astype(np.float32).to_numpy().copy()
    y = tracks["yCenter"].astype(np.float32).to_numpy().copy()
    xv = tracks["lonVelocity"].astype(np.float32).to_numpy()
    yv = tracks["latVelocity"].astype(np.float32).to_numpy()
    xa = tracks["lonAcceleration"].astype(np.float32).to_numpy()
    ya = tracks["latAcceleration"].astype(np.float32).to_numpy()
    heading_deg = tracks["heading"].astype(np.float32).to_numpy() if "heading" in tracks.columns else np.zeros(len(tracks), np.float32)
    heading_rad = np.deg2rad(heading_deg).astype(np.float32)

    x_min = float(np.nanmin(x)) if x.size else 0.0
    y_min = float(np.nanmin(y)) if y.size else 0.0
    x = (x - x_min).astype(np.float32)
    y = (y - y_min).astype(np.float32)

    per_vid_frame_to_row: dict[int, dict[int, int]] = {}
    for v, idxs in tracks.groupby("trackId").indices.items():
        idxs = np.array(idxs, np.int32)
        idxs = idxs[np.argsort(frame[idxs])]
        per_vid_frame_to_row[int(v)] = {int(fr): int(r) for fr, r in zip(frame[idxs], idxs)}
    per_vid_frame_to_hdg = {
        int(v): {int(frame[r]): float(heading_rad[r]) for r in idxs}
        for v, idxs in tracks.groupby("trackId").indices.items()
    }
    nb_ids_all = np.stack([_safe_numeric_series(tracks[c]) for c in EXI_D.NEIGHBOR_COLS_8], axis=1)
    return {
        "step": step,
        "frame_to_row": per_vid_frame_to_row,
        "frame_to_hdg": per_vid_frame_to_hdg,
        "nb_ids_all": nb_ids_all,
        "x": x,
        "y": y,
        "xv": xv,
        "yv": yv,
        "xa": xa,
        "ya": ya,
    }


def _exid_state(ctx: dict[str, Any], tid: int, frame: int, ref_x: float, ref_y: float, ref_hdg: float) -> np.ndarray | None:
    row = ctx["frame_to_row"].get(int(tid), {}).get(int(frame))
    if row is None:
        return None
    hdg = float(ctx["frame_to_hdg"].get(int(tid), {}).get(int(frame), ref_hdg))
    px, py = EXI_D._norm_pos(float(ctx["x"][row]), float(ctx["y"][row]), ref_x, ref_y, ref_hdg)
    vx, vy = EXI_D._local_to_norm_frame(float(ctx["xv"][row]), float(ctx["yv"][row]), hdg, ref_hdg)
    ax, ay = EXI_D._local_to_norm_frame(float(ctx["xa"][row]), float(ctx["ya"][row]), hdg, ref_hdg)
    return np.array([px, py, vx, vy, ax, ay], dtype=np.float32)


def _allocate_outputs(out_dir: Path, n: int, hist: int, fut: int, k: int) -> dict[str, np.ndarray]:
    return {
        "nb_ids": open_memmap(out_dir / "nb_ids.npy", "w+", "int32", (n, k)),
        "nb_attr": open_memmap(out_dir / "nb_attr.npy", "w+", "float32", (n, k, 2)),
        "nb_attr_mask": open_memmap(out_dir / "nb_attr_mask.npy", "w+", "bool", (n, k)),
        "x_nb_abs": open_memmap(out_dir / "x_nb_abs.npy", "w+", "float32", (n, hist, k, 6)),
        "x_nb_abs_mask": open_memmap(out_dir / "x_nb_abs_mask.npy", "w+", "bool", (n, hist, k)),
        "y_nb": open_memmap(out_dir / "y_nb.npy", "w+", "float32", (n, fut, k, 6)),
        "y_nb_mask": open_memmap(out_dir / "y_nb_mask.npy", "w+", "bool", (n, fut, k)),
    }


def build_dataset(paths: BuildPaths, target_hz: float, overwrite: bool = False, max_samples: int | None = None) -> dict[str, Any]:
    prepare_output_tree(paths)
    arrays = _read_source_arrays(paths.source_dimI)
    n = int(arrays["x_ego"].shape[0])
    hist = int(arrays["x_ego"].shape[1])
    fut = int(arrays["y"].shape[1])
    k = int(arrays["x_nb"].shape[2])
    out_dir = paths.output_dimI
    done_marker = out_dir / "par_preprocess_report.json"
    if done_marker.exists() and not overwrite:
        raise FileExistsError(f"{done_marker} exists; pass --overwrite to rebuild PAR arrays")
    fps = _allocate_outputs(out_dir, n, hist, fut, k)
    for arr in fps.values():
        arr[...] = 0

    meta_rec = arrays["meta_recordingId"]
    meta_track = arrays["meta_trackId"]
    meta_frame = arrays["meta_frame"]
    selected_indices = np.arange(n, dtype=np.int64)
    if max_samples is not None:
        selected_indices = selected_indices[: int(max_samples)]
    rec_ids = sorted(int(v) for v in np.unique(meta_rec[selected_indices]))
    built = 0
    active_slots = 0
    future_points = 0

    for rec_id in tqdm(rec_ids, desc=f"{paths.dataset} recordings"):
        ctx = (
            _build_highd_context(paths.raw_dir, rec_id, target_hz)
            if paths.dataset == "highD"
            else _build_exid_context(paths.raw_dir, rec_id, target_hz)
        )
        indices = np.intersect1d(np.flatnonzero(meta_rec == rec_id), selected_indices, assume_unique=True)
        for idx in tqdm(indices, desc=f"rec {_rec_file_id(rec_id)}", leave=False):
            tid = int(meta_track[idx])
            t0 = int(meta_frame[idx])
            step = int(ctx["step"])
            hist_frames = [t0 - (hist - 1 - i) * step for i in range(hist)]
            fut_frames = [t0 + (i + 1) * step for i in range(fut)]
            ego_row = ctx["frame_to_row"].get(tid, {}).get(t0)
            if ego_row is None:
                continue
            ref_x = float(arrays["x_last_abs"][idx, 0])
            ref_y = float(arrays["x_last_abs"][idx, 1])
            ref_hdg = 0.0
            if paths.dataset == "exiD":
                ref_hdg = float(ctx["frame_to_hdg"].get(tid, {}).get(t0, 0.0))
            ids8 = ctx["nb_ids_all"][ego_row]
            active = np.asarray(arrays["nb_mask"][idx, -1], dtype=bool)
            attr = np.asarray(arrays["x_nb"][idx, -1, :, 8:10], dtype=np.float32)

            for slot in range(k):
                if not active[slot]:
                    continue
                nid = int(ids8[slot])
                if nid <= 0:
                    continue
                fps["nb_ids"][idx, slot] = nid
                fps["nb_attr"][idx, slot] = attr[slot]
                fps["nb_attr_mask"][idx, slot] = True
                active_slots += 1

                for ti, frame in enumerate(hist_frames):
                    state = (
                        _highd_state(ctx, nid, frame, ref_x, ref_y)
                        if paths.dataset == "highD"
                        else _exid_state(ctx, nid, frame, ref_x, ref_y, ref_hdg)
                    )
                    if state is not None:
                        fps["x_nb_abs"][idx, ti, slot] = state
                        fps["x_nb_abs_mask"][idx, ti, slot] = True
                for fi, frame in enumerate(fut_frames):
                    state = (
                        _highd_state(ctx, nid, frame, ref_x, ref_y)
                        if paths.dataset == "highD"
                        else _exid_state(ctx, nid, frame, ref_x, ref_y, ref_hdg)
                    )
                    if state is not None:
                        fps["y_nb"][idx, fi, slot] = state
                        fps["y_nb_mask"][idx, fi, slot] = True
                        future_points += 1
            built += 1

    for arr in fps.values():
        arr.flush()
    report = {
        "dataset": paths.dataset,
        "source_dimI": str(paths.source_dimI),
        "raw_dir": str(paths.raw_dir),
        "output_dimI": str(paths.output_dimI),
        "num_samples": n,
        "history_len": hist,
        "future_len": fut,
        "max_neighbors": k,
        "samples_visited": built,
        "partial": max_samples is not None,
        "max_samples": max_samples,
        "active_t0_slots": active_slots,
        "neighbor_future_points": future_points,
        "arrays": {name: list(arr.shape) for name, arr in fps.items()},
    }
    done_marker.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", choices=["highD", "exiD", "both"], default="both")
    parser.add_argument("--source-data-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT / "data" / "par")
    parser.add_argument("--raw-root", type=Path, default=None, help="Override raw dir for a single-dataset run")
    parser.add_argument("--target-hz", type=float, default=3.0)
    parser.add_argument("--max-samples", type=int, default=None, help="Debug/smoke: only populate the first N samples")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = ["highD", "exiD"] if args.dataset == "both" else [args.dataset]
    reports = []
    for dataset in datasets:
        raw_root = args.raw_root
        if raw_root is not None and len(datasets) > 1:
            raise SystemExit("--raw-root can only be used with --dataset highD or --dataset exiD")
        paths = BuildPaths(
            dataset=dataset,
            source_root=args.source_data_root.resolve(),
            output_root=args.output_root.resolve(),
            raw_root=raw_root.resolve() if raw_root is not None else None,
        )
        reports.append(
            build_dataset(
                paths,
                target_hz=float(args.target_hz),
                overwrite=bool(args.overwrite),
                max_samples=args.max_samples,
            )
        )
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
