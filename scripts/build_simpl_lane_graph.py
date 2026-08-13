#!/usr/bin/env python3
"""Build portable SIMPL lane graph caches for highD/exiD.

Examples
--------
python scripts/build_simpl_lane_graph.py --dataset highD --data-root /path/to/data
python scripts/build_simpl_lane_graph.py --dataset exiD --data-root /path/to/data --map-dir data/exiD/maps
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.simpl.lane_graph import build_segment_arrays, interp_polyline, save_cache  # noqa: E402


def _import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pandas is required to build lane graph caches.") from exc
    return pd


def _parse_semicolon_floats(value: Any) -> list[float]:
    if not isinstance(value, str):
        return []
    return [float(x) for x in value.strip().split(";") if x.strip()]


def _find_recording_ids(raw_dir: Path, tracks_name: str = "tracks") -> list[str]:
    pat = re.compile(rf"(\d+)_{tracks_name}\.csv$")
    ids = []
    for path in raw_dir.glob(f"*_{tracks_name}.csv"):
        m = pat.match(path.name)
        if m:
            ids.append(m.group(1))
    return sorted(set(ids))


def _lane_centers(markings: np.ndarray) -> np.ndarray:
    if markings.size < 2:
        return np.zeros((0,), dtype=np.float32)
    return ((markings[:-1] + markings[1:]) * 0.5).astype(np.float32)


def build_highd(data_root: Path, out_root: Path, normalize_upper_xy: bool = True) -> None:
    pd = _import_pandas()
    raw_dir = data_root / "highD" / "raw"
    if not raw_dir.exists():
        raise SystemExit(f"highD raw directory not found: {raw_dir}")
    out_root.mkdir(parents=True, exist_ok=True)
    rec_ids = _find_recording_ids(raw_dir)
    if not rec_ids:
        raise SystemExit(f"No highD recordings found in {raw_dir}")

    built = 0
    for rec_id in rec_ids:
        rec_meta = pd.read_csv(raw_dir / f"{rec_id}_recordingMeta.csv")
        trk_meta = pd.read_csv(raw_dir / f"{rec_id}_tracksMeta.csv")
        tracks = pd.read_csv(raw_dir / f"{rec_id}_tracks.csv")
        upper = np.asarray(_parse_semicolon_floats(str(rec_meta.loc[0, "upperLaneMarkings"])), dtype=np.float32)
        lower = np.asarray(_parse_semicolon_floats(str(rec_meta.loc[0, "lowerLaneMarkings"])), dtype=np.float32)
        c_y = float(upper[-1] + lower[0]) if upper.size and lower.size else 0.0

        id_to_dd = dict(zip(trk_meta["id"].astype(int), trk_meta["drivingDirection"].astype(int)))
        id_to_w = dict(zip(trk_meta["id"].astype(int), trk_meta["width"].astype(float)))
        id_to_h = dict(zip(trk_meta["id"].astype(int), trk_meta["height"].astype(float)))
        vid = tracks["id"].astype(np.int32).to_numpy()
        dd = np.asarray([id_to_dd.get(int(v), 0) for v in vid], dtype=np.int8)
        x = tracks["x"].astype(np.float32).to_numpy().copy()
        y = tracks["y"].astype(np.float32).to_numpy().copy()
        x += 0.5 * np.asarray([id_to_w.get(int(v), 0.0) for v in vid], dtype=np.float32)
        y += 0.5 * np.asarray([id_to_h.get(int(v), 0.0) for v in vid], dtype=np.float32)
        x_max_raw = float(np.nanmax(x)) if x.size else 0.0
        if normalize_upper_xy and upper.size:
            mask = dd == 1
            x = x.copy()
            y = y.copy()
            x[mask] = x_max_raw - x[mask]
            y[mask] = c_y - y[mask]

        x_min = float(np.nanmin(x)) if x.size else 0.0
        y_min = float(np.nanmin(y)) if y.size else 0.0
        x0 = float(np.nanmin(x) - x_min)
        x1 = float(np.nanmax(x) - x_min)
        if x1 <= x0:
            x1 = x0 + 1.0

        centerlines: list[np.ndarray] = []
        upper_for_graph = np.sort((c_y - upper).astype(np.float32)) if normalize_upper_xy and upper.size else upper
        for y_center in _lane_centers(upper_for_graph):
            yy = float(y_center - y_min)
            centerlines.append(np.array([[x0, yy], [x1, yy]], dtype=np.float32))
        for y_center in _lane_centers(lower):
            yy = float(y_center - y_min)
            centerlines.append(np.array([[x0, yy], [x1, yy]], dtype=np.float32))

        arrays = build_segment_arrays(centerlines)
        payload = {
            "dataset": "highD",
            "recording_id": int(rec_id),
            "coordinate_frame": "neighformer_highd_preprocessed",
            "normalize_upper_xy": bool(normalize_upper_xy),
            "segments": arrays["segments"],
            "left": arrays["left"],
            "right": arrays["right"],
            "x_min": x_min,
            "y_min": y_min,
        }
        save_cache(out_root / f"recording_{int(rec_id):02d}.pkl", payload)
        built += 1
        print(f"[highD] recording {rec_id}: {len(centerlines)} centerlines, {arrays['segments'].shape[0]} segments")
    print(f"[DONE] highD lane graph cache -> {out_root} ({built} recordings)")


def _utm_transformer_for(lon: float, lat: float):
    try:
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pyproj is required to build exiD lane graph caches from OSM.") from exc
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)


def _way_points(
    way_id: str,
    ways: dict[str, dict[str, Any]],
    nodes_xy: dict[str, tuple[float, float]],
) -> np.ndarray | None:
    refs = ways[way_id]["refs"]
    pts = [nodes_xy[r] for r in refs if r in nodes_xy]
    if len(pts) < 2:
        return None
    return np.asarray(pts, dtype=np.float32)


def _average_boundaries(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    same_dir = np.linalg.norm(left[0] - right[0]) + np.linalg.norm(left[-1] - right[-1])
    opposite_dir = np.linalg.norm(left[0] - right[-1]) + np.linalg.norm(left[-1] - right[0])
    if opposite_dir < same_dir:
        right = right[::-1]
    _, len_left = _polyline_length(left)
    _, len_right = _polyline_length(right)
    n = max(16, int(max(len_left, len_right) / 2.0) + 1)
    left_i = interp_polyline(left, np.linspace(0.0, len_left, n, dtype=np.float32))
    right_i = interp_polyline(right, np.linspace(0.0, len_right, n, dtype=np.float32))
    return ((left_i + right_i) * 0.5).astype(np.float32)


def _polyline_length(points: np.ndarray) -> tuple[np.ndarray, float]:
    if len(points) < 2:
        return np.zeros(len(points), dtype=np.float32), 0.0
    seg = np.linalg.norm(points[1:] - points[:-1], axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)
    return cum, float(cum[-1])


def _parse_exid_osm(osm_path: Path, origin_x: float, origin_y: float) -> dict[str, Any]:
    tree = ET.parse(osm_path)
    root = tree.getroot()
    raw_nodes: dict[str, tuple[float, float]] = {}
    first_lon = first_lat = None
    for node in root.findall("node"):
        lon = float(node.attrib["lon"])
        lat = float(node.attrib["lat"])
        raw_nodes[node.attrib["id"]] = (lon, lat)
        if first_lon is None:
            first_lon, first_lat = lon, lat
    if first_lon is None or first_lat is None:
        raise SystemExit(f"No nodes in OSM: {osm_path}")
    transformer = _utm_transformer_for(first_lon, first_lat)
    nodes_xy = {}
    for node_id, (lon, lat) in raw_nodes.items():
        x, y = transformer.transform(lon, lat)
        nodes_xy[node_id] = (float(x - origin_x), float(y - origin_y))

    ways: dict[str, dict[str, Any]] = {}
    for way in root.findall("way"):
        tags = {tag.attrib.get("k", ""): tag.attrib.get("v", "") for tag in way.findall("tag")}
        refs = [nd.attrib["ref"] for nd in way.findall("nd")]
        ways[way.attrib["id"]] = {"refs": refs, "tags": tags}

    lanelets = []
    for rel in root.findall("relation"):
        tags = {tag.attrib.get("k", ""): tag.attrib.get("v", "") for tag in rel.findall("tag")}
        if tags.get("type") != "lanelet":
            continue
        left_id = right_id = None
        for member in rel.findall("member"):
            if member.attrib.get("type") != "way":
                continue
            role = member.attrib.get("role")
            if role == "left":
                left_id = member.attrib.get("ref")
            elif role == "right":
                right_id = member.attrib.get("ref")
        if left_id not in ways or right_id not in ways:
            continue
        left = _way_points(left_id, ways, nodes_xy)
        right = _way_points(right_id, ways, nodes_xy)
        if left is None or right is None:
            continue
        lanelets.append(
            {
                "id": int(rel.attrib["id"]),
                "left_way": left_id,
                "right_way": right_id,
                "centerline": _average_boundaries(left, right),
            }
        )

    left_flags = []
    right_flags = []
    for lanelet in lanelets:
        has_left = any(other["right_way"] == lanelet["left_way"] for other in lanelets)
        has_right = any(other["left_way"] == lanelet["right_way"] for other in lanelets)
        left_flags.append(1.0 if has_left else 0.0)
        right_flags.append(1.0 if has_right else 0.0)
    return {
        "lanelets": lanelets,
        "centerlines": [x["centerline"] for x in lanelets],
        "left_flags": left_flags,
        "right_flags": right_flags,
    }


def build_exid(data_root: Path, map_dir: Path, out_root: Path, mmap_name: str = "dimI") -> None:
    pd = _import_pandas()
    raw_dir = data_root / "exiD" / "raw"
    if not raw_dir.exists():
        raise SystemExit(f"exiD raw directory not found: {raw_dir}")
    if not map_dir.exists():
        raise SystemExit(f"exiD map directory not found: {map_dir}")
    out_root.mkdir(parents=True, exist_ok=True)
    rec_ids = _find_recording_ids(raw_dir)
    if not rec_ids:
        raise SystemExit(f"No exiD recordings found in {raw_dir}")

    map_cache: dict[int, dict[str, Any]] = {}
    rec_info: dict[int, dict[str, Any]] = {}
    built = 0
    for rec_id in rec_ids:
        rec_meta = pd.read_csv(raw_dir / f"{rec_id}_recordingMeta.csv")
        tracks = pd.read_csv(raw_dir / f"{rec_id}_tracks.csv", low_memory=False)
        location_id = int(rec_meta.loc[0, "locationId"])
        origin_x = float(rec_meta.loc[0, "xUtmOrigin"])
        origin_y = float(rec_meta.loc[0, "yUtmOrigin"])
        if location_id not in map_cache:
            osm_path = map_dir / f"location{location_id}.osm"
            map_cache[location_id] = _parse_exid_osm(osm_path, origin_x, origin_y)

        x = tracks["xCenter"].astype(np.float32).to_numpy()
        y = tracks["yCenter"].astype(np.float32).to_numpy()
        x_min = float(np.nanmin(x)) if x.size else 0.0
        y_min = float(np.nanmin(y)) if y.size else 0.0
        shifted = [(line - np.array([x_min, y_min], dtype=np.float32)) for line in map_cache[location_id]["centerlines"]]
        arrays = build_segment_arrays(shifted, map_cache[location_id]["left_flags"], map_cache[location_id]["right_flags"])
        payload = {
            "dataset": "exiD",
            "recording_id": int(rec_id),
            "location_id": location_id,
            "coordinate_frame": "neighformer_exid_recording_preprocessed",
            "segments": arrays["segments"],
            "left": arrays["left"],
            "right": arrays["right"],
            "x_min": x_min,
            "y_min": y_min,
        }
        save_cache(out_root / f"recording_{int(rec_id):02d}.pkl", payload)
        rec_info[int(rec_id)] = {"location_id": location_id}
        built += 1
        print(
            f"[exiD] recording {rec_id}: location={location_id}, "
            f"{len(shifted)} lanelets, {arrays['segments'].shape[0]} segments"
        )

    mmap_dir = data_root / "exiD" / mmap_name
    if mmap_dir.exists() and (mmap_dir / "meta_recordingId.npy").exists():
        _build_exid_sample_pose(raw_dir, mmap_dir, out_root, mmap_name, rec_info)
    else:
        print(f"[WARN] mmap directory not found for sample pose cache: {mmap_dir}")
    print(f"[DONE] exiD lane graph cache -> {out_root} ({built} recordings)")


def _build_exid_sample_pose(
    raw_dir: Path,
    mmap_dir: Path,
    out_root: Path,
    mmap_name: str,
    rec_info: dict[int, dict[str, Any]],
) -> None:
    pd = _import_pandas()
    rec_arr = np.load(mmap_dir / "meta_recordingId.npy", mmap_mode="r")
    track_arr = np.load(mmap_dir / "meta_trackId.npy", mmap_mode="r")
    frame_arr = np.load(mmap_dir / "meta_frame.npy", mmap_mode="r")
    ref_heading = np.zeros(rec_arr.shape[0], dtype=np.float32)
    location_id = np.full(rec_arr.shape[0], -1, dtype=np.int16)

    for rec_id in sorted(set(int(x) for x in rec_arr.tolist())):
        tracks = pd.read_csv(raw_dir / f"{rec_id:02d}_tracks.csv", usecols=["trackId", "frame", "heading"])
        lookup = {
            (int(tid), int(frame)): math.radians(float(heading))
            for tid, frame, heading in zip(tracks["trackId"], tracks["frame"], tracks["heading"])
        }
        rows = np.flatnonzero(rec_arr == rec_id)
        for row in rows:
            ref_heading[row] = float(lookup.get((int(track_arr[row]), int(frame_arr[row])), 0.0))
        location_id[rows] = int(rec_info.get(rec_id, {}).get("location_id", -1))
        print(f"[exiD] sample pose recording {rec_id:02d}: {rows.size} samples")

    out_path = out_root / f"sample_pose_{mmap_name}.npz"
    np.savez_compressed(out_path, ref_heading=ref_heading, location_id=location_id)
    print(f"[exiD] sample pose cache -> {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["highD", "exiD", "all"], required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--map-dir", type=Path, default=Path("data/exiD/maps"))
    parser.add_argument("--mmap-name", default="dimI")
    parser.add_argument("--no-normalize-upper-xy", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    map_dir = args.map_dir.expanduser()
    if not map_dir.is_absolute():
        map_dir = (ROOT / map_dir).resolve()
    if args.dataset in ("highD", "all"):
        out = args.out_root or (data_root / "highD" / "simpl_lane_graph")
        build_highd(data_root, out.expanduser().resolve(), normalize_upper_xy=not args.no_normalize_upper_xy)
    if args.dataset in ("exiD", "all"):
        out = args.out_root or (data_root / "exiD" / "simpl_lane_graph")
        build_exid(data_root, map_dir, out.expanduser().resolve(), mmap_name=args.mmap_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
