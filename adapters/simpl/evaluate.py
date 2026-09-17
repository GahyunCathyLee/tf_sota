#!/usr/bin/env python3
"""Evaluate a SIMPL adapter checkpoint with NeighFormer-compatible metrics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ADAPTER_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ADAPTER_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from adapters.common import dataset_dir, print_hparam_summary, split_indices_path  # noqa: E402
from adapters.multiagent_common import multiagent_indices, multiagent_split_dir  # noqa: E402
from adapters.mtp_go.metrics import (  # noqa: E402
    MetricAccumulator,
    SampleMetaLookup,
    SceneMetricAccumulator,
    load_scenario_labels,
    print_latency,
    print_metrics,
    print_scene_metrics,
    print_scenario_results,
)
from adapters.simpl.dataset import NeighFormerSIMPLDataset  # noqa: E402
from adapters.simpl.multiagent_dataset import MultiAgentSIMPLDataset  # noqa: E402
from adapters.simpl.train import flatten_simpl_targets, resolve_path, trainable_parameter_count  # noqa: E402
from adapters.simpl.upstream import add_upstream_to_path  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--data-root", type=Path)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--device")
    p.add_argument("--multiagent", action="store_true", help="Use data/{dataset}_multiagent/{split}_full arrays")
    p.add_argument("--scenario", action="store_true")
    p.add_argument("--scenario-labels", type=Path)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--lane-cache-root", type=Path)
    p.add_argument("--lane-radius", type=float)
    p.add_argument("--lane-max-segments", type=int)
    p.add_argument("--measure-time", action="store_true")
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--upstream-dir", type=Path)
    p.add_argument("--output-json", type=Path)
    return p.parse_args(argv)


def load_multiagent_scene_labels(path: Path) -> dict[int, dict[str, str]] | None:
    """scenario_labels_scene.csv -> {scene_index: {event_label, state_label}}."""
    import pandas as pd

    path = Path(path)
    if not path.exists():
        print(f"[WARN] scenario_labels_scene not found: {path} -> scenario breakdown disabled")
        return None
    df = pd.read_csv(path)
    if "scene_index" not in df.columns:
        print("[WARN] scenario_labels_scene missing scene_index -> disabled")
        return None
    event_col = "scene_event_label" if "scene_event_label" in df.columns else "event_label"
    if event_col not in df.columns:
        print("[WARN] scenario_labels_scene has no scene_event_label/event_label -> disabled")
        return None
    out: dict[int, dict[str, str]] = {}
    for row in df.itertuples(index=False):
        out[int(getattr(row, "scene_index"))] = {
            "event_label": str(getattr(row, event_col)),
            "state_label": "multiagent",
        }
    return out


class SceneIndexLookup:
    """scene index -> scene-level scenario label for multi-agent splits."""

    def __init__(self, labels_lut: dict[int, dict[str, str]] | None) -> None:
        self.labels_lut = labels_lut

    @property
    def enabled(self) -> bool:
        return self.labels_lut is not None

    def lookup(self, sample_indices: np.ndarray) -> list[dict[str, str] | None] | None:
        if not self.enabled:
            return None
        return [self.labels_lut.get(int(i)) for i in np.asarray(sample_indices, dtype=np.int64).reshape(-1)]


@torch.no_grad()
def run_evaluate(model, loader, device: torch.device, hz: float, labels: SampleMetaLookup | None):
    model.eval()
    acc = MetricAccumulator(dt=1.0 / hz, hz=hz)
    scene_acc = SceneMetricAccumulator()
    for data in loader:
        out = model(model.pre_process(data))
        chosen, target, all_modes, valid_mask, scene_rows = flatten_simpl_targets(out, data, device)
        label_rows = None
        if labels is not None and labels.enabled:
            sample_indices = np.asarray(data["SAMPLE_INDEX"], dtype=np.int64)
            scene_labels = labels.lookup(sample_indices)
            label_rows = [scene_labels[int(i)] if scene_labels is not None else None for i in scene_rows]
        acc.update(chosen, target, all_modes=all_modes, valid_mask=valid_mask, labels=label_rows)
        scene_acc.update(all_modes, target, valid_mask, scene_rows)
    return acc, scene_acc


def measure_latency(fn, device: torch.device, warmup: int, iters: int) -> dict[str, float]:
    print(f"  Warm-up      : {warmup:,} iters ...", end=" ", flush=True)
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("done")
    times = []
    print(f"  Measurement  : {iters:,} iters ...", end=" ", flush=True)
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            starter.record()
            fn()
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0)
    print("done")
    arr = np.asarray(times, dtype=np.float64)
    return {"avg_ms": float(arr.mean()), "min_ms": float(arr.min()), "max_ms": float(arr.max())}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ckpt_path = args.ckpt if args.ckpt.is_absolute() else resolve_path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: dict[str, Any] = ckpt["cfg"]
    model_cfg: dict[str, Any] = ckpt["model_cfg"]
    upstream_dir = add_upstream_to_path(args.upstream_dir or cfg.get("upstream_dir"))
    from simpl.simpl import Simpl  # noqa: WPS433

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.data_root:
        cfg = dict(cfg)
        cfg["data_root"] = str(args.data_root)
    use_multiagent = bool(args.multiagent or cfg.get("multiagent"))
    data_root = resolve_path(cfg["data_root"])
    lane_cache_value = args.lane_cache_root or cfg.get("lane_cache_root")
    lane_cache_root = resolve_path(str(lane_cache_value).format(**cfg)) if lane_cache_value else None
    lane_cache_exists = bool(lane_cache_root and lane_cache_root.exists())
    lane_radius = args.lane_radius if args.lane_radius is not None else cfg.get("lane_radius", 120.0)
    lane_max_segments = args.lane_max_segments if args.lane_max_segments is not None else cfg.get("lane_max_segments", 192)
    if use_multiagent:
        split_path = multiagent_split_dir(data_root, cfg["dataset"], args.split)
        indices = multiagent_indices(split_path, args.max_samples)
        ds = MultiAgentSIMPLDataset(
            split_path,
            cfg["dataset"],
            args.split,
            indices=indices,
            lane_half_length=cfg["lane_half_length"],
            lane_cache_root=lane_cache_root if lane_cache_exists else None,
            lane_radius=lane_radius,
            lane_max_segments=lane_max_segments,
        )
        data_path = split_path
    else:
        data_path = dataset_dir(data_root, cfg["dataset"])
        indices = np.load(split_indices_path(data_root, cfg["dataset"], args.split))
        if args.max_samples is not None:
            indices = indices[: args.max_samples]
        ds = NeighFormerSIMPLDataset(
            data_path,
            indices,
            cfg["dataset"],
            cfg["feature_mode"],
            args.split,
            cfg["lane_half_length"],
            lane_cache_root=lane_cache_root,
            lane_radius=lane_radius,
            lane_max_segments=lane_max_segments,
        )
    batch_size = args.batch_size or int(cfg["batch_size"])
    num_workers = args.num_workers if args.num_workers is not None else int(cfg["num_workers"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=ds.collate_fn, persistent_workers=num_workers > 0)

    model = Simpl(model_cfg, device).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[INFO] Checkpoint : {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")
    print(f"[INFO] Upstream   : {upstream_dir}")
    print(f"[INFO] Dataset    : {args.split} split  n={len(ds):,}  {cfg['dataset']} {cfg['feature_mode']}")
    print(f"[INFO] Source     : {'multiagent' if use_multiagent else 'single-agent'}")
    print(f"[INFO] Lanes      : {lane_cache_root if lane_cache_root else 'pseudo fallback'}")
    print(f"[INFO] Lane cache : exists={lane_cache_exists}  source={'cached lanes' if lane_cache_exists else 'pseudo fallback'}")
    gpu = f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    print(f"[INFO] Device     : {device}{gpu}")
    print(f"[INFO] Metrics    : most-likely mode by class argmax; minADE/minFDE also reported over all modes")
    print_hparam_summary(
        [
            ("adapter", "SIMPL"),
            ("dataset", cfg.get("dataset")),
            ("feature_mode", cfg.get("feature_mode")),
            ("eval_scope", "multiagent_scene" if use_multiagent else ("ego+neighbors" if ds.has_neighbor_future else "ego_only")),
            ("checkpoint_epoch", ckpt.get("epoch")),
            ("epochs", cfg.get("epochs")),
            ("batch_size", cfg.get("batch_size")),
            ("learning_rate", cfg.get("lr")),
            ("weight_decay", cfg.get("weight_decay")),
            ("grad_clip_norm", cfg.get("grad_clip_norm")),
            ("seed", cfg.get("seed")),
            ("history_steps", model_cfg.get("g_obs_len")),
            ("future_steps", model_cfg.get("g_pred_len")),
            ("eval_hz", cfg.get("eval_hz", 3.0)),
            ("actor_input_dim", model_cfg.get("in_actor")),
            ("actor_features", getattr(ds, "actor_feature_names", None)),
            ("actor_dim", model_cfg.get("d_actor")),
            ("lane_dim", model_cfg.get("d_lane")),
            ("embed_dim", model_cfg.get("d_embed")),
            ("scene_layers", model_cfg.get("n_scene_layer")),
            ("scene_heads", model_cfg.get("n_scene_head")),
            ("num_modes", model_cfg.get("g_num_modes")),
            ("param_out", model_cfg.get("param_out")),
            ("trainable_params", trainable_parameter_count(model)),
        ]
    )

    if args.measure_time:
        sample_loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=ds.collate_fn)
        sample = next(iter(sample_loader))

        def infer_one():
            model(model.pre_process(sample))

        lat = measure_latency(infer_one, device, args.warmup, args.iters)
        print_latency(lat, batch_size=1, warmup=args.warmup, iters=args.iters)
        return 0

    labels = None
    if args.scenario:
        if use_multiagent:
            labels_path = args.scenario_labels or (data_path / "scenario_labels_scene.csv")
            labels = SceneIndexLookup(load_multiagent_scene_labels(resolve_path(labels_path)))
        else:
            labels_path = args.scenario_labels or (data_path / "scenario_labels.csv")
            labels = SampleMetaLookup(data_path, load_scenario_labels(resolve_path(labels_path)))

    acc, scene_acc = run_evaluate(model, loader, device, float(cfg.get("eval_hz", 3.0)), labels)
    results = acc.result()
    results.update(scene_acc.result())
    print(f"\n  n_samples = {int(results['n_samples']):,}")
    print_metrics(results)
    print_scene_metrics(results)
    if labels is not None and acc.has_scenario:
        print_scenario_results(acc.event_stats, "Event")
        print_scenario_results(acc.state_stats, "State")
    if args.output_json:
        out = args.output_json if args.output_json.is_absolute() else resolve_path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
