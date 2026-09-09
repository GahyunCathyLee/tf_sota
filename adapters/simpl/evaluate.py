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
from adapters.mtp_go.metrics import (  # noqa: E402
    MetricAccumulator,
    SampleMetaLookup,
    load_scenario_labels,
    print_latency,
    print_metrics,
    print_scenario_results,
)
from adapters.simpl.dataset import NeighFormerSIMPLDataset  # noqa: E402
from adapters.simpl.train import flatten_simpl_targets, resolve_path  # noqa: E402
from adapters.simpl.upstream import add_upstream_to_path  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--data-root", type=Path)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--device")
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


@torch.no_grad()
def run_evaluate(model, loader, device: torch.device, hz: float, labels: SampleMetaLookup | None):
    model.eval()
    acc = MetricAccumulator(dt=1.0 / hz, hz=hz)
    for data in loader:
        out = model(model.pre_process(data))
        chosen, target, all_modes, valid_mask, scene_rows = flatten_simpl_targets(out, data, device)
        label_rows = None
        if labels is not None and labels.enabled:
            sample_indices = np.asarray(data["SAMPLE_INDEX"], dtype=np.int64)
            scene_labels = labels.lookup(sample_indices)
            label_rows = [scene_labels[int(i)] if scene_labels is not None else None for i in scene_rows]
        acc.update(chosen, target, all_modes=all_modes, valid_mask=valid_mask, labels=label_rows)
    return acc


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
    data_root = resolve_path(cfg["data_root"])
    data_path = dataset_dir(data_root, cfg["dataset"])
    lane_cache_value = args.lane_cache_root or cfg.get("lane_cache_root")
    lane_cache_root = resolve_path(str(lane_cache_value).format(**cfg)) if lane_cache_value else None
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
        lane_radius=args.lane_radius if args.lane_radius is not None else cfg.get("lane_radius", 120.0),
        lane_max_segments=args.lane_max_segments
        if args.lane_max_segments is not None
        else cfg.get("lane_max_segments", 192),
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
    print(f"[INFO] Lanes      : {lane_cache_root if lane_cache_root else 'pseudo fallback'}")
    gpu = f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    print(f"[INFO] Device     : {device}{gpu}")
    print_hparam_summary(
        [
            ("adapter", "SIMPL"),
            ("dataset", cfg.get("dataset")),
            ("feature_mode", cfg.get("feature_mode")),
            ("eval_scope", "ego+neighbors" if ds.has_neighbor_future else "ego_only"),
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
            ("actor_dim", model_cfg.get("d_actor")),
            ("lane_dim", model_cfg.get("d_lane")),
            ("embed_dim", model_cfg.get("d_embed")),
            ("scene_layers", model_cfg.get("n_scene_layer")),
            ("scene_heads", model_cfg.get("n_scene_head")),
            ("num_modes", model_cfg.get("g_num_modes")),
            ("param_out", model_cfg.get("param_out")),
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
        labels_path = args.scenario_labels or (data_path / "scenario_labels.csv")
        labels = SampleMetaLookup(data_path, load_scenario_labels(resolve_path(labels_path)))

    acc = run_evaluate(model, loader, device, float(cfg.get("eval_hz", 3.0)), labels)
    results = acc.result()
    print(f"\n  n_samples = {int(results['n_samples']):,}")
    print_metrics(results)
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
