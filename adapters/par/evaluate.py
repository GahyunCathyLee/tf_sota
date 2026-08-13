#!/usr/bin/env python3
"""Evaluate a PAR adapter checkpoint with NeighFormer-compatible metrics."""

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

from adapters.common import dataset_dir, split_indices_path  # noqa: E402
from adapters.mtp_go.metrics import (  # noqa: E402
    MetricAccumulator,
    SampleMetaLookup,
    load_scenario_labels,
    print_latency,
    print_metrics,
    print_scenario_results,
)
from adapters.par.dataset import NeighFormerPARDataset  # noqa: E402
from adapters.par.model import PARTrajectoryModel  # noqa: E402
from adapters.par.train import build_model, move_batch, predict_batch, resolve_path  # noqa: E402
from adapters.par.upstream import add_upstream_to_path  # noqa: E402


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
    p.add_argument("--measure-time", action="store_true")
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--upstream-dir", type=Path)
    p.add_argument("--output-json", type=Path)
    return p.parse_args(argv)


@torch.no_grad()
def run_evaluate(
    model: PARTrajectoryModel,
    loader: DataLoader,
    ds: NeighFormerPARDataset,
    device: torch.device,
    cfg: dict[str, Any],
    labels: SampleMetaLookup | None,
) -> tuple[MetricAccumulator, dict[str, float]]:
    model.eval()
    acc = MetricAccumulator(dt=float(cfg.get("dt", 1.0 / float(cfg["eval_hz"]))), hz=float(cfg["eval_hz"]))
    total_loss = 0.0
    total_batches = 0
    for raw in loader:
        batch = move_batch(raw, device)
        loss, _ = model.loss(batch)
        pred = predict_batch(model, batch, ds, bool(cfg.get("multinomial_sampling", False)))
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        sample_indices = raw["sample_index"].detach().cpu().numpy()
        label_rows = labels.lookup(sample_indices) if labels is not None and labels.enabled else None
        acc.update(pred, batch["target"], labels=label_rows)
        total_loss += float(loss.detach())
        total_batches += 1
    return acc, {"loss": total_loss / max(1, total_batches)}


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
    upstream_dir = add_upstream_to_path(args.upstream_dir or cfg.get("upstream_dir"))

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = resolve_path(args.data_root) if args.data_root else resolve_path(cfg["data_root"])
    data_path = dataset_dir(data_root, cfg["dataset"])
    indices = np.load(split_indices_path(data_root, cfg["dataset"], args.split))
    if args.max_samples is not None:
        indices = indices[: args.max_samples]
    ds = NeighFormerPARDataset(
        data_path,
        indices,
        cfg["dataset"],
        cfg["feature_mode"],
        args.split,
        acc_token_size=int(cfg["acc_token_size"]),
        velocity_bins=int(cfg["velocity_bins"]),
    )
    batch_size = args.batch_size or int(cfg["batch_size"])
    num_workers = args.num_workers if args.num_workers is not None else int(cfg["num_workers"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=num_workers > 0)

    model = build_model(cfg, ds).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[INFO] Checkpoint : {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")
    print(f"[INFO] Upstream   : {upstream_dir}")
    print(f"[INFO] Dataset    : {args.split} split  n={len(ds):,}  {cfg['dataset']} {cfg['feature_mode']}")
    gpu = f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    print(f"[INFO] Device     : {device}{gpu}")

    if args.measure_time:
        sample_loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        sample = move_batch(next(iter(sample_loader)), device)

        def infer_one():
            predict_batch(model, sample, ds, bool(cfg.get("multinomial_sampling", False)))

        lat = measure_latency(infer_one, device, args.warmup, args.iters)
        print_latency(lat, batch_size=1, warmup=args.warmup, iters=args.iters)
        return 0

    labels = None
    if args.scenario:
        labels_path = args.scenario_labels or (data_path / "scenario_labels.csv")
        labels = SampleMetaLookup(data_path, load_scenario_labels(resolve_path(labels_path)))

    acc, extra = run_evaluate(model, loader, ds, device, cfg, labels)
    results = acc.result()
    results.update(extra)
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

