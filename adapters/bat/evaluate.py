#!/usr/bin/env python3
"""Evaluate a BAT adapter checkpoint with NeighFormer-compatible metrics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ADAPTER_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ADAPTER_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from adapters.bat.dataset import polar_to_cart  # noqa: E402
from adapters.bat.train import build_dataset, make_loader, move_batch, prediction_cart, resolve_path  # noqa: E402
from adapters.bat.upstream import import_bat_model  # noqa: E402


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
    p.add_argument("--processed-dir", type=Path)
    p.add_argument("--reuse-processed", action="store_true")
    p.add_argument("--upstream-dir", type=Path)
    p.add_argument("--output-json", type=Path)
    p.add_argument("--single-mode", action="store_true", help="Use BAT's train_flag=True single trajectory path.")
    return p.parse_args(argv)


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("BAT evaluation requires PyTorch. Install torch before running evaluation.") from exc
    return torch


def forward_bat(gd_encoder: Any, generator: Any, batch: dict[str, Any]) -> tuple[Any, Any, Any]:
    values = gd_encoder(
        batch["hist"],
        batch["nbrs"],
        batch["hist_relative"],
        batch["mask"],
        batch["va"],
        batch["nbrsva"],
        batch["lane"],
        batch["nbrslane"],
        batch["cls"],
        batch["nbrscls"],
        batch["nbrs_ref_self"],
        batch["nbrs_ref_nbrs"],
        batch["feature_matrix"],
        batch["behavior"],
    )
    return generator(values, batch["lat_enc"], batch["lon_enc"])


def select_prediction(fut_pred: Any, lat_pred: Any, lon_pred: Any, polar: bool) -> tuple[Any, Any | None]:
    import torch

    if not isinstance(fut_pred, list):
        return prediction_cart(fut_pred, polar), None
    modes = torch.stack([polar_to_cart(p[:, :, 0:2].permute(1, 0, 2)) if polar else p[:, :, 0:2].permute(1, 0, 2) for p in fut_pred], dim=2)
    probs = torch.stack([lon_pred[:, k] * lat_pred[:, l] for k in range(lon_pred.shape[1]) for l in range(lat_pred.shape[1])], dim=1)
    best = probs.argmax(dim=1)
    gather_idx = best.view(-1, 1, 1, 1).expand(-1, modes.shape[1], 1, 2)
    pred = modes.gather(2, gather_idx).squeeze(2)
    return pred, modes


def measure_latency(fn, device: Any, warmup: int, iters: int) -> dict[str, float]:
    import torch

    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times, dtype=np.float64)
    return {"avg_ms": float(arr.mean()), "min_ms": float(arr.min()), "max_ms": float(arr.max())}


def run_evaluate(gd_encoder: Any, generator: Any, loader: Any, device: Any, cfg: dict[str, Any], ds: Any, labels: Any) -> Any:
    import torch
    from adapters.mtp_go.metrics import MetricAccumulator

    gd_encoder.eval()
    generator.eval()
    acc = MetricAccumulator(dt=float(cfg.get("dt", 1.0 / float(cfg["eval_hz"]))), hz=float(cfg["eval_hz"]))
    with torch.no_grad():
        for raw in loader:
            batch = move_batch(raw, device)
            fut_pred, lat_pred, lon_pred = forward_bat(gd_encoder, generator, batch)
            pred, all_modes = select_prediction(fut_pred, lat_pred, lon_pred, ds.polar)
            sample_indices = raw["sample_index"].detach().cpu().numpy()
            label_rows = labels.lookup(sample_indices) if labels is not None and labels.enabled else None
            acc.update(pred, batch["target"], all_modes=all_modes, labels=label_rows)
    return acc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch = require_torch()
    from adapters.mtp_go.metrics import (
        SampleMetaLookup,
        load_scenario_labels,
        print_latency,
        print_metrics,
        print_scenario_results,
    )

    ckpt_path = args.ckpt if args.ckpt.is_absolute() else resolve_path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: dict[str, Any] = dict(ckpt["cfg"])
    if args.data_root:
        cfg["data_root"] = str(resolve_path(args.data_root))
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.upstream_dir:
        cfg["upstream_dir"] = str(args.upstream_dir)
    if args.max_samples is not None:
        cfg["max_eval_samples"] = args.max_samples

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = build_dataset(cfg, args.split, cfg.get("max_eval_samples"))
    loader = make_loader(ds, cfg, shuffle=False)
    GDEncoder, Generator, upstream_dir = import_bat_model(cfg.get("upstream_dir"))
    model_args = dict(ckpt["model_args"])
    model_args["device"] = device
    model_args["train_flag"] = bool(args.single_mode)
    gd_encoder = GDEncoder(model_args).to(device)
    generator = Generator(model_args).to(device)
    gd_encoder.load_state_dict(ckpt["gd_encoder"])
    generator.load_state_dict(ckpt["generator"])

    print(f"[INFO] Checkpoint : {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")
    print(f"[INFO] Upstream   : {upstream_dir}")
    print(f"[INFO] Dataset    : {args.split} split  n={len(ds):,}  {cfg['dataset']} {cfg['feature_mode']}")
    gpu = f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    print(f"[INFO] Device     : {device}{gpu}")

    if args.measure_time:
        sample_loader = make_loader(ds, {**cfg, "batch_size": 1, "num_workers": 0}, shuffle=False)
        sample = move_batch(next(iter(sample_loader)), device)

        def infer_one():
            fut_pred, lat_pred, lon_pred = forward_bat(gd_encoder, generator, sample)
            select_prediction(fut_pred, lat_pred, lon_pred, ds.polar)

        lat = measure_latency(infer_one, device, args.warmup, args.iters)
        print_latency(lat, batch_size=1, warmup=args.warmup, iters=args.iters)
        return 0

    labels = None
    if args.scenario:
        labels_path = args.scenario_labels or (ds.data_dir / "scenario_labels.csv")
        labels = SampleMetaLookup(ds.data_dir, load_scenario_labels(resolve_path(labels_path)))
        labels.warn_if_incomplete(ds.sample_indices, context=args.split)

    acc = run_evaluate(gd_encoder, generator, loader, device, cfg, ds, labels)
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
