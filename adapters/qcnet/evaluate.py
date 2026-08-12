#!/usr/bin/env python3
"""Evaluate a QCNet adapter checkpoint with NeighFormer-compatible metrics."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
from adapters.qcnet.dataset import NeighFormerQCNetDataset  # noqa: E402
from adapters.qcnet.train import format_path_template, resolve_path  # noqa: E402
from adapters.qcnet.upstream import add_upstream_to_path  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--data-root", type=Path)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--device", type=str)
    p.add_argument("--scenario", action="store_true")
    p.add_argument("--scenario-labels", type=Path)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--measure-time", action="store_true")
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--upstream-dir", type=Path)
    p.add_argument("--output-json", type=Path)
    return p.parse_args(argv)


def load_adapter_checkpoint(path: Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "cfg" not in ckpt or "model_args" not in ckpt:
        raise SystemExit(f"{path} is not an adapter checkpoint produced by adapters/qcnet/train.py")
    return ckpt


def recover_global_trajectories(data, pred: dict[str, torch.Tensor], eval_mask: torch.Tensor, num_historical_steps: int):
    traj = pred["loc_refine_pos"][eval_mask, :, :, :2]
    pi = torch.softmax(pred["pi"][eval_mask], dim=-1)
    origin = data["agent"]["position"][eval_mask, num_historical_steps - 1]
    theta = data["agent"]["heading"][eval_mask, num_historical_steps - 1]
    cos, sin = theta.cos(), theta.sin()
    rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=traj.device)
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = sin
    rot_mat[:, 1, 0] = -sin
    rot_mat[:, 1, 1] = cos
    traj_global = torch.matmul(traj, rot_mat.unsqueeze(1)) + origin[:, :2].reshape(-1, 1, 1, 2)
    return traj_global, pi


@torch.no_grad()
def run_evaluate(model, loader, device, hz: float, dt: float, labels: SampleMetaLookup | None = None):
    model.eval()
    acc = MetricAccumulator(dt=dt, hz=hz)
    th = int(model.num_historical_steps)
    for data in loader:
        data = data.to(device)
        pred = model(data)
        eval_mask = data["agent"]["category"] == 3
        all_modes, pi = recover_global_trajectories(data, pred, eval_mask, th)
        best = pi.argmax(dim=-1)
        chosen = all_modes[torch.arange(all_modes.size(0), device=device), best]
        target = data["agent"]["position"][eval_mask, th:, :2]
        all_modes_metric = all_modes.transpose(1, 2)
        sample_indices = data["sample_index"].detach().cpu().numpy().reshape(-1)
        label_rows = labels.lookup(sample_indices) if labels is not None and labels.enabled else None
        acc.update(chosen, target, all_modes=all_modes_metric, labels=label_rows)
    return acc


def print_device_info(device: torch.device) -> None:
    print("\n====== Device Info ======")
    print(f"  PyTorch version : {torch.__version__}")
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        print(f"  Device          : {props.name}  (index={idx})")
        print(f"  CUDA version    : {torch.version.cuda}")
        print(f"  VRAM            : {props.total_memory / (1024 ** 3):.1f} GB total")
    else:
        print("  Device          : CPU")


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
    try:
        from torch_geometric.loader import DataLoader
    except ImportError as exc:
        raise SystemExit("torch_geometric is required to evaluate QCNet checkpoints.") from exc

    ckpt_path = args.ckpt if args.ckpt.is_absolute() else resolve_path(args.ckpt)
    ckpt = load_adapter_checkpoint(ckpt_path)
    cfg = ckpt["cfg"]
    model_args = ckpt["model_args"]
    upstream_dir = add_upstream_to_path(args.upstream_dir or cfg.get("upstream_dir"))
    try:
        from adapters.qcnet.model import build_qcnet
    except ImportError as exc:
        raise SystemExit(
            "Could not import the official QCNet model. Install torchvision, torch_cluster, "
            "and torch_scatter in this environment."
        ) from exc

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = resolve_path(args.data_root) if args.data_root else resolve_path(cfg["data_root"])
    data_path = dataset_dir(data_root, cfg["dataset"])
    indices = np.load(split_indices_path(data_root, cfg["dataset"], args.split))
    if args.max_samples is not None:
        indices = indices[: args.max_samples]

    ds = NeighFormerQCNetDataset(
        data_path, indices, cfg["dataset"], cfg["feature_mode"], args.split, cfg["lane_half_length"]
    )
    batch_size = args.batch_size or int(cfg["batch_size"])
    num_workers = args.num_workers if args.num_workers is not None else int(cfg["num_workers"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        persistent_workers=num_workers > 0)

    model = build_qcnet(model_args, cfg["feature_mode"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    print(f"[INFO] Checkpoint : {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")
    print(f"[INFO] Upstream   : {upstream_dir}")
    print(f"[INFO] Dataset    : {args.split} split  n={len(ds):,}  {cfg['dataset']} {cfg['feature_mode']}")
    gpu = f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    print(f"[INFO] Device     : {device}{gpu}")

    if args.measure_time:
        sample_loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        sample = next(iter(sample_loader)).to(device)

        def infer_one():
            model(sample)

        print_device_info(device)
        lat = measure_latency(infer_one, device, args.warmup, args.iters)
        print_latency(lat, batch_size=1, warmup=args.warmup, iters=args.iters)
        return 0

    labels = None
    if args.scenario:
        labels_path = args.scenario_labels
        if labels_path is None:
            labels_path = data_path / "scenario_labels.csv"
        labels = SampleMetaLookup(data_path, load_scenario_labels(resolve_path(labels_path)))

    dt = 1.0 / float(cfg.get("eval_hz", 3.0))
    acc = run_evaluate(model, loader, device, float(cfg.get("eval_hz", 3.0)), dt, labels)
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
