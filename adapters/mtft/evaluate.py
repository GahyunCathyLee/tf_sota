#!/usr/bin/env python3
"""Evaluate an MTFT checkpoint with NeighFormer-compatible metrics."""

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
    print_metrics,
    print_scenario_results,
)
from adapters.mtft.dataset import MTFTDataset, collate_fn, dataset_summary  # noqa: E402
from adapters.mtft.model import MTFT  # noqa: E402
from adapters.mtft.train import resolve_path  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", "--checkpoint", dest="ckpt", required=True, type=Path)
    p.add_argument("--config", type=Path, help="Accepted for CLI symmetry; checkpoint config is authoritative.")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--data-root", type=Path)
    p.add_argument("--split-root", type=Path)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--device", type=str)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--scenario", action="store_true")
    p.add_argument("--scenario-labels", type=Path)
    p.add_argument("--measure-time", action="store_true")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--output-json", type=Path)
    return p.parse_args(argv)


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if "cfg" not in ckpt or "state_dict" not in ckpt:
        raise SystemExit(f"{path} is not an MTFT adapter checkpoint")
    return ckpt


def make_eval_dataset(cfg: dict[str, Any], split: str, args: argparse.Namespace) -> MTFTDataset:
    data_root = resolve_path(args.data_root) if args.data_root else resolve_path(cfg["data_root"])
    split_root = resolve_path(args.split_root) if args.split_root else resolve_path(cfg.get("split_root") or cfg["data_root"])
    indices = np.load(split_indices_path(split_root, cfg["dataset"], split))
    if args.max_samples is not None:
        indices = indices[: int(args.max_samples)]
    use_i = bool(cfg.get("model_hparams", {}).get("use_I", cfg.get("feature_mode") == "I"))
    missing = cfg.get("missing", {})
    return MTFTDataset(
        dataset_dir(data_root, cfg["dataset"]),
        indices=indices,
        use_i=use_i,
        missing_enabled=bool(missing.get("enabled", False)),
        missing_min_ratio=float(missing.get("min_ratio", 0.0)),
        missing_max_ratio=float(missing.get("max_ratio", 0.0)),
        seed=int(cfg.get("seed", 42)),
        return_meta=True,
    )


def make_loader(dataset: MTFTDataset, batch_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_fn,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)


@torch.no_grad()
def run_evaluate(
    model: MTFT,
    loader: DataLoader,
    device: torch.device,
    cfg: dict[str, Any],
    labels: SampleMetaLookup | None = None,
) -> tuple[dict[str, Any], MetricAccumulator]:
    model.eval()
    acc = MetricAccumulator(dt=float(cfg.get("dt", 0.32)), hz=float(cfg.get("eval_hz", 3.0)))
    mse_sum = 0.0
    n_values = 0
    for batch in loader:
        agents = batch["agents"].to(device, non_blocking=True)
        obs_mask = batch["obs_mask"].to(device, non_blocking=True)
        agent_mask = batch["agent_mask"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        pred = model(agents, obs_mask, agent_mask).float()
        target = target.float()
        mse_sum += float(torch.nn.functional.mse_loss(pred, target, reduction="sum"))
        n_values += int(target.numel())
        label_rows = None
        if labels is not None and labels.enabled:
            sample_indices = batch["sample_index"].detach().cpu().numpy().reshape(-1)
            label_rows = labels.lookup(sample_indices)
        acc.update(pred, target, labels=label_rows)
    metrics = acc.result()
    metrics["loss_mse"] = mse_sum / max(1, n_values)
    metrics["loss_rmse"] = float(np.sqrt(metrics["loss_mse"]))
    return metrics, acc


@torch.no_grad()
def measure_latency(model: MTFT, loader: DataLoader, device: torch.device, warmup: int, iters: int) -> dict[str, float]:
    batch = next(iter(loader))
    agents = batch["agents"].to(device)
    obs_mask = batch["obs_mask"].to(device)
    agent_mask = batch["agent_mask"].to(device)

    def infer() -> None:
        model(agents, obs_mask, agent_mask)

    for _ in range(warmup):
        infer()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        infer()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return {"avg_ms": float(arr.mean()), "min_ms": float(arr.min()), "max_ms": float(arr.max())}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ckpt_path = args.ckpt if args.ckpt.is_absolute() else resolve_path(args.ckpt)
    ckpt = load_checkpoint(ckpt_path)
    cfg = ckpt["cfg"]
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = make_eval_dataset(cfg, args.split, args)
    print("====== MTFT Eval Data ======")
    print(json.dumps(dataset_summary(ds), indent=2))
    batch_size = args.batch_size or int(cfg.get("eval_batch_size", cfg.get("batch_size", 256)))
    num_workers = args.num_workers if args.num_workers is not None else int(cfg.get("num_workers", 4))
    loader = make_loader(ds, batch_size, num_workers, pin_memory=device.type == "cuda")

    model = MTFT(**ckpt["model_args"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    labels = None
    if args.scenario:
        data_root = resolve_path(args.data_root) if args.data_root else resolve_path(cfg["data_root"])
        data_path = dataset_dir(data_root, cfg["dataset"])
        labels_path = args.scenario_labels or (data_path / "scenario_labels.csv")
        labels = SampleMetaLookup(data_path, load_scenario_labels(resolve_path(labels_path)))
        if labels.enabled:
            labels.warn_if_incomplete(ds.indices, context=f"{cfg['dataset']} {args.split}")
    metrics, acc = run_evaluate(model, loader, device, cfg, labels=labels)
    print(f"\n====== MTFT {cfg['dataset']} {cfg['feature_mode']} {args.split} ======")
    print_metrics(metrics)
    if labels is not None and acc.has_scenario:
        print_scenario_results(acc.event_stats, "Event")
        print_scenario_results(acc.state_stats, "State")
    if args.measure_time:
        lat = measure_latency(model, loader, device, args.warmup, args.iters)
        print(f"[Latency] avg={lat['avg_ms']:.3f}ms min={lat['min_ms']:.3f}ms max={lat['max_ms']:.3f}ms batch={batch_size}")
        metrics["latency"] = lat
    if args.output_json:
        out = args.output_json if args.output_json.is_absolute() else resolve_path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
