#!/usr/bin/env python3
"""Evaluate an MTR++ adapter checkpoint with NeighFormer-compatible metrics."""

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

from adapters.common import dataset_dir, split_indices_path  # noqa: E402
from adapters.multiagent_common import multiagent_indices, multiagent_split_dir  # noqa: E402
from adapters.mtrpp.dataset import NeighFormerMTRDataset, build_intention_points_from_data, processed_root  # noqa: E402
from adapters.mtrpp.multiagent_dataset import MultiAgentMTRDataset, build_intention_points_from_multiagent  # noqa: E402
from adapters.mtrpp.mtrpp_model import MTRPPMotionTransformer  # noqa: E402
from adapters.mtrpp.train import (  # noqa: E402
    build_builder_kwargs,
    format_path_template,
    make_loader,
    move_batch_to_device,
    prediction_tensors,
    resolve_path,
    to_attrdict,
)
from adapters.mtrpp.upstream import import_motion_transformer, using_cuda_op_stubs  # noqa: E402
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
    p.add_argument("--measure-time", action="store_true")
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--processed-dir", type=Path)
    p.add_argument("--reuse-processed", action="store_true")
    p.add_argument("--upstream-dir", type=Path)
    p.add_argument("--global-attention-fallback", action="store_true")
    p.add_argument("--output-json", type=Path)
    return p.parse_args(argv)


def measure_latency(fn, device, warmup: int, iters: int) -> dict[str, float]:
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


def run_evaluate(model, loader, device, cfg: dict[str, Any], labels: SampleMetaLookup | None) -> MetricAccumulator:
    import torch

    model.eval()
    acc = MetricAccumulator(dt=float(cfg.get("dt", 1.0 / float(cfg["eval_hz"]))), hz=float(cfg["eval_hz"]))
    scene_acc = SceneMetricAccumulator()
    with torch.no_grad():
        for raw in loader:
            batch = move_batch_to_device(raw, device)
            out = model(batch)
            pred, target, all_modes = prediction_tensors(out)
            if out["pred_scores"].ndim == 3:
                focal_mask = out["input_dict"]["focal_mask"].bool()
                valid_mask = out["input_dict"]["focal_gt_mask"].bool()[focal_mask]
                sample_raw = raw["input_dict"]["sample_index"].detach().cpu()
                sample_indices = torch.repeat_interleave(
                    sample_raw,
                    torch.as_tensor(raw["batch_sample_count"], dtype=torch.long),
                ).numpy().reshape(-1)
            else:
                sample_indices = raw["input_dict"]["sample_index"].detach().cpu().numpy().reshape(-1)
                valid_mask = out["input_dict"]["center_gt_trajs_mask"].bool()
            label_rows = labels.lookup(sample_indices) if labels is not None and labels.enabled else None
            acc.update(pred, target, all_modes=all_modes, valid_mask=valid_mask, labels=label_rows)
            scene_acc.update(all_modes, target, valid_mask, sample_indices)
    return acc, scene_acc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("MTR++ evaluation requires PyTorch.") from exc

    ckpt_path = args.ckpt if args.ckpt.is_absolute() else resolve_path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: dict[str, Any] = ckpt["cfg"]
    architecture = ckpt.get("metadata", {}).get("architecture", cfg.get("architecture", "mtr"))
    cfg["architecture"] = architecture
    model_cfg = to_attrdict(ckpt["model_cfg"])
    if architecture == "mtrpp" and cfg.get("use_importance"):
        raise SystemExit("MTR++ importance integration has not been enabled yet.")
    if cfg.get("use_importance") and cfg.get("feature_mode") != "baseline":
        raise SystemExit("Importance-enabled MTR checkpoints must use feature_mode=baseline.")
    upstream_dir = args.upstream_dir or cfg.get("upstream_dir")
    force_global = bool(args.global_attention_fallback or cfg.get("global_attention_fallback", False))
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = resolve_path(args.data_root) if args.data_root else resolve_path(cfg["data_root"])
    cfg = {**cfg, "data_root": str(data_root)}
    use_multiagent = bool(args.multiagent or cfg.get("multiagent"))
    data_path = multiagent_split_dir(data_root, cfg["dataset"], args.split).parent if use_multiagent else dataset_dir(data_root, cfg["dataset"])
    processed_dir = resolve_path(args.processed_dir) if args.processed_dir else resolve_path(cfg.get("processed_dir", "processed/mtrpp"))
    default_intention_file = processed_root(processed_dir, cfg["dataset"], cfg["feature_mode"]) / "intention_points.pkl"
    intention_file = Path(model_cfg.MOTION_DECODER.INTENTION_POINTS_FILE)
    if not intention_file.is_absolute():
        intention_file = (EXPERIMENT_ROOT / intention_file).resolve()
    if not intention_file.exists():
        print(f"[WARN] intention points missing: {intention_file}; regenerating from train split.", flush=True)
        fallback_file = intention_file
        try:
            if use_multiagent:
                build_intention_points_from_multiagent(
                    multiagent_split_dir(data_root, cfg["dataset"], "train"),
                    fallback_file,
                    num_modes=int(model_cfg.MOTION_DECODER.NUM_MOTION_MODES),
                )
            else:
                train_indices = np.load(split_indices_path(data_root, cfg["dataset"], "train"))
                build_intention_points_from_data(
                    data_path,
                    train_indices,
                    fallback_file,
                    num_modes=int(model_cfg.MOTION_DECODER.NUM_MOTION_MODES),
                )
        except OSError:
            fallback_file = default_intention_file
            if use_multiagent:
                build_intention_points_from_multiagent(
                    multiagent_split_dir(data_root, cfg["dataset"], "train"),
                    fallback_file,
                    num_modes=int(model_cfg.MOTION_DECODER.NUM_MOTION_MODES),
                )
            else:
                train_indices = np.load(split_indices_path(data_root, cfg["dataset"], "train"))
                build_intention_points_from_data(
                    data_path,
                    train_indices,
                    fallback_file,
                    num_modes=int(model_cfg.MOTION_DECODER.NUM_MOTION_MODES),
                )
        model_cfg.MOTION_DECODER.INTENTION_POINTS_FILE = str(fallback_file)
        print(f"[INFO] intention points regenerated -> {fallback_file}", flush=True)

    MotionTransformer, mtr_global_cfg, resolved_upstream = import_motion_transformer(
        upstream_dir,
        global_attention_fallback=force_global,
    )
    if cfg.get("use_importance") and (force_global or using_cuda_op_stubs()):
        raise SystemExit("Importance-enabled MTR requires local attention support.")
    mtr_global_cfg.ROOT_DIR = EXPERIMENT_ROOT
    if use_multiagent:
        split_path = multiagent_split_dir(data_root, cfg["dataset"], args.split)
        indices = multiagent_indices(split_path, args.max_samples)
    else:
        indices = np.load(split_indices_path(data_root, cfg["dataset"], args.split))
        if args.max_samples is not None:
            indices = indices[: args.max_samples]
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if use_multiagent:
        ds = MultiAgentMTRDataset(
            split_path,
            cfg["dataset"],
            args.split,
            indices=indices,
            target_agent_mode=architecture != "mtrpp",
            scene_first_mtrpp=architecture == "mtrpp",
            **build_builder_kwargs(cfg),
        )
    else:
        ds = NeighFormerMTRDataset(
            data_path,
            indices,
            cfg["dataset"],
            cfg["feature_mode"],
            args.split,
            build_builder_kwargs(cfg),
            processed_dir=processed_dir,
            reuse_processed=bool(args.reuse_processed or cfg.get("reuse_processed", False)),
        )
    loader = make_loader(ds, cfg, shuffle=False)
    model_cls = MTRPPMotionTransformer if architecture == "mtrpp" else MotionTransformer
    model = model_cls(config=model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[INFO] Checkpoint : {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")
    print(f"[INFO] Upstream   : {resolved_upstream}")
    print(f"[INFO] Dataset    : {args.split} split  n={len(ds):,}  {cfg['dataset']} {cfg['feature_mode']}")
    print(f"[INFO] Architecture: {architecture}")
    print(f"[INFO] Source     : {'multiagent' if use_multiagent else 'single-agent'}")
    gpu = f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    print(f"[INFO] Device     : {device}{gpu}")

    if args.measure_time:
        sample_loader = make_loader(ds, {**cfg, "batch_size": 1, "num_workers": 0}, shuffle=False)
        sample = move_batch_to_device(next(iter(sample_loader)), device)

        def infer_one():
            prediction_tensors(model(sample))

        lat = measure_latency(infer_one, device, args.warmup, args.iters)
        print_latency(lat, batch_size=1, warmup=args.warmup, iters=args.iters)
        return 0

    labels = None
    if args.scenario:
        labels_path = args.scenario_labels or (data_path / "scenario_labels.csv")
        labels = SampleMetaLookup(data_path, load_scenario_labels(resolve_path(labels_path)))

    acc, scene_acc = run_evaluate(model, loader, device, cfg, labels)
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
