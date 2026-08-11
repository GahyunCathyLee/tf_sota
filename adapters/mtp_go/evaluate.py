#!/usr/bin/env python3
"""evaluate.py — Evaluation entry point for the MTP-GO adapter.

Mirrors `neighformer/evaluate.py`: it loads a checkpoint, rebuilds the model
from the config stored inside it, runs a split, and prints the same ADE/FDE/RMSE
and RMSE@1..5s tables. Metric definitions are identical to NeighFormer's, so the
numbers are directly comparable.

Usage
─────
  python adapters/mtp_go/evaluate.py --ckpt runs/mtp_go/highD/baseline/checkpoints/best.ckpt
  python adapters/mtp_go/evaluate.py --ckpt .../best.ckpt --split val
  python adapters/mtp_go/evaluate.py --ckpt .../best.ckpt --scenario
  python adapters/mtp_go/evaluate.py --ckpt .../best.ckpt --measure-time
  python adapters/mtp_go/evaluate.py --ckpt .../best.ckpt --data-root ./data   # Colab
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ADAPTER_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ADAPTER_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from adapters.common import dataset_dir, feature_mode_indices, split_indices_path  # noqa: E402
from adapters.mtp_go.dataset import NeighFormerGraphDataset  # noqa: E402
from adapters.mtp_go.lit_module import evaluate as run_evaluate  # noqa: E402
from adapters.mtp_go.lit_module import make_lit_module_class  # noqa: E402
from adapters.mtp_go.metrics import (  # noqa: E402
    SampleMetaLookup,
    load_scenario_labels,
    print_latency,
    print_metrics,
    print_scenario_results,
)
from adapters.mtp_go.train import resolve_path, to_plain  # noqa: E402
from adapters.mtp_go.upstream import (  # noqa: E402
    add_upstream_to_path,
    build_motion_model,
    resolve_upstream_dir,
)

LATENCY_WARMUP = 1_000
LATENCY_ITERS = 10_000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an MTP-GO adapter checkpoint.")
    p.add_argument("--ckpt", required=True, type=Path, help="Path to a .ckpt file")
    p.add_argument("--split", default="test", choices=["train", "val", "test"],
                   help="Dataset split to evaluate on (default: test)")
    p.add_argument("--data-root", type=Path,
                   help="Override the data root stored in the checkpoint (e.g. ./data on Colab)")
    p.add_argument("--scenario", action="store_true",
                   help="Per-scenario (event / state) breakdown from scenario_labels.csv")
    p.add_argument("--scenario-labels", type=Path,
                   help="Override the scenario_labels.csv path")
    p.add_argument("--batch-size", type=int, help="Override the checkpoint's batch size")
    p.add_argument("--num-workers", type=int, help="Override the checkpoint's num_workers")
    p.add_argument("--device", type=str, help="cuda | cpu (default: cuda if available)")
    p.add_argument("--max-samples", type=int, help="Evaluate only the first N samples")
    p.add_argument("--measure-time", action="store_true",
                   help=f"Measure single-sample inference latency "
                        f"({LATENCY_WARMUP:,} warmup + {LATENCY_ITERS:,} iters)")
    p.add_argument("--warmup", type=int, default=LATENCY_WARMUP)
    p.add_argument("--iters", type=int, default=LATENCY_ITERS)
    p.add_argument("--upstream-dir", type=Path, help="Override the MTP-GO checkout location")
    p.add_argument("--output-json", type=Path, help="Write the metrics to this JSON file")
    return p.parse_args(argv)


def print_device_info(device) -> None:
    import torch

    print("\n====== Device Info ======")
    print(f"  PyTorch version : {torch.__version__}")
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024 ** 3)
        alloc_gb = torch.cuda.memory_allocated(idx) / (1024 ** 3)
        print(f"  Device          : {props.name}  (index={idx})")
        print(f"  CUDA version    : {torch.version.cuda}")
        print(f"  SM count        : {props.multi_processor_count}")
        print(f"  VRAM            : {total_gb:.1f} GB total  /  {total_gb - alloc_gb:.1f} GB free")
    else:
        import platform

        print(f"  Device          : CPU  ({platform.processor() or platform.machine()})")


def measure_latency(fn, device, warmup: int, iters: int) -> dict[str, float]:
    import torch

    print(f"  Warm-up      : {warmup:,} iters ...", end=" ", flush=True)
    with torch.no_grad():
        for _ in range(warmup):
            fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("done")

    times_ms: list[float] = []
    print(f"  Measurement  : {iters:,} iters ...", end=" ", flush=True)
    with torch.no_grad():
        if device.type == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            for _ in range(iters):
                starter.record()
                fn()
                ender.record()
                torch.cuda.synchronize()
                times_ms.append(starter.elapsed_time(ender))
        else:
            for _ in range(iters):
                t0 = time.perf_counter()
                fn()
                times_ms.append((time.perf_counter() - t0) * 1000.0)
    print("done")

    arr = np.asarray(times_ms, dtype=np.float64)
    return {"avg_ms": float(arr.mean()), "min_ms": float(arr.min()), "max_ms": float(arr.max())}


def load_checkpoint_config(ckpt_path: Path) -> tuple[dict, SimpleNamespace]:
    """Read a checkpoint written by adapters/mtp_go/train.py."""
    import torch

    # Our own checkpoint: it stores the config namespace, so weights_only is off.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {}).get("args")
    if hp is None:
        raise SystemExit(
            f"{ckpt_path} has no stored adapter config (hyper_parameters['args']). "
            "It was probably not produced by adapters/mtp_go/train.py."
        )
    if not hasattr(hp, "feature_mode"):
        raise SystemExit(
            f"{ckpt_path} was written by an older version of the adapter that did not "
            "store dataset_name/feature_mode. Retrain, or evaluate via train.py."
        )
    return ckpt, hp


def build_model_from_config(hp: SimpleNamespace, n_features: int, history_len: int, dt: float):
    from base_mdn import LitEncoderDecoder  # upstream
    from models.gru_gnn import GRUGNNDecoder, GRUGNNEncoder  # upstream

    static_f_dim = 2 * int(bool(hp.n_ode_static))
    motion_model = build_motion_model(hp, dt, static_f_dim)
    encoder = GRUGNNEncoder(
        input_size=n_features,
        hidden_size=hp.hidden_size,
        n_mixtures=motion_model.mixtures,
        n_layers=hp.n_gnn_layers,
        gnn_layer=hp.gnn_layer,
        n_heads=hp.n_heads,
        static_f_dim=static_f_dim,
        init_static=hp.init_static,
        use_edge_features=hp.use_edge_features,
    )
    decoder = GRUGNNDecoder(
        motion_model,
        hidden_size=encoder.hidden_size,
        max_length=history_len + 1,
        n_layers=hp.n_gnn_layers,
        n_heads=hp.n_heads,
        static_f_dim=static_f_dim,
        gnn_layer=hp.gnn_layer,
        init_static=hp.init_static,
    )
    return make_lit_module_class(LitEncoderDecoder)(encoder, decoder, hp), motion_model


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch
    from torch_geometric.loader import DataLoader

    ckpt_path = args.ckpt if args.ckpt.is_absolute() else resolve_path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    upstream_dir = resolve_upstream_dir(args.upstream_dir)
    add_upstream_to_path(upstream_dir)

    ckpt, hp = load_checkpoint_config(ckpt_path)
    print(f"[INFO] Checkpoint : {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")

    dataset_name = hp.dataset_name
    feature_mode = hp.feature_mode
    dt = float(hp.dt)
    hz = float(getattr(hp, "eval_hz", 3.0))

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    gpu = f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    print(f"[INFO] Device     : {device}{gpu}")

    # ── Data ──────────────────────────────────────────────────────────────────
    data_root = resolve_path(args.data_root) if args.data_root else resolve_path(hp.data_root)
    data_dir = dataset_dir(data_root, dataset_name)
    if not data_dir.exists():
        raise SystemExit(
            f"Data directory not found: {data_dir}\n"
            f"Pass --data-root pointing at the directory that holds "
            f"{dataset_name}/dimI and {dataset_name}/splits."
        )
    split_file = split_indices_path(data_root, dataset_name, args.split)
    if not split_file.exists():
        raise SystemExit(f"Split index file not found: {split_file}")
    split_idx = np.load(split_file).astype(np.int64)
    if args.max_samples:
        split_idx = split_idx[: args.max_samples]

    ds = NeighFormerGraphDataset(data_dir, split_idx, feature_mode, split=args.split)
    n_features = len(feature_mode_indices(feature_mode))
    print(f"[INFO] Dataset    : {dataset_name}/{feature_mode}  {args.split} split  "
          f"n={ds.len():,}  node_channels={n_features}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model, motion_model = build_model_from_config(hp, n_features, ds.history_len, dt)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model      : {hp.motion_model}/{hp.gnn_layer}  params={n_params:,}  "
          f"n_states={motion_model.n_states}  mixtures={motion_model.mixtures}")

    # ── Latency mode ──────────────────────────────────────────────────────────
    if args.measure_time:
        from torch_geometric.data import Batch

        print_device_info(device)
        sample = Batch.from_data_list([ds[0]]).to(device)

        def _infer():
            model._ego_prediction(sample)

        print("\n====== Inference Latency ======")
        lat = measure_latency(_infer, device, warmup=args.warmup, iters=args.iters)
        print_latency(lat, batch_size=1, warmup=args.warmup, iters=args.iters)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(to_plain(lat), indent=2), encoding="utf-8")
        print()
        return 0

    # ── Metric evaluation ─────────────────────────────────────────────────────
    batch_size = args.batch_size if args.batch_size is not None else int(hp.batch_size)
    num_workers = args.num_workers if args.num_workers is not None else int(hp.n_workers)
    loader_kwargs: dict[str, Any] = dict(num_workers=num_workers, pin_memory=device.type == "cuda")
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    meta_lookup = None
    if args.scenario:
        labels_path = (
            args.scenario_labels
            if args.scenario_labels
            else (resolve_path(hp.scenario_labels) if str(getattr(hp, "scenario_labels", ""))
                  else data_dir / "scenario_labels.csv")
        )
        meta_lookup = SampleMetaLookup(data_dir, load_scenario_labels(Path(labels_path)))

    print(f"\n====== Evaluation  [{args.split}] ======")
    results = run_evaluate(model, loader, device, dt, hz=hz, meta_lookup=meta_lookup, progress=True)

    print(f"\n  n_samples = {results['n_samples']:,}")
    print_metrics(results)

    ev = results.pop("_event_stats", None)
    st = results.pop("_state_stats", None)
    if ev:
        print_scenario_results(ev, label_type="Event")
    if st:
        print_scenario_results(st, label_type="State")

    if args.output_json:
        payload = {
            "checkpoint": str(ckpt_path),
            "dataset": dataset_name,
            "feature_mode": feature_mode,
            "split": args.split,
            "metrics": results,
        }
        if ev:
            payload["event_stats"] = ev
        if st:
            payload["state_stats"] = st
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(to_plain(payload), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n[INFO] Metrics written to {args.output_json}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
