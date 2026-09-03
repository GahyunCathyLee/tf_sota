#!/usr/bin/env python3
"""Evaluate a Trajectron++ adapter checkpoint with shared metrics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import dill
import numpy as np
import torch

ADAPTER_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ADAPTER_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from adapters.common import dataset_dir, split_indices_path  # noqa: E402
from adapters.mtp_go.metrics import MetricAccumulator, print_latency, print_metrics  # noqa: E402
from adapters.trajectronpp.preprocess import build_environment, write_environment  # noqa: E402
from adapters.trajectronpp.train import resolve_path, subset_indices, to_plain  # noqa: E402
from adapters.trajectronpp.upstream import add_upstream_to_path  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", help="Optional legacy selector; ignored by this adapter.")
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--data-root", type=Path)
    p.add_argument("--batch-size", type=int, help="Kept for CLI consistency; Trajectron++ eval iterates scenes.")
    p.add_argument("--num-workers", type=int, help="Kept for CLI consistency.")
    p.add_argument("--device")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--measure-time", action="store_true")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--output-json", type=Path)
    return p.parse_args(argv)


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_env_for_split(ckpt: dict[str, Any], args: argparse.Namespace, upstream_dir: Path):
    cfg = ckpt["cfg"]
    data_root = resolve_path(args.data_root) if args.data_root else resolve_path(cfg["data_root"])
    if args.max_samples is None and Path(ckpt.get("processed_dir", "")).exists():
        cached = Path(ckpt["processed_dir"]) / f"{args.split}_env.pkl"
        if cached.exists():
            with cached.open("rb") as f:
                return dill.load(f, encoding="latin1")
    data_path = dataset_dir(data_root, cfg["dataset"])
    indices = np.load(split_indices_path(data_root, cfg["dataset"], args.split))
    indices = subset_indices(indices, args.max_samples)
    env, _ = build_environment(data_path, indices, cfg["dataset"], cfg["feature_mode"], args.split, upstream_dir, dt=float(cfg["dt"]))
    if args.max_samples is None and Path(ckpt.get("processed_dir", "")).exists():
        write_environment(Path(ckpt["processed_dir"]) / f"{args.split}_env.pkl", env)
    return env


def build_trajectron(ckpt: dict[str, Any], env, device: torch.device):
    from model.model_registrar import ModelRegistrar
    from model.trajectron import Trajectron

    with Path(ckpt["upstream_config"]).open("r", encoding="utf-8") as f:
        hyperparams = json.load(f)
    hyperparams["dynamic_edges"] = "yes"
    hyperparams["edge_state_combine_method"] = "sum"
    hyperparams["edge_influence_combine_method"] = "attention"
    hyperparams["edge_addition_filter"] = [0.25, 0.5, 0.75, 1.0]
    hyperparams["edge_removal_filter"] = [1.0, 0.0]
    hyperparams["offline_scene_graph"] = "yes"
    hyperparams["incl_robot_node"] = False
    hyperparams["edge_encoding"] = True
    hyperparams["use_map_encoding"] = False
    hyperparams["augment"] = False
    hyperparams["override_attention_radius"] = []
    hyperparams["node_freq_mult_train"] = False
    hyperparams["node_freq_mult_eval"] = False
    hyperparams["scene_freq_mult_train"] = False
    hyperparams["scene_freq_mult_eval"] = False
    hyperparams["scene_freq_mult_viz"] = False

    registrar = ModelRegistrar(ckpt["upstream_model_dir"], device)
    registrar.load_models(int(ckpt["epoch"]))
    trajectron = Trajectron(registrar, hyperparams, None, device)
    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    registrar.to(device)
    return trajectron, hyperparams


def target_for_scene(scene, ph: int) -> torch.Tensor | None:
    t = scene.timesteps - ph - 1
    for node in scene.nodes:
        if str(node.id).startswith("ego_"):
            return torch.tensor(node.get(np.array([t + 1, t + ph]), {"position": ["x", "y"]}), dtype=torch.float32)
    return None


@torch.no_grad()
def evaluate_env(trajectron, env, hyperparams: dict[str, Any], device: torch.device, hz: float, max_scenes: int | None = None) -> dict[str, Any]:
    ph = int(hyperparams["prediction_horizon"])
    max_hl = int(hyperparams["maximum_history_length"])
    acc = MetricAccumulator(dt=float(env.scenes[0].dt if env.scenes else 0.32), hz=hz)
    preds, targets = [], []
    scenes = env.scenes[:max_scenes] if max_scenes else env.scenes
    for scene in scenes:
        t = np.array([scene.timesteps - ph - 1])
        pred_dict = trajectron.predict(
            scene,
            t,
            ph,
            num_samples=1,
            min_history_timesteps=max_hl,
            min_future_timesteps=ph,
            z_mode=True,
            gmm_mode=True,
            full_dist=False,
        )
        if int(t[0]) not in pred_dict:
            continue
        ego_items = [(node, pred) for node, pred in pred_dict[int(t[0])].items() if str(node.id).startswith("ego_")]
        if not ego_items:
            continue
        pred_np = ego_items[0][1][0, 0]
        target = target_for_scene(scene, ph)
        if target is None:
            continue
        preds.append(torch.tensor(pred_np, dtype=torch.float32))
        targets.append(target)
    if preds:
        acc.update(torch.stack(preds).to(device), torch.stack(targets).to(device))
    return acc.result()


def measure_latency(fn, device: torch.device, warmup: int, iters: int) -> dict[str, float]:
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ckpt_path = args.ckpt if args.ckpt.is_absolute() else resolve_path(args.ckpt)
    ckpt = load_checkpoint(ckpt_path)
    cfg = ckpt["cfg"]
    upstream_dir = add_upstream_to_path(ckpt.get("upstream_dir"))
    requested_device = args.device or cfg.get("device", "auto")
    if requested_device == "auto":
        requested_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    env = load_env_for_split(ckpt, args, upstream_dir)
    trajectron, hyperparams = build_trajectron(ckpt, env, device)
    print(f"[INFO] Checkpoint : {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")
    print(f"[INFO] Upstream   : {upstream_dir}")
    print(f"[INFO] Dataset    : {args.split} split  scenes={len(env.scenes):,}  {cfg['dataset']} {cfg['feature_mode']}")
    print(f"[INFO] Device     : {device}")

    if args.measure_time:
        sample_env = env
        sample_env.scenes = sample_env.scenes[:1]

        def infer_one():
            evaluate_env(trajectron, sample_env, hyperparams, device, float(cfg["eval_hz"]), max_scenes=1)

        lat = measure_latency(infer_one, device, args.warmup, args.iters)
        print_latency(lat, batch_size=1, warmup=args.warmup, iters=args.iters)
        return 0

    results = evaluate_env(trajectron, env, hyperparams, device, float(cfg["eval_hz"]), max_scenes=args.max_samples)
    print(f"\n  n_samples = {int(results.get('n_samples', 0)):,}")
    print_metrics(results)
    if args.output_json:
        out = args.output_json if args.output_json.is_absolute() else resolve_path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(to_plain(results), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
