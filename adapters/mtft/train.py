#!/usr/bin/env python3
"""Train MTFT on NeighFormer highD/exiD dimI arrays."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

ADAPTER_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ADAPTER_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from adapters.common import dataset_dir, split_indices_path  # noqa: E402
from adapters.mtp_go.metrics import MetricAccumulator, print_metrics  # noqa: E402
from adapters.mtft.dataset import MTFTDataset, collate_fn, dataset_summary  # noqa: E402
from adapters.mtft.model import MTFT  # noqa: E402


DEFAULTS: dict[str, Any] = {
    "adapter": "mtft",
    "dataset": "",
    "feature_mode": "baseline",
    "exp_tag": "",
    "data_root": "/home/gahyun/neighformer/data",
    "split_root": "/home/gahyun/neighformer/data",
    "eval_hz": 3.0,
    "dt": 0.32,
    "batch_size": 128,
    "eval_batch_size": 256,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "seed": 42,
    "device": "cuda",
    "epochs": 100,
    "lr": 1.0e-4,
    "weight_decay": 0.0,
    "amp": True,
    "grad_clip": 1.0,
    "ckpt_dir": "ckpts/mtft",
    "output_dir": "runs/mtft/{dataset}/{feature_mode}/{exp_tag}",
    "max_train_samples": None,
    "max_eval_samples": None,
    "model_hparams": {},
    "missing": {"enabled": False, "min_ratio": 0.0, "max_ratio": 0.0},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--dataset", choices=["highD", "exiD"])
    p.add_argument("--data-root", type=Path)
    p.add_argument("--split-root", type=Path)
    p.add_argument("--ckpt-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--exp-tag", type=str)
    p.add_argument("--use-I", action="store_true")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--eval-batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", type=str)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--grad-clip", type=float)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction)
    p.add_argument("--hidden-dim", type=int)
    p.add_argument("--num-layers", type=int)
    p.add_argument("--num-heads", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--max-train-samples", type=int)
    p.add_argument("--max-eval-samples", type=int)
    p.add_argument("--check-data", action="store_true")
    p.add_argument("--forward-smoke", action="store_true")
    p.add_argument("--tiny-overfit", action="store_true")
    p.add_argument("--overfit-steps", type=int, default=80)
    return p.parse_args(argv)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_raw_config(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise SystemExit("Circular config base chain: " + " -> ".join(str(p) for p in (*seen, path)))
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base_ref = raw.pop("base", None)
    if not base_ref:
        return raw
    candidates = [Path(base_ref)] if Path(base_ref).is_absolute() else [path.parent / base_ref, EXPERIMENT_ROOT / base_ref]
    for cand in candidates:
        if cand.exists():
            return _deep_merge(load_raw_config(cand, (*seen, path)), raw)
    raise SystemExit(f"{path}: base config '{base_ref}' not found")


def load_config(path: Path) -> dict[str, Any]:
    raw = load_raw_config(path)
    cfg = dict(DEFAULTS)
    for key in ("adapter", "dataset", "feature_mode", "exp_tag"):
        if key in raw:
            cfg[key] = raw[key]
    data = raw.get("data", {})
    if isinstance(data, dict):
        for key, value in data.items():
            cfg[{"root": "data_root", "hz": "eval_hz"}.get(key, key)] = value
    training = raw.get("training", {})
    if isinstance(training, dict):
        for key, value in training.items():
            cfg[key] = value
    evaluation = raw.get("evaluation", {})
    if isinstance(evaluation, dict):
        if "batch_size" in evaluation:
            cfg["eval_batch_size"] = evaluation["batch_size"]
        for key in ("hz", "dt"):
            if key in evaluation:
                cfg[{"hz": "eval_hz"}.get(key, key)] = evaluation[key]
    paths = raw.get("paths", {})
    if isinstance(paths, dict):
        for key, value in paths.items():
            cfg[key] = value
    cfg["model_hparams"] = raw.get("model", {})
    cfg["missing"] = _deep_merge(DEFAULTS["missing"], raw.get("missing", {}) or {})
    if not cfg["dataset"]:
        raise SystemExit("Config must set dataset: highD or exiD")
    if not cfg["exp_tag"]:
        cfg["exp_tag"] = f"{cfg['dataset']}_{cfg['feature_mode']}"
    return cfg


def apply_cli(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    for arg_name, cfg_name in (
        ("dataset", "dataset"),
        ("data_root", "data_root"),
        ("split_root", "split_root"),
        ("ckpt_dir", "ckpt_dir"),
        ("output_dir", "output_dir"),
        ("exp_tag", "exp_tag"),
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("eval_batch_size", "eval_batch_size"),
        ("num_workers", "num_workers"),
        ("seed", "seed"),
        ("device", "device"),
        ("lr", "lr"),
        ("weight_decay", "weight_decay"),
        ("grad_clip", "grad_clip"),
        ("amp", "amp"),
        ("max_train_samples", "max_train_samples"),
        ("max_eval_samples", "max_eval_samples"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            cfg[cfg_name] = str(value) if isinstance(value, Path) else value
    for arg_name, model_name in (
        ("hidden_dim", "hidden_dim"),
        ("num_layers", "num_layers"),
        ("num_heads", "num_heads"),
        ("dropout", "dropout"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            cfg.setdefault("model_hparams", {})[model_name] = value
    if args.use_I:
        cfg["feature_mode"] = "I"
        cfg.setdefault("model_hparams", {})["use_I"] = True
    return cfg


def resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (EXPERIMENT_ROOT / p).resolve()


def format_path(value: str | Path, cfg: dict[str, Any]) -> Path:
    return resolve_path(str(value).format(
        dataset=cfg["dataset"],
        feature_mode=cfg["feature_mode"],
        exp_tag=cfg["exp_tag"],
        data_root=cfg["data_root"],
    ))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_datasets(cfg: dict[str, Any]) -> tuple[MTFTDataset, MTFTDataset]:
    data_root = resolve_path(cfg["data_root"])
    split_root = resolve_path(cfg.get("split_root") or cfg["data_root"])
    data_path = dataset_dir(data_root, cfg["dataset"])
    train_idx = np.load(split_indices_path(split_root, cfg["dataset"], "train"))
    val_idx = np.load(split_indices_path(split_root, cfg["dataset"], "val"))
    use_i = bool(cfg["model_hparams"].get("use_I", cfg["feature_mode"] == "I"))
    missing = cfg.get("missing", {})
    ds_kwargs = dict(
        data_dir=data_path,
        use_i=use_i,
        missing_enabled=bool(missing.get("enabled", False)),
        missing_min_ratio=float(missing.get("min_ratio", 0.0)),
        missing_max_ratio=float(missing.get("max_ratio", 0.0)),
        seed=int(cfg["seed"]),
    )
    train_ds = MTFTDataset(indices=train_idx, max_samples=cfg.get("max_train_samples"), **ds_kwargs)
    val_ds = MTFTDataset(indices=val_idx, max_samples=cfg.get("max_eval_samples"), return_meta=True, **ds_kwargs)
    return train_ds, val_ds


def make_loader(dataset: MTFTDataset, batch_size: int, num_workers: int, shuffle: bool, pin_memory: bool, persistent: bool) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": shuffle,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "collate_fn": collate_fn,
        "persistent_workers": bool(persistent) and int(num_workers) > 0,
    }
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)


def build_model(cfg: dict[str, Any], dataset: MTFTDataset) -> MTFT:
    mh = dict(cfg.get("model_hparams", {}))
    use_i = bool(mh.get("use_I", cfg["feature_mode"] == "I"))
    mh.setdefault("input_dim", 3 if use_i else 2)
    mh.setdefault("future_steps", dataset.shape.future_steps)
    mh.setdefault("hidden_dim", 128)
    mh.setdefault("num_layers", 4)
    mh.setdefault("num_heads", 5)
    mh.setdefault("dropout", 0.15)
    return MTFT(
        input_dim=int(mh["input_dim"]),
        hidden_dim=int(mh["hidden_dim"]),
        num_layers=int(mh["num_layers"]),
        num_heads=int(mh["num_heads"]),
        future_steps=int(mh["future_steps"]),
        dropout=float(mh["dropout"]),
    )


@torch.no_grad()
def evaluate_model(model: MTFT, loader: DataLoader, device: torch.device, cfg: dict[str, Any]) -> dict[str, Any]:
    model.eval()
    acc = MetricAccumulator(dt=float(cfg["dt"]), hz=float(cfg["eval_hz"]))
    total_loss = 0.0
    total_n = 0
    loss_fn = torch.nn.MSELoss(reduction="sum")
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    for batch in tqdm(loader, desc="Val", leave=False, dynamic_ncols=True):
        agents = batch["agents"].to(device, non_blocking=True)
        obs_mask = batch["obs_mask"].to(device, non_blocking=True)
        agent_mask = batch["agent_mask"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
            pred = model(agents, obs_mask, agent_mask)
        total_loss += float(loss_fn(pred.float(), target.float()))
        total_n += int(target.numel())
        acc.update(pred.float(), target.float())
    out = acc.result()
    out["loss_mse"] = total_loss / max(1, total_n)
    out["loss_rmse"] = float(np.sqrt(out["loss_mse"]))
    return out


def train_one_epoch(
    model: MTFT,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    epoch: int,
) -> dict[str, float]:
    model.train()
    loss_fn = torch.nn.MSELoss()
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    total = 0.0
    count = 0
    pbar = tqdm(loader, desc=f"Train {epoch}", leave=False, dynamic_ncols=True)
    for batch in pbar:
        agents = batch["agents"].to(device, non_blocking=True)
        obs_mask = batch["obs_mask"].to(device, non_blocking=True)
        agent_mask = batch["agent_mask"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
            pred = model(agents, obs_mask, agent_mask)
            loss = loss_fn(pred.float(), target.float())
        loss.backward()
        grad_clip = float(cfg.get("grad_clip") or 0.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.detach())
        count += 1
        pbar.set_postfix(mse=f"{float(loss.detach()):.4f}", rmse=f"{math_sqrt(float(loss.detach())):.3f}")
    mse = total / max(1, count)
    return {"loss_mse": mse, "loss_rmse": math_sqrt(mse)}


def math_sqrt(value: float) -> float:
    return float(np.sqrt(max(0.0, value)))


def run_forward_smoke(model: MTFT, loader: DataLoader, device: torch.device) -> None:
    model.eval()
    batch = next(iter(loader))
    agents = batch["agents"].to(device)
    obs_mask = batch["obs_mask"].to(device)
    agent_mask = batch["agent_mask"].to(device)
    with torch.no_grad():
        pred = model(agents, obs_mask, agent_mask)
    if not torch.isfinite(pred).all():
        raise RuntimeError("forward smoke produced NaN/Inf")
    print(f"[Forward smoke] agents={tuple(agents.shape)} pred={tuple(pred.shape)} finite=true")


def run_tiny_overfit(cfg: dict[str, Any], device: torch.device) -> None:
    tiny_cfg = dict(cfg)
    tiny_cfg["max_train_samples"] = min(int(cfg.get("max_train_samples") or 128), 128)
    tiny_cfg["max_eval_samples"] = min(int(cfg.get("max_eval_samples") or 128), 128)
    train_ds, _ = make_datasets(tiny_cfg)
    loader = make_loader(train_ds, min(32, int(cfg["batch_size"])), 0, True, False, False)
    model = build_model(cfg, train_ds).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    loss_fn = torch.nn.MSELoss()
    first_loss = None
    last_loss = None
    steps = int(cfg.get("overfit_steps", 80))
    it = iter(loader)
    model.train()
    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        agents = batch["agents"].to(device)
        obs_mask = batch["obs_mask"].to(device)
        agent_mask = batch["agent_mask"].to(device)
        target = batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(agents, obs_mask, agent_mask)
        loss = loss_fn(pred, target)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip") or 1.0))
        optimizer.step()
        value = float(loss.detach())
        first_loss = value if first_loss is None else first_loss
        last_loss = value
    print(f"[Tiny overfit] samples={len(train_ds)} steps={steps} first_mse={first_loss:.4f} last_mse={last_loss:.4f} grad_norm={float(grad_norm):.4f}")


def save_checkpoint(path: Path, model: MTFT, cfg: dict[str, Any], metrics: dict[str, Any], epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_args = {
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "num_heads": model.num_heads,
        "future_steps": model.future_steps,
        "dropout": float(cfg.get("model_hparams", {}).get("dropout", 0.15)),
    }
    torch.save(
        {
            "cfg": cfg,
            "model_args": model_args,
            "state_dict": model.state_dict(),
            "metrics": metrics,
            "epoch": epoch,
        },
        path,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = apply_cli(load_config(args.config), args)
    cfg["overfit_steps"] = args.overfit_steps
    set_seed(int(cfg["seed"]))
    device = torch.device(cfg["device"] if torch.cuda.is_available() or str(cfg["device"]) == "cpu" else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_ds, val_ds = make_datasets(cfg)
    print("====== MTFT Data ======")
    print(json.dumps({"train": dataset_summary(train_ds), "val": dataset_summary(val_ds)}, indent=2))
    if args.check_data and not args.forward_smoke and not args.tiny_overfit and int(cfg["epochs"]) <= 0:
        return 0

    train_loader = make_loader(
        train_ds,
        int(cfg["batch_size"]),
        int(cfg["num_workers"]),
        True,
        bool(cfg["pin_memory"]) and device.type == "cuda",
        bool(cfg["persistent_workers"]),
    )
    val_loader = make_loader(
        val_ds,
        int(cfg["eval_batch_size"]),
        int(cfg["num_workers"]),
        False,
        bool(cfg["pin_memory"]) and device.type == "cuda",
        bool(cfg["persistent_workers"]),
    )

    model = build_model(cfg, train_ds).to(device)
    print(f"[Model] params={sum(p.numel() for p in model.parameters()):,} input_dim={model.input_dim} hidden={model.hidden_dim}")
    if args.forward_smoke:
        run_forward_smoke(model, val_loader, device)
    if args.tiny_overfit:
        run_tiny_overfit(cfg, device)
    if int(cfg["epochs"]) <= 0:
        return 0

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    ckpt_root = format_path(cfg["ckpt_dir"], cfg) / cfg["dataset"] / cfg["feature_mode"] / cfg["exp_tag"]
    out_dir = format_path(cfg["output_dir"], cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ade = float("inf")
    start = time.perf_counter()
    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_metrics = train_one_epoch(model, train_loader, device, optimizer, cfg, epoch)
        val_metrics = evaluate_model(model, val_loader, device, cfg)
        elapsed = time.perf_counter() - start
        print(f"[Epoch {epoch}] train_rmse={train_metrics['loss_rmse']:.4f} val_ADE={val_metrics['ade']:.4f} val_FDE={val_metrics['fde']:.4f} val_RMSE={val_metrics['rmse']:.4f} elapsed={elapsed:.1f}s")
        all_metrics = {"train": train_metrics, "val": val_metrics}
        save_checkpoint(ckpt_root / "last.pt", model, cfg, all_metrics, epoch)
        if float(val_metrics["ade"]) < best_ade:
            best_ade = float(val_metrics["ade"])
            save_checkpoint(ckpt_root / "best.pt", model, cfg, all_metrics, epoch)
            (out_dir / "best_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print("\n====== MTFT validation best/last ======")
    print_metrics(val_metrics)
    print(f"[Checkpoint] {ckpt_root / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
