#!/usr/bin/env python3
"""Train official BAT modules on NeighFormer highD/exiD npy data."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

ADAPTER_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ADAPTER_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from adapters.bat.dataset import NeighFormerBATDataset, polar_to_cart, write_preprocess_manifest  # noqa: E402
from adapters.bat.upstream import import_bat_model, upstream_commit  # noqa: E402
from adapters.common import dataset_dir, split_indices_path  # noqa: E402

try:
    import yaml
except ImportError:
    yaml = None


DEFAULTS: dict[str, Any] = {
    "adapter": "bat",
    "dataset": "",
    "feature_mode": "",
    "exp_tag": "",
    "data_root": "/home/gahyun/neighformer/data",
    "eval_hz": 3.0,
    "dt": 0.32,
    "batch_size": 32,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "seed": 42,
    "device": "auto",
    "epochs": 100,
    "lr": 1.0e-3,
    "weight_decay": 0.0,
    "grad_clip_norm": 10.0,
    "log_interval": 100,
    "ckpt_dir": "ckpts/bat",
    "output_dir": "runs/bat/{dataset}/{feature_mode}/{exp_tag}",
    "processed_dir": "processed/bat",
    "reuse_processed": False,
    "max_train_samples": None,
    "max_eval_samples": None,
    "upstream_dir": "external/bat",
    "model_hparams": {},
    "smoke": {
        "epochs": 1,
        "batch_size": 4,
        "train_samples": 64,
        "eval_samples": 32,
        "num_workers": 0,
        "model_hparams": {
            "lstm_encoder_size": 32,
            "lstm_encoder_size_behavior": 8,
            "traj_linear_hidden": 16,
            "traj_linear_behavior": 4,
            "n_head": 2,
            "att_out": 16,
        },
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--mode", default="full", choices=["smoke", "full", "check-data", "preprocess"])
    p.add_argument("--dataset", choices=["highD", "exiD"])
    p.add_argument("--feature-mode", choices=["baseline", "dimI"])
    p.add_argument("--data-root", type=Path)
    p.add_argument("--ckpt-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--processed-dir", type=Path)
    p.add_argument("--reuse-processed", action="store_true")
    p.add_argument("--exp-tag")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--device")
    p.add_argument("--lr", type=float)
    p.add_argument("--max-train-samples", type=int)
    p.add_argument("--max-eval-samples", type=int)
    p.add_argument("--upstream-dir", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--check-data", action="store_true")
    return p.parse_args(argv)


def _parse_scalar(value: str) -> Any:
    value = value.strip().strip('"').strip("'")
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _simple_yaml_load(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        line = raw_line.split("#", 1)[0].rstrip()
        if ":" not in line or line.lstrip().startswith("-"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, value = line.strip().split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key.strip()] = child
            stack.append((indent, child))
        else:
            parent[key.strip()] = _parse_scalar(value)
    return root


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
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    raw = (yaml.safe_load(path.read_text(encoding="utf-8")) if yaml is not None else _simple_yaml_load(path.read_text(encoding="utf-8"))) or {}
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
    cfg = _deep_merge(DEFAULTS, {})
    for section in ("data", "training", "runtime"):
        block = raw.get(section)
        if isinstance(block, dict):
            for key, value in block.items():
                cfg[{"root": "data_root", "hz": "eval_hz", "accelerator": "device", "n_workers": "num_workers", "clip": "grad_clip_norm"}.get(key, key)] = value
    cfg["model_hparams"] = raw.get("model_hparams", raw.get("model", {})) or {}
    if isinstance(raw.get("smoke"), dict):
        cfg["smoke"] = _deep_merge(cfg.get("smoke", {}), raw["smoke"])
    for key in ("adapter", "dataset", "feature_mode", "exp_tag", "upstream_dir"):
        if key in raw:
            cfg[key] = raw[key]
    return cfg


def apply_cli(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mode = "check-data" if args.check_data else args.mode
    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.feature_mode:
        cfg["feature_mode"] = args.feature_mode
    if mode == "smoke":
        smoke = cfg.get("smoke") or {}
        for key, value in smoke.items():
            if key == "model_hparams" and isinstance(value, dict):
                cfg["model_hparams"] = _deep_merge(cfg.get("model_hparams", {}), value)
            else:
                cfg[{"train_samples": "max_train_samples", "eval_samples": "max_eval_samples"}.get(key, key)] = value
    for cli_name, cfg_name in (
        ("data_root", "data_root"),
        ("ckpt_dir", "ckpt_dir"),
        ("output_dir", "output_dir"),
        ("processed_dir", "processed_dir"),
        ("exp_tag", "exp_tag"),
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("num_workers", "num_workers"),
        ("seed", "seed"),
        ("device", "device"),
        ("lr", "lr"),
        ("max_train_samples", "max_train_samples"),
        ("max_eval_samples", "max_eval_samples"),
        ("upstream_dir", "upstream_dir"),
    ):
        value = getattr(args, cli_name)
        if value is not None:
            cfg[cfg_name] = value
    if args.reuse_processed:
        cfg["reuse_processed"] = True
    if not cfg["dataset"] or not cfg["feature_mode"]:
        raise SystemExit("dataset and feature_mode must be set by config or CLI")
    if not cfg["exp_tag"]:
        cfg["exp_tag"] = f"{cfg['dataset']}{1 if cfg['feature_mode'] == 'dimI' else 0}"
    cfg["mode"] = mode
    return cfg


def resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (EXPERIMENT_ROOT / p).resolve()


def format_path_template(value: str | Path, cfg: dict[str, Any]) -> Path:
    return resolve_path(str(value).format(dataset=cfg["dataset"], feature_mode=cfg["feature_mode"], exp_tag=cfg["exp_tag"]))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subset_indices(indices: np.ndarray, limit: int | None) -> np.ndarray:
    return indices if limit is None else indices[: int(limit)]


def build_dataset(cfg: dict[str, Any], split: str, limit: int | None = None) -> NeighFormerBATDataset:
    data_root = resolve_path(cfg["data_root"])
    indices = np.load(split_indices_path(data_root, cfg["dataset"], split))
    indices = subset_indices(indices, limit)
    hp = cfg.get("model_hparams") or {}
    return NeighFormerBATDataset(
        dataset_dir(data_root, cfg["dataset"]),
        indices,
        cfg["dataset"],
        cfg["feature_mode"],
        split,
        grid_size=(int(hp.get("grid_cols", 13)), int(hp.get("grid_rows", 3))),
        enc_size=int(hp.get("lstm_encoder_size", 64)),
        polar=bool(hp.get("polar", True)),
        longitudinal_cell=float(hp.get("longitudinal_cell", 15.0)),
        lane_width=float(hp.get("lane_width", 3.7)),
        neighbor_distance=float(hp.get("neighbor_distance", 100.0)),
    )


def build_data_report(cfg: dict[str, Any], train_ds: NeighFormerBATDataset, val_ds: NeighFormerBATDataset) -> dict[str, Any]:
    sample = train_ds[0] if len(train_ds) else {}
    sample_shapes = {
        key: list(value.shape) if hasattr(value, "shape") else None
        for key, value in sample.items()
        if key not in {"sample_index"}
    }
    valid_neighbors = int(sample.get("nbr_valid", np.zeros(0, dtype=bool)).sum()) if sample else 0
    return {
        "train": train_ds.describe(),
        "val": val_ds.describe(),
        "channels": train_ds.channel_stats(),
        "sample_shapes": sample_shapes,
        "sample_valid_neighbors": valid_neighbors,
        "upstream": {
            "url": "https://github.com/Petrichor625/BATraj-Behavior-aware-Model",
            "local_dir": str(resolve_path(cfg["upstream_dir"])),
            "commit": upstream_commit(cfg["upstream_dir"]),
        },
    }


def build_bat_args(cfg: dict[str, Any], ds: NeighFormerBATDataset, device: Any, train_flag: bool = True) -> dict[str, Any]:
    hp = cfg.get("model_hparams") or {}
    lstm_size = int(hp.get("lstm_encoder_size", 64))
    return {
        "device": device,
        "lstm_encoder_size": lstm_size,
        "lstm_encoder_carnum": 39,
        "lstm_encoder_size_behavior": int(hp.get("lstm_encoder_size_behavior", 16)),
        "n_head": int(hp.get("n_head", 4)),
        "att_out": int(hp.get("att_out", 48)),
        "in_length": int(ds.history_len),
        "out_length": int(ds.future_len),
        "f_length": 12,
        "traj_linear_hidden": int(hp.get("traj_linear_hidden", 32)),
        "behavior_size": 6,
        "traj_linear_behavior": int(hp.get("traj_linear_behavior", 4)),
        "batch_size": int(cfg["batch_size"]),
        "use_elu": bool(hp.get("use_elu", True)),
        "dropout": float(hp.get("dropout", 0.0)),
        "relu": float(hp.get("relu", 0.1)),
        "lat_length": 3,
        "lon_length": 3,
        "use_true_man": bool(hp.get("use_true_man", False)),
        "use_spatial": False,
        "use_maneuvers": bool(hp.get("use_maneuvers", True)),
        "cat_pred": bool(hp.get("cat_pred", True)),
        "use_mse": bool(hp.get("use_mse", False)),
        "pre_epoch": int(hp.get("pre_epoch", 7)),
        "val_use_mse": bool(hp.get("val_use_mse", True)),
        "train_flag": bool(train_flag),
    }


def require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("BAT training/evaluation requires PyTorch. Install torch before running smoke/full modes.") from exc
    return torch, DataLoader


def move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def mse_loss(pred: Any, fut: Any, mask: Any) -> Any:
    import torch

    acc = torch.zeros_like(mask)
    out = torch.pow(fut[:, :, 0] - pred[:, :, 0], 2) + torch.pow(fut[:, :, 1] - pred[:, :, 1], 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    return torch.sum(acc * mask) / torch.sum(mask).clamp(min=1.0)


def nll_loss(pred: Any, fut: Any, mask: Any) -> Any:
    import torch

    acc = torch.zeros_like(mask)
    mu_x, mu_y = pred[:, :, 0], pred[:, :, 1]
    sig_x, sig_y = pred[:, :, 2].clamp(min=1.0e-3), pred[:, :, 3].clamp(min=1.0e-3)
    rho = pred[:, :, 4].clamp(min=-0.999, max=0.999)
    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
    x, y = fut[:, :, 0], fut[:, :, 1]
    out = (
        0.5
        * torch.pow(ohr, 2)
        * (torch.pow(sig_x, 2) * torch.pow(x - mu_x, 2) + torch.pow(sig_y, 2) * torch.pow(y - mu_y, 2) - 2 * rho * sig_x * sig_y * (x - mu_x) * (y - mu_y))
        - torch.log(sig_x * sig_y * ohr)
        + 1.8379
    )
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    return torch.sum(acc * mask) / torch.sum(mask).clamp(min=1.0)


def ce_loss(pred: Any, target: Any) -> Any:
    import torch

    return -torch.log(torch.sum(pred * target, dim=-1).clamp(min=1.0e-8)).mean()


def forward_models(gd_encoder: Any, generator: Any, batch: dict[str, Any]) -> tuple[Any, Any, Any]:
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


def prediction_cart(pred: Any, polar: bool) -> Any:
    return polar_to_cart(pred[:, :, 0:2].permute(1, 0, 2)) if polar else pred[:, :, 0:2].permute(1, 0, 2)


def make_loader(ds: NeighFormerBATDataset, cfg: dict[str, Any], shuffle: bool, drop_last: bool = False):
    _, DataLoader = require_torch()
    workers = int(cfg["num_workers"])
    return DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=workers,
        pin_memory=bool(cfg["pin_memory"]),
        persistent_workers=bool(cfg["persistent_workers"]) and workers > 0,
        collate_fn=ds.collate_fn,
    )


def evaluate_epoch(gd_encoder: Any, generator: Any, loader: Any, device: Any, cfg: dict[str, Any], ds: NeighFormerBATDataset) -> tuple[dict[str, Any], float]:
    import torch
    from adapters.mtp_go.metrics import MetricAccumulator

    gd_encoder.eval()
    generator.eval()
    total_loss = 0.0
    total_batches = 0
    acc = MetricAccumulator(dt=float(cfg.get("dt", 1.0 / float(cfg["eval_hz"]))), hz=float(cfg["eval_hz"]))
    with torch.no_grad():
        for raw in loader:
            batch = move_batch(raw, device)
            pred, lat_pred, lon_pred = forward_models(gd_encoder, generator, batch)
            loss = mse_loss(pred, batch["fut"], batch["op_mask"])
            acc.update(prediction_cart(pred, ds.polar), batch["target"])
            total_loss += float(loss.detach())
            total_batches += 1
            _ = lat_pred, lon_pred
    return acc.result(), total_loss / max(1, total_batches)


def save_checkpoint(path: Path, gd_encoder: Any, generator: Any, cfg: dict[str, Any], model_args: dict[str, Any], epoch: int, val_metrics: dict[str, Any]) -> None:
    import torch

    serializable_args = {k: (str(v) if k == "device" else v) for k, v in model_args.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gd_encoder": gd_encoder.state_dict(),
            "generator": generator.state_dict(),
            "cfg": {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()},
            "model_args": serializable_args,
            "epoch": int(epoch),
            "val_metrics": val_metrics,
            "upstream_commit": upstream_commit(cfg.get("upstream_dir")),
        },
        path,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = apply_cli(load_config(args.config), args)
    set_seed(int(cfg["seed"]))

    train_ds = build_dataset(cfg, "train", cfg.get("max_train_samples"))
    val_ds = build_dataset(cfg, "val", cfg.get("max_eval_samples"))
    output_dir = format_path_template(cfg["output_dir"], cfg)
    ckpt_dir = format_path_template(cfg["ckpt_dir"], cfg) / cfg["exp_tag"]
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_report = build_data_report(cfg, train_ds, val_ds)
    (output_dir / "data_report.json").write_text(json.dumps(data_report, indent=2), encoding="utf-8")
    if cfg["mode"] == "check-data":
        print(json.dumps(data_report, indent=2))
        return 0
    if cfg["mode"] == "preprocess":
        reports = {"train": train_ds.describe(), "val": val_ds.describe()}
        for split in ("test",):
            reports[split] = build_dataset(cfg, split, cfg.get("max_eval_samples")).describe()
        manifest = write_preprocess_manifest(resolve_path(cfg["processed_dir"]), cfg["dataset"], cfg["feature_mode"], reports)
        print(f"[INFO] BAT computes graph tensors on the fly; manifest written -> {manifest}")
        return 0

    torch, _ = require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg["device"] == "auto" else torch.device(cfg["device"])
    GDEncoder, Generator, upstream_dir = import_bat_model(cfg["upstream_dir"])
    model_args = build_bat_args(cfg, train_ds, device, train_flag=True)
    gd_encoder = GDEncoder(model_args).to(device)
    generator = Generator(model_args).to(device)

    if args.resume:
        ckpt = torch.load(resolve_path(args.resume), map_location=device, weights_only=False)
        gd_encoder.load_state_dict(ckpt["gd_encoder"])
        generator.load_state_dict(ckpt["generator"])

    train_loader = make_loader(train_ds, cfg, shuffle=True, drop_last=True)
    val_loader = make_loader(val_ds, cfg, shuffle=False)
    opt_gd = torch.optim.Adam(gd_encoder.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    opt_g = torch.optim.Adam(generator.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    print("====== BAT Train ======")
    print(f"upstream : {upstream_dir} ({upstream_commit(upstream_dir)})")
    print(f"data     : {dataset_dir(resolve_path(cfg['data_root']), cfg['dataset'])}")
    print(f"samples  : train={len(train_ds):,} val={len(val_ds):,}")
    print(f"ckpt     : {ckpt_dir}")
    best_fde = float("inf")
    for epoch in range(1, int(cfg["epochs"]) + 1):
        gd_encoder.train()
        generator.train()
        total = 0.0
        for step, raw in enumerate(train_loader, start=1):
            batch = move_batch(raw, device)
            pred, lat_pred, lon_pred = forward_models(gd_encoder, generator, batch)
            reg = mse_loss(pred, batch["fut"], batch["op_mask"]) if epoch <= int(model_args["pre_epoch"]) or model_args["use_mse"] else nll_loss(pred, batch["fut"], batch["op_mask"])
            man = ce_loss(lat_pred, batch["lat_enc"]) + ce_loss(lon_pred, batch["lon_enc"])
            loss = reg + man
            opt_gd.zero_grad(set_to_none=True)
            opt_g.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gd_encoder.parameters(), float(cfg["grad_clip_norm"]))
            torch.nn.utils.clip_grad_norm_(generator.parameters(), float(cfg["grad_clip_norm"]))
            opt_gd.step()
            opt_g.step()
            total += float(loss.detach())
            if step % int(cfg["log_interval"]) == 0:
                print(f"epoch {epoch:03d} step {step:05d}/{len(train_loader):05d} loss={total / step:.4f}", flush=True)
        metrics, val_loss = evaluate_epoch(gd_encoder, generator, val_loader, device, cfg, train_ds)
        metrics["val_loss"] = val_loss
        print(f"epoch {epoch:03d} train_loss={total / max(1, len(train_loader)):.4f} val_loss={val_loss:.4f} ade={metrics.get('ade', float('nan')):.4f} fde={metrics.get('fde', float('nan')):.4f}")
        save_checkpoint(ckpt_dir / "last.pt", gd_encoder, generator, cfg, model_args, epoch, metrics)
        if float(metrics.get("fde", float("inf"))) < best_fde:
            best_fde = float(metrics["fde"])
            save_checkpoint(ckpt_dir / "best.pt", gd_encoder, generator, cfg, model_args, epoch, metrics)
    from adapters.mtp_go.metrics import print_metrics

    print_metrics(metrics)
    print(f"[DONE] best.pt saved -> {ckpt_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
