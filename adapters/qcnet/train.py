#!/usr/bin/env python3
"""Train official QCNet on NeighFormer highD/exiD npy data."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ADAPTER_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ADAPTER_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from adapters.common import dataset_dir, split_indices_path  # noqa: E402
from adapters.qcnet.dataset import NeighFormerQCNetDataset  # noqa: E402
from adapters.qcnet.upstream import add_upstream_to_path, upstream_commit  # noqa: E402


DEFAULTS: dict[str, Any] = {
    "dataset": "",
    "feature_mode": "",
    "exp_tag": "",
    "data_root": "data",
    "eval_hz": 3.0,
    "batch_size": 32,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "seed": 42,
    "accelerator": "auto",
    "devices": 1,
    "epochs": 100,
    "ckpt_dir": "ckpts/qcnet",
    "tensorboard_dir": "tensorboard/qcnet",
    "output_dir": "runs/qcnet/{dataset}/{feature_mode}/{exp_tag}",
    "lane_half_length": 120.0,
    "max_train_samples": None,
    "max_eval_samples": None,
    "upstream_dir": "external/qcnet",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--dataset", choices=["highD", "exiD"])
    p.add_argument("--feature-mode", choices=["baseline", "dimI"])
    p.add_argument("--data-root", type=Path)
    p.add_argument("--ckpt-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--tensorboard-dir", type=Path)
    p.add_argument("--exp-tag", type=str)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--accelerator", choices=["auto", "gpu", "cpu"])
    p.add_argument("--devices", type=int)
    p.add_argument("--max-train-samples", type=int)
    p.add_argument("--max-eval-samples", type=int)
    p.add_argument("--upstream-dir", type=Path)
    p.add_argument("--check-data", action="store_true")
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
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
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
    for section in ("data", "training", "runtime"):
        block = raw.get(section)
        if isinstance(block, dict):
            for key, value in block.items():
                mapped = {"root": "data_root", "hz": "eval_hz"}.get(key, key)
                cfg[mapped] = value
    cfg["model_hparams"] = raw.get("model", {})
    for key in ("dataset", "feature_mode", "exp_tag"):
        if key in raw:
            cfg[key] = raw[key]
    return cfg


def resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (EXPERIMENT_ROOT / p).resolve()


def format_path_template(value: str | Path, cfg: dict[str, Any]) -> Path:
    return resolve_path(str(value).format(
        dataset=cfg["dataset"],
        feature_mode=cfg["feature_mode"],
        exp_tag=cfg["exp_tag"],
    ))


def apply_cli(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.feature_mode:
        cfg["feature_mode"] = args.feature_mode
    for cli_name, cfg_name in (
        ("data_root", "data_root"),
        ("ckpt_dir", "ckpt_dir"),
        ("output_dir", "output_dir"),
        ("tensorboard_dir", "tensorboard_dir"),
        ("exp_tag", "exp_tag"),
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("num_workers", "num_workers"),
        ("seed", "seed"),
        ("accelerator", "accelerator"),
        ("devices", "devices"),
        ("max_train_samples", "max_train_samples"),
        ("max_eval_samples", "max_eval_samples"),
        ("upstream_dir", "upstream_dir"),
    ):
        value = getattr(args, cli_name)
        if value is not None:
            cfg[cfg_name] = value
    if not cfg["dataset"] or not cfg["feature_mode"]:
        raise SystemExit("dataset and feature_mode must be set by config or CLI")
    if not cfg["exp_tag"]:
        cfg["exp_tag"] = f"{cfg['dataset']}{1 if cfg['feature_mode'] == 'dimI' else 0}"
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subset_indices(indices: np.ndarray, limit: int | None) -> np.ndarray:
    return indices if limit is None else indices[: int(limit)]


def build_model_args(cfg: dict[str, Any]) -> dict[str, Any]:
    model = {
        "dataset": "argoverse_v2",
        "input_dim": 2,
        "hidden_dim": 64,
        "output_dim": 2,
        "output_head": False,
        "num_historical_steps": 6,
        "num_future_steps": 15,
        "num_modes": 6,
        "num_recurrent_steps": 3,
        "num_freq_bands": 32,
        "num_map_layers": 1,
        "num_agent_layers": 2,
        "num_dec_layers": 2,
        "num_heads": 4,
        "head_dim": 16,
        "dropout": 0.1,
        "pl2pl_radius": 150.0,
        "time_span": 6,
        "pl2a_radius": 150.0,
        "a2a_radius": 150.0,
        "num_t2m_steps": 6,
        "pl2m_radius": 150.0,
        "a2m_radius": 150.0,
        "lr": 5.0e-4,
        "weight_decay": 1.0e-4,
        "T_max": int(cfg["epochs"]),
        "submission_dir": "./",
        "submission_file_name": "submission",
    }
    model.update(cfg.get("model_hparams") or {})
    model["num_historical_steps"] = 6
    model["num_future_steps"] = 15
    model["T_max"] = int(model.get("T_max") or cfg["epochs"])
    return model


def write_adapter_checkpoint(src_ckpt: Path, dst: Path, cfg: dict[str, Any], model_args: dict[str, Any]) -> None:
    lightning_ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": lightning_ckpt["state_dict"],
            "cfg": cfg,
            "model_args": model_args,
            "source_lightning_ckpt": str(src_ckpt),
            "epoch": lightning_ckpt.get("epoch"),
            "global_step": lightning_ckpt.get("global_step"),
        },
        dst,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = apply_cli(load_config(args.config), args)
    set_seed(int(cfg["seed"]))

    data_root = resolve_path(cfg["data_root"])
    data_path = dataset_dir(data_root, cfg["dataset"])
    train_idx = np.load(split_indices_path(data_root, cfg["dataset"], "train"))
    val_idx = np.load(split_indices_path(data_root, cfg["dataset"], "val"))
    train_idx = subset_indices(train_idx, cfg.get("max_train_samples"))
    val_idx = subset_indices(val_idx, cfg.get("max_eval_samples"))

    train_ds = NeighFormerQCNetDataset(
        data_path, train_idx, cfg["dataset"], cfg["feature_mode"], "train", cfg["lane_half_length"]
    )
    val_ds = NeighFormerQCNetDataset(
        data_path, val_idx, cfg["dataset"], cfg["feature_mode"], "val", cfg["lane_half_length"]
    )

    output_dir = format_path_template(cfg["output_dir"], cfg)
    ckpt_dir = format_path_template(cfg["ckpt_dir"], cfg) / cfg["exp_tag"]
    tb_dir = format_path_template(cfg["tensorboard_dir"], cfg) / cfg["exp_tag"]
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    data_report = {"train": train_ds.describe(), "val": val_ds.describe(), "channels": train_ds.channel_stats()}
    (output_dir / "data_report.json").write_text(json.dumps(data_report, indent=2), encoding="utf-8")
    if args.check_data:
        print(json.dumps(data_report, indent=2))
        return 0

    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
        from torch_geometric.loader import DataLoader
    except ImportError as exc:
        raise SystemExit(
            "QCNet dependencies are missing. Install PyTorch Geometric, torch_cluster, "
            "torch_scatter, torchvision, and pytorch_lightning before training."
        ) from exc

    upstream_dir = add_upstream_to_path(cfg["upstream_dir"])
    try:
        from adapters.qcnet.model import build_qcnet
    except ImportError as exc:
        raise SystemExit(
            "Could not import the official QCNet model. Install the upstream dependencies first, "
            "notably torchvision plus PyG extension packages torch_cluster and torch_scatter."
        ) from exc

    loader_kwargs = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": int(cfg["num_workers"]),
        "pin_memory": bool(cfg["pin_memory"]),
        "persistent_workers": bool(cfg["persistent_workers"]) and int(cfg["num_workers"]) > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)

    model_args = build_model_args(cfg)
    model = build_qcnet(model_args, cfg["feature_mode"])
    callbacks = [
        ModelCheckpoint(dirpath=str(ckpt_dir), filename="lightning-best", monitor="val_minFDE", mode="min",
                        save_top_k=1, save_last=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = pl.Trainer(
        accelerator=str(cfg["accelerator"]),
        devices=int(cfg["devices"]),
        max_epochs=int(cfg["epochs"]),
        default_root_dir=str(tb_dir),
        callbacks=callbacks,
        logger=True,
    )

    print("====== QCNet Train ======")
    print(f"upstream : {upstream_dir} ({upstream_commit(upstream_dir)})")
    print(f"data     : {data_path}")
    print(f"samples  : train={len(train_ds):,} val={len(val_ds):,}")
    print(f"ckpt     : {ckpt_dir}")
    trainer.fit(model, train_loader, val_loader)

    checkpoint_cb = callbacks[0]
    best_source = Path(checkpoint_cb.best_model_path) if checkpoint_cb.best_model_path else ckpt_dir / "last.ckpt"
    if not best_source.exists():
        raise SystemExit("Lightning did not produce a checkpoint.")
    write_adapter_checkpoint(best_source, ckpt_dir / "best.pt", cfg, model_args)
    last_source = ckpt_dir / "last.ckpt"
    if last_source.exists():
        write_adapter_checkpoint(last_source, ckpt_dir / "last.pt", cfg, model_args)
    shutil.copy2(best_source, output_dir / "best.lightning.ckpt")
    print(f"[DONE] best.pt saved -> {ckpt_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
