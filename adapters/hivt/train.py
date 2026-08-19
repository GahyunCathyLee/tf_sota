#!/usr/bin/env python3
"""Train official HiVT on NeighFormer highD/exiD npy data."""

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
from adapters.hivt.dataset import NeighFormerHiVTDataset  # noqa: E402
from adapters.hivt.upstream import add_upstream_to_path, upstream_commit  # noqa: E402


DEFAULTS: dict[str, Any] = {
    "adapter": "hivt",
    "dataset": "",
    "feature_mode": "",
    "exp_tag": "",
    "data_root": "data",
    "eval_hz": 3.0,
    "batch_size": 128,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "seed": 42,
    "accelerator": "auto",
    "devices": 1,
    "epochs": 100,
    "ckpt_dir": "ckpts/hivt",
    "tensorboard_dir": "tensorboard/hivt",
    "output_dir": "runs/hivt/{dataset}/{feature_mode}/{exp_tag}",
    "lane_half_length": 120.0,
    "lane_cache_root": "{data_root}/{dataset}/simpl_lane_graph",
    "lane_radius": 120.0,
    "lane_max_segments": 192,
    "max_train_samples": None,
    "max_eval_samples": None,
    "upstream_dir": "external/hivt",
    "model_hparams": {},
    "smoke": {
        "epochs": 2,
        "batch_size": 16,
        "train_samples": 512,
        "eval_samples": 256,
        "model_hparams": {
            "embed_dim": 32,
            "num_heads": 4,
            "num_temporal_layers": 1,
            "num_global_layers": 1,
        },
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--mode", default="smoke", choices=["smoke", "full", "check-data"])
    p.add_argument("--dataset", choices=["highD", "exiD"])
    p.add_argument("--feature-mode", choices=["baseline", "dimI"])
    p.add_argument("--data-root", type=Path)
    p.add_argument("--ckpt-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--tensorboard-dir", type=Path)
    p.add_argument("--exp-tag")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--accelerator", choices=["auto", "gpu", "cpu"])
    p.add_argument("--devices", type=int)
    p.add_argument("--max-train-samples", type=int)
    p.add_argument("--max-eval-samples", type=int)
    p.add_argument("--lane-cache-root", type=Path)
    p.add_argument("--lane-radius", type=float)
    p.add_argument("--lane-max-segments", type=int)
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
    cfg = _deep_merge(DEFAULTS, {})
    for section in ("data", "training", "runtime"):
        block = raw.get(section)
        if isinstance(block, dict):
            for key, value in block.items():
                cfg[{"root": "data_root", "hz": "eval_hz", "n_workers": "num_workers"}.get(key, key)] = value
    cfg["model_hparams"] = raw.get("model", raw.get("model_hparams", {})) or {}
    if isinstance(raw.get("smoke"), dict):
        cfg["smoke"] = _deep_merge(cfg.get("smoke", {}), raw["smoke"])
    for key in ("adapter", "dataset", "feature_mode", "exp_tag"):
        if key in raw:
            cfg[key] = raw[key]
    return cfg


def apply_cli(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.feature_mode:
        cfg["feature_mode"] = args.feature_mode
    mode = "check-data" if args.check_data else args.mode
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
        ("lane_cache_root", "lane_cache_root"),
        ("lane_radius", "lane_radius"),
        ("lane_max_segments", "lane_max_segments"),
        ("upstream_dir", "upstream_dir"),
    ):
        value = getattr(args, cli_name)
        if value is not None:
            cfg[cfg_name] = value
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
    return resolve_path(
        str(value).format(
            dataset=cfg["dataset"],
            feature_mode=cfg["feature_mode"],
            exp_tag=cfg["exp_tag"],
            data_root=cfg["data_root"],
        )
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subset_indices(indices: np.ndarray, limit: int | None) -> np.ndarray:
    return indices if limit is None else indices[: int(limit)]


def build_model_args(cfg: dict[str, Any], ds: NeighFormerHiVTDataset) -> dict[str, Any]:
    model = {
        "historical_steps": ds.history_len,
        "future_steps": ds.future_len,
        "num_modes": 6,
        "rotate": False,
        "node_dim": ds.node_dim,
        "edge_dim": ds.edge_dim,
        "embed_dim": 64,
        "num_heads": 4,
        "dropout": 0.1,
        "num_temporal_layers": 2,
        "num_global_layers": 2,
        "local_radius": 50.0,
        "parallel": False,
        "lr": 5.0e-4,
        "weight_decay": 1.0e-4,
        "T_max": int(cfg["epochs"]),
    }
    model.update(cfg.get("model_hparams") or {})
    model["historical_steps"] = ds.history_len
    model["future_steps"] = ds.future_len
    model["node_dim"] = ds.node_dim
    model["edge_dim"] = ds.edge_dim
    model["T_max"] = int(model.get("T_max") or cfg["epochs"])
    if model["rotate"] and model["node_dim"] != 2:
        raise SystemExit("HiVT rotate=True is only compatible with node_dim=2; use rotate: false for baseline/dimI.")
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
    cfg["data_root"] = str(data_root)
    data_path = dataset_dir(data_root, cfg["dataset"])
    train_idx = subset_indices(np.load(split_indices_path(data_root, cfg["dataset"], "train")), cfg.get("max_train_samples"))
    val_idx = subset_indices(np.load(split_indices_path(data_root, cfg["dataset"], "val")), cfg.get("max_eval_samples"))
    lane_cache_root = format_path_template(cfg["lane_cache_root"], cfg) if cfg.get("lane_cache_root") else None

    train_ds = NeighFormerHiVTDataset(
        data_path,
        train_idx,
        cfg["dataset"],
        cfg["feature_mode"],
        "train",
        cfg["lane_half_length"],
        lane_cache_root=lane_cache_root,
        lane_radius=cfg["lane_radius"],
        lane_max_segments=cfg["lane_max_segments"],
    )
    val_ds = NeighFormerHiVTDataset(
        data_path,
        val_idx,
        cfg["dataset"],
        cfg["feature_mode"],
        "val",
        cfg["lane_half_length"],
        lane_cache_root=lane_cache_root,
        lane_radius=cfg["lane_radius"],
        lane_max_segments=cfg["lane_max_segments"],
    )

    output_dir = format_path_template(cfg["output_dir"], cfg)
    ckpt_dir = format_path_template(cfg["ckpt_dir"], cfg) / cfg["exp_tag"]
    tb_dir = format_path_template(cfg["tensorboard_dir"], cfg) / cfg["exp_tag"]
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    data_report = {"train": train_ds.describe(), "val": val_ds.describe(), "channels": train_ds.channel_stats()}
    (output_dir / "data_report.json").write_text(json.dumps(data_report, indent=2), encoding="utf-8")
    if cfg["mode"] == "check-data":
        print(json.dumps(data_report, indent=2))
        return 0

    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
        from torch_geometric.loader import DataLoader
    except ImportError as exc:
        raise SystemExit("HiVT dependencies are missing: pytorch_lightning and torch_geometric are required.") from exc

    upstream_dir = add_upstream_to_path(cfg["upstream_dir"])
    from models.hivt import HiVT  # noqa: WPS433

    loader_kwargs = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": int(cfg["num_workers"]),
        "pin_memory": bool(cfg["pin_memory"]),
        "persistent_workers": bool(cfg["persistent_workers"]) and int(cfg["num_workers"]) > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)

    model_args = build_model_args(cfg, train_ds)
    model = HiVT(**model_args)
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

    print("====== HiVT Train ======")
    print(f"upstream : {upstream_dir} ({upstream_commit(upstream_dir)})")
    print(f"data     : {data_path}")
    print(f"lanes    : {lane_cache_root if lane_cache_root else 'pseudo fallback'}")
    print(f"samples  : train={len(train_ds):,} val={len(val_ds):,}")
    print(f"node_dim : {train_ds.node_dim}")
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
