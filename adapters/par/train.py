#!/usr/bin/env python3
"""Train PAR-style trajectory-token model on NeighFormer highD/exiD npy data."""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ADAPTER_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ADAPTER_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from adapters.common import dataset_dir, split_indices_path  # noqa: E402
from adapters.mtp_go.metrics import MetricAccumulator, print_metrics  # noqa: E402
from adapters.par.dataset import NeighFormerPARDataset, reconstruct_accel_tokens  # noqa: E402
from adapters.par.model import PARTrajectoryModel  # noqa: E402
from adapters.par.upstream import add_upstream_to_path, upstream_commit  # noqa: E402


DEFAULTS: dict[str, Any] = {
    "adapter": "par",
    "dataset": "",
    "feature_mode": "",
    "exp_tag": "",
    "data_root": "data",
    "eval_hz": 3.0,
    "dt": 0.32,
    "batch_size": 1024,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "seed": 42,
    "device": "auto",
    "epochs": 100,
    "lr": 1.0e-4,
    "weight_decay": 0.05,
    "grad_clip_norm": 5.0,
    "log_interval": 100,
    "ckpt_dir": "ckpts/par",
    "tensorboard_dir": "tensorboard/par",
    "output_dir": "runs/par/{dataset}/{feature_mode}/{exp_tag}",
    "max_train_samples": None,
    "max_eval_samples": None,
    "upstream_dir": "external/par",
    "acc_token_size": 13,
    "velocity_bins": 128,
    "multinomial_sampling": False,
    "model_hparams": {},
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
    p.add_argument("--device")
    p.add_argument("--lr", type=float)
    p.add_argument("--max-train-samples", type=int)
    p.add_argument("--max-eval-samples", type=int)
    p.add_argument("--upstream-dir", type=Path)
    p.add_argument("--resume", type=Path)
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
                cfg[
                    {
                        "root": "data_root",
                        "hz": "eval_hz",
                        "clip": "grad_clip_norm",
                        "accelerator": "device",
                        "n_workers": "num_workers",
                    }.get(key, key)
                ] = value
    cfg["model_hparams"] = raw.get("model_hparams", raw.get("model", {})) or {}
    for key in ("acc_token_size", "velocity_bins", "multinomial_sampling"):
        if key in cfg["model_hparams"]:
            cfg[key] = cfg["model_hparams"][key]
    for key in ("adapter", "dataset", "feature_mode", "exp_tag", "upstream_dir"):
        if key in raw:
            cfg[key] = raw[key]
    if isinstance(raw.get("smoke"), dict):
        cfg["smoke"] = raw["smoke"]
    return cfg


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
        ("device", "device"),
        ("lr", "lr"),
        ("max_train_samples", "max_train_samples"),
        ("max_eval_samples", "max_eval_samples"),
        ("upstream_dir", "upstream_dir"),
    ):
        value = getattr(args, cli_name)
        if value is not None:
            cfg[cfg_name] = value
    if args.mode == "smoke":
        smoke = cfg.get("smoke") or {}
        for key, value in smoke.items():
            if key == "model_hparams" and isinstance(value, dict):
                cfg["model_hparams"] = _deep_merge(cfg.get("model_hparams", {}), value)
            else:
                cfg[{"train_samples": "max_train_samples", "eval_samples": "max_eval_samples"}.get(key, key)] = value
    if not cfg["dataset"] or not cfg["feature_mode"]:
        raise SystemExit("dataset and feature_mode must be set by config or CLI")
    if not cfg["exp_tag"]:
        cfg["exp_tag"] = f"{cfg['dataset']}{1 if cfg['feature_mode'] == 'dimI' else 0}"
    if not cfg.get("output_dir"):
        cfg["output_dir"] = DEFAULTS["output_dir"]
    cfg["mode"] = "check-data" if args.check_data else args.mode
    return cfg


def resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (EXPERIMENT_ROOT / p).resolve()


def format_path_template(value: str | Path, cfg: dict[str, Any]) -> Path:
    return resolve_path(str(value).format(dataset=cfg["dataset"], feature_mode=cfg["feature_mode"], exp_tag=cfg["exp_tag"]))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subset_indices(indices: np.ndarray, limit: int | None) -> np.ndarray:
    return indices if limit is None else indices[: int(limit)]


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def make_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: make_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_serializable(v) for v in value]
    return value


def build_model(cfg: dict[str, Any], ds: NeighFormerPARDataset) -> PARTrajectoryModel:
    hp = cfg.get("model_hparams") or {}
    return PARTrajectoryModel(
        vocab_size=ds.pad_index + 1,
        pad_index=ds.pad_index,
        num_agents=ds.num_agents,
        side_dim=ds.side_dim,
        transformer=hp.get("transformer", {}),
        use_agent_embedding=bool(hp.get("use_agent_embedding", True)),
        dropout_hidden=float(hp.get("dropout_hidden", 0.0)),
        dropout_attn=float(hp.get("dropout_attn", 0.0)),
    )


@torch.no_grad()
def predict_batch(
    model: PARTrajectoryModel,
    batch: dict[str, torch.Tensor],
    ds: NeighFormerPARDataset,
    multinomial_sampling: bool = False,
) -> torch.Tensor:
    generated = model.generate_ego_future_tokens(
        batch,
        obs_token_steps=ds.obs_token_steps,
        future_len=ds.future_len,
        multinomial_sampling=multinomial_sampling,
        generate_all_agents=bool(ds.has_neighbor_future),
    )
    prefix = batch["ego_tokens"][:, : ds.obs_token_steps]
    ego_tokens = torch.cat([prefix, generated], dim=1)
    hist = batch["hist_pos"]
    start = hist[:, 0]
    start_velocity = hist[:, 1] - hist[:, 0]
    recon = reconstruct_accel_tokens(start, start_velocity, ego_tokens, ds.bins, ds.acc_token_size, ds.pad_index)
    return recon[:, ds.history_len : ds.history_len + ds.future_len]


@torch.no_grad()
def evaluate_model(
    model: PARTrajectoryModel,
    loader: DataLoader,
    ds: NeighFormerPARDataset,
    device: torch.device,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    acc = MetricAccumulator(dt=float(cfg.get("dt", 1.0 / float(cfg["eval_hz"]))), hz=float(cfg["eval_hz"]))
    total_loss = 0.0
    total_batches = 0
    total_tokens = 0
    for raw in loader:
        batch = move_batch(raw, device)
        loss, parts = model.loss(batch)
        pred = predict_batch(model, batch, ds, bool(cfg.get("multinomial_sampling", False)))
        target = batch["target"]
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        acc.update(pred, target)
        total_loss += float(loss.detach())
        total_tokens += int(parts["tokens_supervised"])
        total_batches += 1
    out = acc.result()
    out.update({"loss": total_loss / max(1, total_batches), "tokens_supervised": total_tokens})
    return out


def runtime_report(cfg: dict[str, Any], upstream_dir: Path) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "upstream_dir": str(upstream_dir),
        "upstream_commit": upstream_commit(upstream_dir),
        "argv": sys.argv,
        "config": make_serializable(cfg),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = apply_cli(load_config(args.config), args)
    set_seed(int(cfg["seed"]))
    upstream_dir = add_upstream_to_path(cfg["upstream_dir"])

    requested_device = str(cfg["device"]).lower()
    use_cuda = requested_device in {"auto", "gpu", "cuda"} and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data_root = resolve_path(cfg["data_root"])
    data_path = dataset_dir(data_root, cfg["dataset"])
    train_idx = subset_indices(np.load(split_indices_path(data_root, cfg["dataset"], "train")), cfg.get("max_train_samples"))
    val_idx = subset_indices(np.load(split_indices_path(data_root, cfg["dataset"], "val")), cfg.get("max_eval_samples"))
    train_ds = NeighFormerPARDataset(
        data_path,
        train_idx,
        cfg["dataset"],
        cfg["feature_mode"],
        "train",
        acc_token_size=int(cfg["acc_token_size"]),
        velocity_bins=int(cfg["velocity_bins"]),
    )
    val_ds = NeighFormerPARDataset(
        data_path,
        val_idx,
        cfg["dataset"],
        cfg["feature_mode"],
        "val",
        acc_token_size=int(cfg["acc_token_size"]),
        velocity_bins=int(cfg["velocity_bins"]),
    )

    output_dir = format_path_template(cfg["output_dir"], cfg)
    ckpt_dir = (format_path_template(cfg["ckpt_dir"], cfg) / cfg["exp_tag"]) if cfg.get("ckpt_dir") else (output_dir / "checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data_report = {"train": train_ds.describe(), "val": val_ds.describe(), "channels": train_ds.channel_stats()}
    (output_dir / "data_report.json").write_text(json.dumps(data_report, indent=2), encoding="utf-8")
    cfg_to_write = make_serializable(cfg)
    (output_dir / "run_config.yaml").write_text(yaml.safe_dump(cfg_to_write, sort_keys=False), encoding="utf-8")
    (output_dir / "runtime.json").write_text(json.dumps(runtime_report(cfg, upstream_dir), indent=2), encoding="utf-8")
    if cfg["mode"] == "check-data":
        print(json.dumps(data_report, indent=2))
        return 0

    loader_kwargs = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": int(cfg["num_workers"]),
        "pin_memory": bool(cfg["pin_memory"]) and device.type == "cuda",
        "persistent_workers": bool(cfg["persistent_workers"]) and int(cfg["num_workers"]) > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    model = build_model(cfg, train_ds).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    start_epoch = 1
    best_score = float("inf")
    if args.resume:
        resume_path = args.resume if args.resume.is_absolute() else resolve_path(args.resume)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_score = float(ckpt.get("best_score", best_score))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_path = output_dir / "train.log"

    def log(msg: str) -> None:
        print(msg)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("====== PAR Train ======")
    log(f"upstream : {upstream_dir} ({upstream_commit(upstream_dir)})")
    log(f"data     : {data_path}")
    log(f"samples  : train={len(train_ds):,} val={len(val_ds):,}")
    log(f"mode     : {cfg['mode']}  feature={cfg['feature_mode']}  side_dim={train_ds.side_dim}")
    log(f"device   : {device}")
    log(f"params   : {n_params:,}")
    log(f"ckpt     : {ckpt_dir}")

    last_val: dict[str, Any] = {"ade": float("inf"), "loss": float("inf")}
    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        n_batches = 0
        for step, raw in enumerate(train_loader, start=1):
            batch = move_batch(raw, device)
            loss, parts = model.loss(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip_norm"]))
            optimizer.step()
            total_loss += float(loss.detach())
            total_tokens += int(parts["tokens_supervised"])
            n_batches += 1
            if int(cfg["log_interval"]) > 0 and step % int(cfg["log_interval"]) == 0:
                log(f"epoch={epoch:03d} step={step:05d} loss={total_loss / max(1, n_batches):.4f}")

        train_loss = total_loss / max(1, n_batches)
        last_val = evaluate_model(model, val_loader, val_ds, device, cfg)
        log(
            f"Epoch {epoch:03d}/{int(cfg['epochs'])} "
            f"loss={train_loss:.4f} val_loss={last_val['loss']:.4f} "
            f"ADE={last_val['ade']:.3f} FDE={last_val['fde']:.3f} RMSE={last_val['rmse']:.3f} "
            f"tokens={total_tokens}"
        )
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg,
            "dataset": train_ds.describe(),
            "best_score": best_score,
            "num_parameters": int(n_params),
            "upstream_commit": upstream_commit(upstream_dir),
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if last_val["ade"] < best_score:
            best_score = float(last_val["ade"])
            ckpt["best_score"] = best_score
            torch.save(ckpt, ckpt_dir / "best.pt")
            (output_dir / "metrics.json").write_text(json.dumps(last_val, indent=2), encoding="utf-8")
            log(f"  best saved -> {ckpt_dir / 'best.pt'}")

    print_metrics(last_val)
    log(f"[DONE] best ADE={best_score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
