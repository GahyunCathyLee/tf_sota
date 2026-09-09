#!/usr/bin/env python3
"""Train official SIMPL on NeighFormer highD/exiD npy data."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ADAPTER_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ADAPTER_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from adapters.common import dataset_dir, split_indices_path  # noqa: E402
from adapters.mtp_go.metrics import MetricAccumulator, print_metrics  # noqa: E402
from adapters.simpl.dataset import NeighFormerSIMPLDataset  # noqa: E402
from adapters.simpl.upstream import add_upstream_to_path, upstream_commit  # noqa: E402


DEFAULTS: dict[str, Any] = {
    "adapter": "simpl",
    "dataset": "",
    "feature_mode": "",
    "exp_tag": "",
    "data_root": "data",
    "eval_hz": 3.0,
    "batch_size": 1024,
    "micro_batch_size": None,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "seed": 42,
    "device": "cuda",
    "epochs": 100,
    "lr": 5.0e-4,
    "weight_decay": 0.0,
    "grad_clip_norm": 5.0,
    "ckpt_dir": "ckpts/simpl",
    "tensorboard_dir": "tensorboard/simpl",
    "output_dir": "runs/simpl/{dataset}/{feature_mode}/{exp_tag}",
    "lane_half_length": 120.0,
    "lane_cache_root": "{data_root}/{dataset}/simpl_lane_graph",
    "lane_radius": 120.0,
    "lane_max_segments": 192,
    "log_interval": 200,
    "max_train_samples": None,
    "max_eval_samples": None,
    "upstream_dir": "external/simpl",
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
    p.add_argument("--exp-tag")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--micro-batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--device")
    p.add_argument("--lr", type=float)
    p.add_argument("--max-train-samples", type=int)
    p.add_argument("--max-eval-samples", type=int)
    p.add_argument("--lane-cache-root", type=Path)
    p.add_argument("--lane-radius", type=float)
    p.add_argument("--lane-max-segments", type=int)
    p.add_argument("--log-interval", type=int)
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
                cfg[{"root": "data_root", "hz": "eval_hz"}.get(key, key)] = value
    cfg["model_hparams"] = raw.get("model", {})
    for key in ("adapter", "dataset", "feature_mode", "exp_tag"):
        if key in raw:
            cfg[key] = raw[key]
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
        ("micro_batch_size", "micro_batch_size"),
        ("num_workers", "num_workers"),
        ("seed", "seed"),
        ("device", "device"),
        ("lr", "lr"),
        ("max_train_samples", "max_train_samples"),
        ("max_eval_samples", "max_eval_samples"),
        ("lane_cache_root", "lane_cache_root"),
        ("lane_radius", "lane_radius"),
        ("lane_max_segments", "lane_max_segments"),
        ("log_interval", "log_interval"),
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


def build_model_config(cfg: dict[str, Any], actor_feature_dim: int) -> dict[str, Any]:
    model = {
        "init_weights": False,
        "in_actor": actor_feature_dim,
        "d_actor": 128,
        "n_fpn_scale": 2,
        "in_lane": 10,
        "d_lane": 128,
        "d_rpe_in": 5,
        "d_rpe": 128,
        "d_embed": 128,
        "n_scene_layer": 4,
        "n_scene_head": 8,
        "dropout": 0.1,
        "update_edge": True,
        "param_out": "none",
        "param_order": 5,
        "g_num_modes": 6,
        "g_obs_len": 6,
        "g_pred_len": 15,
    }
    model.update(cfg.get("model_hparams") or {})
    model["in_actor"] = actor_feature_dim
    model["g_obs_len"] = 6
    model["g_pred_len"] = 15
    return model


def flatten_simpl_targets(
    out,
    data: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    cls_list, reg_list = out[0], out[1]
    pred_rows = []
    target_rows = []
    mode_rows = []
    mask_rows = []
    scene_rows: list[int] = []
    for scene_idx, (cls_scene, reg_scene, fut_scene, pad_scene) in enumerate(
        zip(cls_list, reg_list, data["TRAJS_FUT"], data["PAD_FUT"])
    ):
        target = fut_scene.to(device)
        valid_mask = pad_scene.to(device).bool()
        keep = valid_mask.any(dim=-1)
        if not bool(keep.any()):
            continue
        modes = reg_scene[keep, :, :, :2]
        probs = cls_scene[keep]
        best = probs.argmax(dim=-1)
        pred_rows.append(modes[torch.arange(modes.shape[0], device=device), best])
        target_rows.append(target[keep])
        mode_rows.append(modes.permute(0, 2, 1, 3).contiguous())
        mask_rows.append(valid_mask[keep])
        scene_rows.extend([scene_idx] * int(keep.sum().item()))
    if not pred_rows:
        empty = torch.empty(0, 0, 2, device=device)
        empty_modes = torch.empty(0, 0, 0, 2, device=device)
        empty_mask = torch.empty(0, 0, dtype=torch.bool, device=device)
        return empty, empty, empty_modes, empty_mask, []
    return (
        torch.cat(pred_rows, dim=0),
        torch.cat(target_rows, dim=0),
        torch.cat(mode_rows, dim=0),
        torch.cat(mask_rows, dim=0),
        scene_rows,
    )


def simpl_loss(out, data: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    cls_list, reg_list = out[0], out[1]
    cls_rows = []
    reg_rows = []
    target_rows = []
    mask_rows = []
    for cls_scene, reg_scene, fut_scene, pad_scene in zip(cls_list, reg_list, data["TRAJS_FUT"], data["PAD_FUT"]):
        target = fut_scene.to(device)
        valid_mask = pad_scene.to(device).bool()
        keep = valid_mask.any(dim=-1)
        if not bool(keep.any()):
            continue
        cls_rows.append(cls_scene[keep])
        reg_rows.append(reg_scene[keep, :, :, :2])
        target_rows.append(target[keep])
        mask_rows.append(valid_mask[keep])
    if not cls_rows:
        zero = sum(x.sum() * 0.0 for x in reg_list)
        return zero, {"loss": 0.0, "reg_loss": 0.0, "cls_loss": 0.0}
    cls = torch.cat(cls_rows, dim=0)
    reg = torch.cat(reg_rows, dim=0)
    target = torch.cat(target_rows, dim=0)
    valid_mask = torch.cat(mask_rows, dim=0)
    dist = torch.norm(reg - target.unsqueeze(1), dim=-1)
    counts = valid_mask.float().sum(dim=-1).clamp_min(1.0)
    mode_ade = (dist * valid_mask.unsqueeze(1).float()).sum(dim=-1) / counts.unsqueeze(1)
    best = mode_ade.argmin(dim=-1)
    row = torch.arange(reg.shape[0], device=device)
    reg_best = reg[row, best]
    reg_loss = F.smooth_l1_loss(reg_best[valid_mask], target[valid_mask], reduction="mean")
    cls_loss = F.nll_loss(torch.log(cls.clamp_min(1e-8)), best)
    loss = reg_loss + 0.1 * cls_loss
    return loss, {"loss": float(loss.detach()), "reg_loss": float(reg_loss.detach()), "cls_loss": float(cls_loss.detach())}


@torch.no_grad()
def evaluate_model(model, loader, device: torch.device, hz: float) -> dict[str, Any]:
    model.eval()
    acc = MetricAccumulator(dt=1.0 / hz, hz=hz)
    total_loss = total_reg = total_cls = n_batches = 0.0
    for data in loader:
        out = model(model.pre_process(data))
        loss, parts = simpl_loss(out, data, device)
        chosen, target, all_modes, valid_mask, _ = flatten_simpl_targets(out, data, device)
        acc.update(chosen, target, all_modes=all_modes, valid_mask=valid_mask)
        total_loss += float(loss)
        total_reg += parts["reg_loss"]
        total_cls += parts["cls_loss"]
        n_batches += 1
    out = acc.result()
    denom = max(1.0, n_batches)
    out.update({"loss": total_loss / denom, "reg_loss": total_reg / denom, "cls_loss": total_cls / denom})
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = apply_cli(load_config(args.config), args)
    set_seed(int(cfg["seed"]))
    upstream_dir = add_upstream_to_path(cfg["upstream_dir"])
    from simpl.simpl import Simpl  # noqa: WPS433

    want_cuda = str(cfg["device"]).lower() == "cuda"
    device = torch.device("cuda" if want_cuda and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data_root = resolve_path(cfg["data_root"])
    data_path = dataset_dir(data_root, cfg["dataset"])
    lane_cache_root = format_path_template(cfg["lane_cache_root"], cfg) if cfg.get("lane_cache_root") else None
    train_idx = subset_indices(np.load(split_indices_path(data_root, cfg["dataset"], "train")), cfg.get("max_train_samples"))
    val_idx = subset_indices(np.load(split_indices_path(data_root, cfg["dataset"], "val")), cfg.get("max_eval_samples"))
    train_ds = NeighFormerSIMPLDataset(
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
    val_ds = NeighFormerSIMPLDataset(
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
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data_report = {"train": train_ds.describe(), "val": val_ds.describe(), "channels": train_ds.channel_stats()}
    (output_dir / "data_report.json").write_text(json.dumps(data_report, indent=2), encoding="utf-8")
    if args.check_data:
        print(json.dumps(data_report, indent=2))
        return 0

    effective_batch_size = int(cfg["batch_size"])
    train_batch_size = int(cfg.get("micro_batch_size") or cfg["batch_size"])
    accum_steps = max(1, int(math.ceil(effective_batch_size / train_batch_size)))
    loader_kwargs = {
        "batch_size": train_batch_size,
        "num_workers": int(cfg["num_workers"]),
        "pin_memory": bool(cfg["pin_memory"]),
        "persistent_workers": bool(cfg["persistent_workers"]) and int(cfg["num_workers"]) > 0,
        "collate_fn": train_ds.collate_fn,
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    model_cfg = build_model_config(cfg, train_ds.actor_feature_dim)
    model = Simpl(model_cfg, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    print("====== SIMPL Train ======")
    print(f"upstream : {upstream_dir} ({upstream_commit(upstream_dir)})")
    print(f"data     : {data_path}")
    print(f"lanes    : {lane_cache_root if lane_cache_root else 'pseudo fallback'}")
    print(f"samples  : train={len(train_ds):,} val={len(val_ds):,}")
    print(f"batch    : micro={train_batch_size} effective={effective_batch_size} accum={accum_steps}")
    print(f"device   : {device}")
    print(f"ckpt     : {ckpt_dir}")

    best_score = float("inf")
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        totals = {"loss": 0.0, "reg_loss": 0.0, "cls_loss": 0.0}
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)
        epoch_start = time.perf_counter()
        for data in train_loader:
            out = model(model.pre_process(data))
            loss, parts = simpl_loss(out, data, device)
            (loss / accum_steps).backward()
            if (n_batches + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip_norm"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for k in totals:
                totals[k] += parts[k]
            n_batches += 1
            if int(cfg["log_interval"]) > 0 and n_batches % int(cfg["log_interval"]) == 0:
                elapsed = time.perf_counter() - epoch_start
                batches_total = len(train_loader)
                samples_seen = min(n_batches * train_batch_size, len(train_ds))
                avg_loss = totals["loss"] / max(1, n_batches)
                print(
                    f"  train epoch={epoch:03d} "
                    f"batch={n_batches:,}/{batches_total:,} "
                    f"samples={samples_seen:,}/{len(train_ds):,} "
                    f"loss={avg_loss:.4f} "
                    f"elapsed={elapsed / 60.0:.1f}m",
                    flush=True,
                )
        if n_batches % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip_norm"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        train_loss = totals["loss"] / max(1, n_batches)
        val = evaluate_model(model, val_loader, device, float(cfg["eval_hz"]))
        print(
            f"Epoch {epoch:03d}/{int(cfg['epochs'])} "
            f"loss={train_loss:.4f} val_loss={val['loss']:.4f} "
            f"ADE={val['ade']:.3f} FDE={val['fde']:.3f} RMSE={val['rmse']:.3f}"
        )
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg,
            "model_cfg": model_cfg,
            "best_score": best_score,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val["ade"] < best_score:
            best_score = float(val["ade"])
            ckpt["best_score"] = best_score
            torch.save(ckpt, ckpt_dir / "best.pt")
            (output_dir / "metrics.json").write_text(json.dumps(val, indent=2), encoding="utf-8")
            print(f"  best saved -> {ckpt_dir / 'best.pt'}")
    print_metrics(val)
    print(f"[DONE] best ADE={best_score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
