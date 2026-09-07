#!/usr/bin/env python3
"""Train official MTR on NeighFormer highD/exiD npy data."""

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

from adapters.common import dataset_dir, split_indices_path  # noqa: E402
from adapters.mtrpp.dataset import (  # noqa: E402
    NeighFormerMTRDataset,
    build_intention_points_from_data,
    processed_root,
    save_processed_split,
)
from adapters.mtrpp.upstream import import_motion_transformer, upstream_commit, using_cuda_op_stubs  # noqa: E402

try:
    import yaml
except ImportError:
    yaml = None


DEFAULTS: dict[str, Any] = {
    "adapter": "mtrpp",
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
    "lr": 1.0e-4,
    "weight_decay": 0.01,
    "grad_clip_norm": 1000.0,
    "log_interval": 50,
    "ckpt_dir": "ckpts/mtrpp",
    "output_dir": "runs/mtrpp/{dataset}/{feature_mode}/{exp_tag}",
    "processed_dir": "processed/mtrpp",
    "reuse_processed": False,
    "preprocess_splits": ["train", "val", "test"],
    "preprocess_shard_size": 4096,
    "max_train_samples": None,
    "max_eval_samples": None,
    "upstream_dir": "external/mtrpp",
    "global_attention_fallback": False,
    "model_hparams": {},
    "smoke": {
        "epochs": 1,
        "batch_size": 4,
        "train_samples": 64,
        "eval_samples": 32,
        "num_workers": 0,
        "global_attention_fallback": True,
        "model_hparams": {
            "d_model": 64,
            "map_d_model": 64,
            "num_attn_layers": 1,
            "num_decoder_layers": 1,
            "num_attn_head": 4,
            "num_channel_agent": 64,
            "num_channel_map": 32,
            "num_motion_modes": 6,
            "num_map_polylines": 7,
        },
    },
}


class AttrDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


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
    p.add_argument("--global-attention-fallback", action="store_true")
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
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [_parse_scalar(part) for part in inner.split(",")]
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
    text = path.read_text(encoding="utf-8")
    raw = (yaml.safe_load(text) if yaml is not None else _simple_yaml_load(text)) or {}
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
                cfg[{"root": "data_root", "hz": "eval_hz", "accelerator": "device", "n_workers": "num_workers"}.get(key, key)] = value
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
    if args.global_attention_fallback:
        cfg["global_attention_fallback"] = True
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


def to_attrdict(value: Any) -> Any:
    if isinstance(value, dict):
        return AttrDict({k: to_attrdict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [to_attrdict(v) for v in value]
    return value


def plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def build_builder_kwargs(cfg: dict[str, Any], hp: dict[str, Any] | None = None) -> dict[str, Any]:
    hp = hp or cfg.get("model_hparams", {}) or {}
    return {
        "dt": float(cfg["dt"]),
        "map_polylines": int(hp.get("num_map_polylines", 9)),
        "map_points_each_polyline": int(hp.get("num_points_each_polyline", 20)),
        "lane_half_length": float(hp.get("lane_half_length", 160.0)),
        "lane_width": float(hp.get("lane_width", 3.7)),
    }


def build_model_config(cfg: dict[str, Any], ds: NeighFormerMTRDataset, intention_file: Path) -> AttrDict:
    hp = cfg.get("model_hparams") or {}
    effective_global_fallback = bool(cfg.get("effective_global_attention_fallback", False))
    d_model = int(hp.get("d_model", 256))
    map_d_model = int(hp.get("map_d_model", d_model))
    heads = int(hp.get("num_attn_head", 8))
    model = {
        "CONTEXT_ENCODER": {
            "NAME": "MTREncoder",
            "NUM_OF_ATTN_NEIGHBORS": int(hp.get("num_attn_neighbors", 16)),
            "NUM_INPUT_ATTR_AGENT": int(ds.builder.agent_attr_dim),
            "NUM_INPUT_ATTR_MAP": 9,
            "NUM_CHANNEL_IN_MLP_AGENT": int(hp.get("num_channel_agent", 256)),
            "NUM_CHANNEL_IN_MLP_MAP": int(hp.get("num_channel_map", 64)),
            "NUM_LAYER_IN_MLP_AGENT": int(hp.get("num_layer_mlp_agent", 3)),
            "NUM_LAYER_IN_MLP_MAP": int(hp.get("num_layer_mlp_map", 5)),
            "NUM_LAYER_IN_PRE_MLP_MAP": int(hp.get("num_layer_pre_mlp_map", 3)),
            "D_MODEL": d_model,
            "NUM_ATTN_LAYERS": int(hp.get("num_attn_layers", 6)),
            "NUM_ATTN_HEAD": heads,
            "DROPOUT_OF_ATTN": float(hp.get("dropout", 0.1)),
            "USE_LOCAL_ATTN": bool(hp.get("use_local_attn", True)) and not effective_global_fallback,
        },
        "MOTION_DECODER": {
            "NAME": "MTRDecoder",
            "OBJECT_TYPE": ["TYPE_VEHICLE"],
            "CENTER_OFFSET_OF_MAP": [30.0, 0.0],
            "NUM_FUTURE_FRAMES": int(ds.builder.future_len),
            "NUM_MOTION_MODES": int(hp.get("num_motion_modes", 6)),
            "INTENTION_POINTS_FILE": str(intention_file),
            "D_MODEL": int(hp.get("decoder_d_model", d_model * 2)),
            "NUM_DECODER_LAYERS": int(hp.get("num_decoder_layers", 6)),
            "NUM_ATTN_HEAD": heads,
            "MAP_D_MODEL": map_d_model,
            "DROPOUT_OF_ATTN": float(hp.get("dropout", 0.1)),
            "NUM_BASE_MAP_POLYLINES": int(hp.get("num_base_map_polylines", 64)),
            "NUM_WAYPOINT_MAP_POLYLINES": int(hp.get("num_waypoint_map_polylines", 32)),
            "LOSS_WEIGHTS": {"cls": 1.0, "reg": 1.0, "vel": 0.5},
            "NMS_DIST_THRESH": float(hp.get("nms_dist_thresh", 2.5)),
        },
    }
    return to_attrdict(model)


def move_batch_to_device(batch: dict[str, Any], device) -> dict[str, Any]:
    for key, value in batch["input_dict"].items():
        if hasattr(value, "to"):
            batch["input_dict"][key] = value.to(device, non_blocking=True)
    return batch


def prediction_tensors(batch_dict: dict[str, Any]):
    import torch

    pred_scores = batch_dict["pred_scores"]
    pred_trajs = batch_dict["pred_trajs"][:, :, :, 0:2]
    best = pred_scores.argmax(dim=-1)
    pred = pred_trajs[torch.arange(pred_trajs.size(0), device=pred_trajs.device), best]
    target = batch_dict["input_dict"]["center_gt_trajs"][:, :, 0:2].type_as(pred)
    all_modes = pred_trajs.permute(0, 2, 1, 3).contiguous()
    return pred, target, all_modes


def evaluate_epoch(model, loader, device, metric_dt: float, eval_hz: float) -> dict[str, Any]:
    from adapters.mtp_go.metrics import MetricAccumulator

    model.eval()
    acc = MetricAccumulator(dt=metric_dt, hz=eval_hz)
    import torch

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            out = model(batch)
            pred, target, all_modes = prediction_tensors(out)
            acc.update(pred, target, all_modes=all_modes)
    return acc.result()


def make_loader(ds: NeighFormerMTRDataset, cfg: dict[str, Any], shuffle: bool):
    import torch

    return torch.utils.data.DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]),
        persistent_workers=bool(cfg["persistent_workers"]) and int(cfg["num_workers"]) > 0,
        collate_fn=ds.collate_batch,
        drop_last=False,
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
    builder_kwargs = build_builder_kwargs(cfg)
    processed_dir = resolve_path(cfg["processed_dir"])
    out_root = processed_root(processed_dir, cfg["dataset"], cfg["feature_mode"])

    if cfg["mode"] == "preprocess":
        builder = NeighFormerMTRDataset(data_path, train_idx[:1], cfg["dataset"], cfg["feature_mode"], "train", builder_kwargs).builder
        report = {"processed_root": str(out_root), "splits": {}}
        build_intention_points_from_data(
            data_path,
            np.load(split_indices_path(data_root, cfg["dataset"], "train")),
            out_root / "intention_points.pkl",
            num_modes=int((cfg.get("model_hparams") or {}).get("num_motion_modes", 6)),
        )
        for split in cfg.get("preprocess_splits", ["train", "val", "test"]):
            indices = np.load(split_indices_path(data_root, cfg["dataset"], split))
            if split == "train" and cfg.get("max_train_samples") is not None:
                indices = indices[: int(cfg["max_train_samples"])]
            if split != "train" and cfg.get("max_eval_samples") is not None:
                indices = indices[: int(cfg["max_eval_samples"])]
            builder.split = split
            report["splits"][split] = save_processed_split(
                builder,
                indices,
                out_root,
                split,
                shard_size=int(cfg["preprocess_shard_size"]),
                overwrite=True,
            )
        print(json.dumps(report, indent=2))
        return 0

    train_ds = NeighFormerMTRDataset(
        data_path,
        train_idx,
        cfg["dataset"],
        cfg["feature_mode"],
        "train",
        builder_kwargs,
        processed_dir=processed_dir,
        reuse_processed=bool(cfg["reuse_processed"]),
    )
    val_ds = NeighFormerMTRDataset(
        data_path,
        val_idx,
        cfg["dataset"],
        cfg["feature_mode"],
        "val",
        builder_kwargs,
        processed_dir=processed_dir,
        reuse_processed=bool(cfg["reuse_processed"]),
    )

    output_dir = format_path_template(cfg["output_dir"], cfg)
    ckpt_dir = format_path_template(cfg["ckpt_dir"], cfg) / cfg["exp_tag"]
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data_report = {"train": train_ds.describe(), "val": val_ds.describe(), "channels": train_ds.channel_stats()}
    (output_dir / "data_report.json").write_text(json.dumps(data_report, indent=2), encoding="utf-8")
    if cfg["mode"] == "check-data":
        print(json.dumps(data_report, indent=2))
        return 0

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("MTR++ training requires PyTorch. check-data/preprocess can run without it.") from exc

    force_global = bool(cfg.get("global_attention_fallback", False))
    MotionTransformer, mtr_global_cfg, upstream_dir = import_motion_transformer(
        cfg["upstream_dir"],
        global_attention_fallback=force_global,
    )
    requested_device = str(cfg["device"]).lower()
    use_cuda = requested_device in {"auto", "gpu", "cuda"} and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    effective_global_fallback = bool(force_global or using_cuda_op_stubs())
    cfg["effective_global_attention_fallback"] = effective_global_fallback
    if cfg["mode"] == "full" and effective_global_fallback and not force_global:
        print(
            "[WARN] MTR++ CUDA ops (knn_cuda, attention_cuda) are not built; "
            "continuing with the adapter's global-attention compatibility fallback. "
            "For the original local-attention path, run: bash scripts/build_mtrpp_cuda_ops.sh",
            flush=True,
        )
    intention_file = out_root / "intention_points.pkl"
    build_intention_points_from_data(
        data_path,
        np.load(split_indices_path(data_root, cfg["dataset"], "train")),
        intention_file,
        num_modes=int((cfg.get("model_hparams") or {}).get("num_motion_modes", 6)),
    )
    model_cfg = build_model_config(cfg, train_ds, intention_file)
    mtr_global_cfg.ROOT_DIR = EXPERIMENT_ROOT
    model = MotionTransformer(config=model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    start_epoch = 0
    if args.resume:
        resume = args.resume if args.resume.is_absolute() else resolve_path(args.resume)
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if ckpt.get("optimizer_state"):
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt.get("epoch", 0))

    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = make_loader(val_ds, cfg, shuffle=False)
    best_fde = float("inf")
    print("====== MTR++ Train ======", flush=True)
    print(f"upstream : {upstream_dir} ({upstream_commit(upstream_dir)})", flush=True)
    print(f"data     : {data_path}", flush=True)
    print(f"samples  : train={len(train_ds):,} val={len(val_ds):,}", flush=True)
    print(f"mode     : {cfg['mode']}  epochs={cfg['epochs']}  batch_size={cfg['batch_size']}", flush=True)
    print(
        f"device   : {device}  cuda_op_stubs={using_cuda_op_stubs()} "
        f"global_fallback={effective_global_fallback}",
        flush=True,
    )

    for epoch in range(start_epoch, int(cfg["epochs"])):
        model.train()
        total_loss = 0.0
        batches = 0
        for it, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, tb_dict, _ = model(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip_norm"]))
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
            if it == 1 or it % int(cfg["log_interval"]) == 0 or it == len(train_loader):
                print(
                    f"epoch={epoch + 1}/{cfg['epochs']} iter={it}/{len(train_loader)} "
                    f"loss={float(loss.detach()):.4f}",
                    flush=True,
                )
        metrics = evaluate_epoch(model, val_loader, device, float(cfg["dt"]), float(cfg["eval_hz"]))
        print(
            f"val epoch={epoch + 1}: ade={metrics.get('ade', float('nan')):.4f} "
            f"fde={metrics.get('fde', float('nan')):.4f} loss={total_loss / max(1, batches):.4f}",
            flush=True,
        )
        state = {
            "cfg": plain(cfg),
            "model_cfg": plain(model_cfg),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch + 1,
            "metrics": metrics,
            "upstream_commit": upstream_commit(upstream_dir),
        }
        torch.save(state, ckpt_dir / "last.pt")
        if float(metrics.get("fde", float("inf"))) <= best_fde:
            best_fde = float(metrics.get("fde", float("inf")))
            torch.save(state, ckpt_dir / "best.pt")
    print(f"[DONE] best.pt saved -> {ckpt_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
