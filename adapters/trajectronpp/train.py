#!/usr/bin/env python3
"""Train Trajectron++ on NeighFormer highD/exiD windows via Environment pickles."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
try:
    import yaml
except ImportError:
    yaml = None

ADAPTER_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ADAPTER_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from adapters.common import dataset_dir, feature_mode_names, split_indices_path, validate_dataset_spec, DatasetSpec  # noqa: E402
from adapters.trajectronpp.preprocess import (  # noqa: E402
    BASE_STATE,
    PRED_STATE,
    build_environment,
    state_spec,
    write_environment,
    write_report,
)
from adapters.trajectronpp.upstream import add_upstream_to_path, upstream_commit  # noqa: E402

DEFAULTS: dict[str, Any] = {
    "adapter": "trajectronpp",
    "dataset": "",
    "feature_mode": "",
    "exp_tag": "",
    "data_root": "data",
    "output_dir": "",
    "ckpt_dir": "",
    "upstream_dir": "../trajectronPP",
    "dt": 0.32,
    "eval_hz": 3.0,
    "seed": 42,
    "device": "auto",
    "epochs": 100,
    "batch_size": 64,
    "eval_batch_size": 64,
    "lr": 1.0e-3,
    "grad_clip_norm": 1.0,
    "preprocess_workers": 0,
    "offline_scene_graph": "yes",
    "dynamic_edges": "yes",
    "max_train_samples": None,
    "max_eval_samples": None,
    "k_eval": 1,
}


def parse_scalar(text: str) -> Any:
    text = text.strip()
    if not text:
        return ""
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    if text in {"null", "None", "~"}:
        return None
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [] if not inner else [parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def simple_yaml_load(text: str) -> dict[str, Any]:
    rows = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        rows.append((len(raw) - len(raw.lstrip(" ")), raw.strip()))

    def parse_block(i: int, indent: int) -> tuple[Any, int]:
        is_list = i < len(rows) and rows[i][0] == indent and rows[i][1].startswith("- ")
        out: Any = [] if is_list else {}
        while i < len(rows):
            row_indent, text_row = rows[i]
            if row_indent < indent:
                break
            if row_indent > indent:
                i += 1
                continue
            if is_list:
                out.append(parse_scalar(text_row[2:]))
                i += 1
                continue
            key, sep, value = text_row.partition(":")
            if not sep:
                i += 1
                continue
            key = key.strip()
            value = value.strip()
            if value:
                out[key] = parse_scalar(value)
                i += 1
            else:
                child, i = parse_block(i + 1, indent + 2)
                out[key] = child
        return out, i

    parsed, _ = parse_block(0, 0)
    return parsed if isinstance(parsed, dict) else {}


def yaml_load(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text) or {}
    return simple_yaml_load(text)


def yaml_dump(data: dict[str, Any]) -> str:
    if yaml is not None:
        return yaml.safe_dump(data, sort_keys=False)
    return json.dumps(data, indent=2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--mode", default="smoke", choices=["smoke", "full", "check-data"])
    p.add_argument("--dataset", choices=["highD", "exiD"])
    p.add_argument("--feature-mode", choices=["baseline", "dimI"])
    p.add_argument("--data-root", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--ckpt-dir", type=Path)
    p.add_argument("--exp-tag")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--eval-batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--device")
    p.add_argument("--num-workers", type=int)
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
    raw = yaml_load(path)
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
                        "n_workers": "preprocess_workers",
                    }.get(key, key)
                ] = value
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
    if args.mode in {"smoke", "check-data"} or args.check_data:
        for key, value in (cfg.get("smoke") or {}).items():
            cfg[{"train_samples": "max_train_samples", "eval_samples": "max_eval_samples"}.get(key, key)] = value
    for cli_name, cfg_name in (
        ("data_root", "data_root"),
        ("output_dir", "output_dir"),
        ("ckpt_dir", "ckpt_dir"),
        ("exp_tag", "exp_tag"),
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("eval_batch_size", "eval_batch_size"),
        ("lr", "lr"),
        ("seed", "seed"),
        ("device", "device"),
        ("num_workers", "preprocess_workers"),
        ("max_train_samples", "max_train_samples"),
        ("max_eval_samples", "max_eval_samples"),
        ("upstream_dir", "upstream_dir"),
    ):
        value = getattr(args, cli_name)
        if value is not None:
            cfg[cfg_name] = value
    cfg["mode"] = "check-data" if args.check_data else args.mode
    if not cfg["dataset"] or not cfg["feature_mode"]:
        raise SystemExit("dataset and feature_mode must be set by config or CLI")
    if not cfg["exp_tag"]:
        cfg["exp_tag"] = f"trajectronpp_{cfg['dataset']}_{cfg['feature_mode']}"
    if not cfg["output_dir"]:
        cfg["output_dir"] = "runs/trajectronpp/{dataset}/{feature_mode}"
    return cfg


def resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (EXPERIMENT_ROOT / p).resolve()


def format_path_template(value: str | Path, cfg: dict[str, Any]) -> Path:
    return resolve_path(str(value).format(dataset=cfg["dataset"], feature_mode=cfg["feature_mode"], exp_tag=cfg["exp_tag"]))


def subset_indices(indices: np.ndarray, limit: int | None) -> np.ndarray:
    return indices if limit is None else indices[: int(limit)]


def to_plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def trajectron_config(cfg: dict[str, Any]) -> dict[str, Any]:
    full_state = {"VEHICLE": state_spec(cfg["feature_mode"])}
    edge_features = feature_mode_names(cfg["feature_mode"])
    return {
        "feature_mode": cfg["feature_mode"],
        "batch_size": int(cfg["batch_size"]),
        "grad_clip": float(cfg["grad_clip_norm"]),
        "learning_rate_style": "exp",
        "learning_rate": float(cfg["lr"]),
        "min_learning_rate": 1.0e-5,
        "learning_decay_rate": 0.9999,
        "prediction_horizon": 15,
        "minimum_history_length": 5,
        "maximum_history_length": 5,
        "k": 1,
        "k_eval": int(cfg["k_eval"]),
        "kl_min": 0.07,
        "kl_weight": 100.0,
        "kl_weight_start": 0,
        "kl_decay_rate": 0.99995,
        "kl_crossover": 400,
        "kl_sigmoid_divisor": 4,
        "rnn_kwargs": {"dropout_keep_prob": 0.75},
        "MLP_dropout_keep_prob": 0.9,
        "enc_rnn_dim_edge": 32,
        "enc_rnn_dim_edge_influence": 32,
        "enc_rnn_dim_history": 32,
        "enc_rnn_dim_future": 32,
        "dec_rnn_dim": 128,
        "q_z_xy_MLP_dims": None,
        "p_z_x_MLP_dims": 32,
        "GMM_components": 1,
        "log_p_yt_xz_max": 6,
        "N": 1,
        "K": 25,
        "tau_init": 2.0,
        "tau_final": 0.05,
        "tau_decay_rate": 0.997,
        "use_z_logit_clipping": True,
        "z_logit_clip_start": 0.05,
        "z_logit_clip_final": 5.0,
        "z_logit_clip_crossover": 300,
        "z_logit_clip_divisor": 5,
        "dynamic": {
            "VEHICLE": {
                "name": "SingleIntegrator",
                "distribution": True,
                "limits": {},
            }
        },
        "state": full_state,
        "ego_state": {"VEHICLE": BASE_STATE},
        "pred_state": {"VEHICLE": PRED_STATE},
        "neighbor_rel_features": edge_features,
        "edge_extra_features": [],
        "edge_extra_feat_params": {},
        "log_histograms": False,
    }


def prepare_data(cfg: dict[str, Any], upstream_dir: Path, output_dir: Path) -> tuple[Path, dict[str, Any]]:
    data_root = resolve_path(cfg["data_root"])
    data_path = dataset_dir(data_root, cfg["dataset"])
    validate_dataset_spec(
        DatasetSpec(cfg["dataset"], cfg["feature_mode"], "train", data_path, split_indices_path(data_root, cfg["dataset"], "train")),
        require_split=True,
    )
    proc_dir = output_dir / "processed"
    reports = {}
    for split in ("train", "val", "test"):
        limit = cfg["max_train_samples"] if split == "train" else cfg["max_eval_samples"]
        indices = subset_indices(np.load(split_indices_path(data_root, cfg["dataset"], split)), limit)
        env, report = build_environment(data_path, indices, cfg["dataset"], cfg["feature_mode"], split, upstream_dir, dt=float(cfg["dt"]))
        write_environment(proc_dir / f"{split}_env.pkl", env)
        write_report(output_dir / f"{split}_data_report.json", report)
        reports[split] = report
    return proc_dir, reports


def latest_model_dir(log_dir: Path, exp_tag: str, before: set[Path]) -> Path | None:
    after = {p for p in log_dir.glob(f"models_*_{exp_tag}") if p.is_dir()}
    new_dirs = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    if new_dirs:
        return new_dirs[0]
    all_dirs = sorted(after, key=lambda p: p.stat().st_mtime, reverse=True)
    return all_dirs[0] if all_dirs else None


def environment_info(cfg: dict[str, Any], upstream_dir: Path) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda": torch.version.cuda,
        "upstream_dir": str(upstream_dir),
        "upstream_commit": upstream_commit(upstream_dir),
        "implementation": "original Trajectron++ model/training code with highD/exiD input adapter",
        "config": to_plain(cfg),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = apply_cli(load_config(args.config), args)
    upstream_dir = add_upstream_to_path(cfg["upstream_dir"])
    output_dir = format_path_template(cfg["output_dir"], cfg)
    ckpt_dir = format_path_template(cfg["ckpt_dir"], cfg) / cfg["exp_tag"] if cfg.get("ckpt_dir") else output_dir / "checkpoints"
    log_dir = output_dir / "upstream_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    proc_dir, reports = prepare_data(cfg, upstream_dir, output_dir)
    hyperparams = trajectron_config(cfg)
    conf_path = output_dir / "trajectron_config.json"
    conf_path.write_text(json.dumps(hyperparams, indent=2), encoding="utf-8")
    env = environment_info(cfg, upstream_dir)
    (output_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    command = " ".join(shlex.quote(a) for a in [sys.executable, *sys.argv])
    run_config = {
        "model": "trajectronpp",
        "command": command,
        "effective_config": to_plain(cfg),
        "upstream_config": hyperparams,
        "data": reports,
        "environment": env,
    }
    (output_dir / "run_config.yaml").write_text(yaml_dump(to_plain(run_config)), encoding="utf-8")
    (output_dir / "data_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")

    if cfg["mode"] == "check-data":
        print(json.dumps(reports, indent=2))
        return 0

    device = str(cfg["device"])
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    eval_device = device
    before = {p for p in log_dir.glob(f"models_*_{cfg['exp_tag']}") if p.is_dir()}
    cmd = [
        sys.executable,
        str(upstream_dir / "trajectron" / "train.py"),
        "--conf",
        str(conf_path),
        "--data_dir",
        str(proc_dir),
        "--train_data_dict",
        "train_env.pkl",
        "--eval_data_dict",
        "val_env.pkl",
        "--log_dir",
        str(log_dir),
        "--log_tag",
        f"_{cfg['exp_tag']}",
        "--device",
        device,
        "--eval_device",
        eval_device,
        "--train_epochs",
        str(int(cfg["epochs"])),
        "--batch_size",
        str(int(cfg["batch_size"])),
        "--eval_batch_size",
        str(int(cfg["eval_batch_size"])),
        "--preprocess_workers",
        str(max(1, int(cfg["preprocess_workers"])) if cfg["offline_scene_graph"] == "yes" else int(cfg["preprocess_workers"])),
        "--seed",
        str(int(cfg["seed"])),
        "--offline_scene_graph",
        str(cfg["offline_scene_graph"]),
        "--dynamic_edges",
        str(cfg["dynamic_edges"]),
        "--eval_every",
        str(int(cfg["epochs"]) + 1),
        "--save_every",
        "1",
    ]
    log_path = output_dir / "train.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(" ".join(shlex.quote(x) for x in cmd) + "\n\n")
        proc = subprocess.run(
            cmd,
            cwd=str(upstream_dir / "trajectron"),
            env={**os.environ, "PYTHONPATH": f"{upstream_dir}:{upstream_dir / 'trajectron'}:{os.environ.get('PYTHONPATH', '')}"},
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
        raise SystemExit(f"Upstream Trajectron++ training failed with exit code {proc.returncode}.\n{tail}")

    model_dir = latest_model_dir(log_dir, cfg["exp_tag"], before)
    if model_dir is None:
        raise SystemExit(f"Training finished but no upstream model directory was found in {log_dir}")
    model_iter = int(cfg["epochs"])
    upstream_ckpt = model_dir / f"model_registrar-{model_iter}.pt"
    if not upstream_ckpt.exists():
        candidates = sorted(model_dir.glob("model_registrar-*.pt"))
        if not candidates:
            raise SystemExit(f"No model_registrar checkpoint found under {model_dir}")
        upstream_ckpt = candidates[-1]
        model_iter = int(upstream_ckpt.stem.split("-")[-1])

    wrapper_ckpt = {
        "cfg": cfg,
        "adapter": "trajectronpp",
        "epoch": model_iter,
        "upstream_dir": str(upstream_dir),
        "upstream_model_dir": str(model_dir),
        "upstream_checkpoint": str(upstream_ckpt),
        "upstream_config": str(conf_path),
        "processed_dir": str(proc_dir),
    }
    torch.save(wrapper_ckpt, ckpt_dir / "best.pt")
    torch.save(wrapper_ckpt, ckpt_dir / "last.pt")
    metrics = {
        "model": "trajectronpp",
        "dataset": cfg["dataset"],
        "feature_mode": cfg["feature_mode"],
        "mode": cfg["mode"],
        "epoch": model_iter,
        "train_seconds": round(time.time() - started, 2),
        "checkpoint": str(ckpt_dir / "best.pt"),
        "upstream_checkpoint": str(upstream_ckpt),
    }
    (output_dir / "metrics.json").write_text(json.dumps(to_plain(metrics), indent=2), encoding="utf-8")
    print(f"[DONE] Trajectron++ training finished. checkpoint={ckpt_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
