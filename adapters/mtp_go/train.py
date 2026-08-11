#!/usr/bin/env python3
"""MTP-GO adapter for the shared highD/exiD baseline-vs-dimI experiment matrix.

Runs the upstream MTP-GO encoder/decoder (imported from ``external/mtp_go``)
on the canonical NeighFormer npy data.

    python adapters/mtp_go/train.py \\
        --config configs/models/mtp_go.yaml \\
        --dataset highD --feature-mode baseline \\
        --data-root /home/gahyun/neighformer/data \\
        --output-dir runs/mtp_go/highD/baseline

The default run mode is ``smoke``: a few epochs over a small subset. Pass
``--mode full`` to start real training.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import shlex
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

ADAPTER_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ADAPTER_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from adapters.common import (  # noqa: E402
    DatasetSpec,
    dataset_dir,
    feature_mode_indices,
    feature_mode_names,
    split_indices_path,
    validate_dataset_spec,
)
from adapters.mtp_go.dataset import (  # noqa: E402
    EGO_EXTRA_FILL,
    NeighFormerGraphDataset,
    estimate_dt,
)
from adapters.mtp_go.lit_module import evaluate, make_lit_module_class  # noqa: E402
from adapters.mtp_go.upstream import (  # noqa: E402
    ROTATIONAL_MOTION_MODELS,
    add_upstream_to_path,
    build_motion_model,
    resolve_upstream_dir,
    upstream_commit,
)

LOGGER = logging.getLogger("mtp_go_adapter")

# Mirrors external/mtp_go/config.py DEFAULTS, with dt/runtime keys added.
DEFAULTS: dict[str, Any] = {
    # trainer
    "epochs": 100,
    "batch_size": 128,
    "lr": 1e-3,
    "clip": 5.0,
    "teacher_forcing": 0.2,
    "log_interval": 100,
    # model
    "u1_lim": 10.0,
    "u2_lim": 10.0,
    "ode_solver": "rk4",
    "n_mixtures": 8,
    "motion_model": "2Xnode",
    "hidden_size": 64,
    "gnn_layer": "natt",
    "n_gnn_layers": 1,
    "n_ode_hidden": 16,
    "n_ode_layers": 1,
    "n_heads": 1,
    "init_static": False,
    "n_ode_static": False,
    "use_edge_features": True,
    # data / runtime
    "dt": 0.32,
    "seed": 42,
    "n_workers": 4,
    "accelerator": "auto",
    # smoke run limits
    "smoke_epochs": 4,
    "smoke_schedule_epochs": 16,
    "smoke_train_samples": 2048,
    "smoke_eval_samples": 1024,
    "smoke_batch_size": 32,
}

# Upstream's loss schedule uses epochs // 2 (teacher forcing), epochs // 4 (warm-up)
# and epochs // 8 (EWTA). Below 16 the schedule misbehaves: wta_epochs == 1 fails
# base_mdn's `wta_epochs > 1` guard, so the pure-EWTA phase is skipped and the
# warm-up weight starts at 2.0, which gives the NLL term a negative coefficient.
# Below 2 epochs it divides by zero outright.
MIN_SCHEDULE_EPOCHS = 16

SPLITS = ("train", "val", "test")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--dataset", required=True, choices=["highD", "exiD"])
    p.add_argument("--feature-mode", required=True, choices=["baseline", "dimI"])
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)

    p.add_argument("--mode", choices=["smoke", "full"], default="smoke",
                   help="smoke (default): a few epochs on a small subset. full: real training.")
    p.add_argument("--full", action="store_true", help="Alias for --mode full.")
    p.add_argument("--check-data", action="store_true",
                   help="Load arrays, build one scene graph, write a data report, then exit.")

    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--accelerator", choices=["auto", "gpu", "cpu"])
    p.add_argument("--max-train-samples", type=int)
    p.add_argument("--max-eval-samples", type=int)
    p.add_argument("--upstream-dir", type=Path, help="Override the MTP-GO checkout location.")
    p.add_argument("--split-fallback", choices=["none", "sequential"], default="none",
                   help="What to do when splits/*_indices.npy are missing. "
                        "'sequential' builds a deterministic 70/15/15 index split "
                        "and records the fact in run_config.yaml.")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoints/last.ckpt if present.")
    return p.parse_args(argv)


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(output_dir / "train.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOGGER.addHandler(fh)
    LOGGER.addHandler(sh)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = dict(DEFAULTS)
    for section in ("training", "model_hparams", "runtime", "smoke"):
        block = raw.get(section)
        if isinstance(block, dict):
            for k, v in block.items():
                key = f"smoke_{k}" if section == "smoke" else k
                cfg[key] = v
    for k, v in raw.items():
        if k in cfg:
            cfg[k] = v
    cfg["_raw_config"] = raw
    return cfg


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mode = "full" if args.full else args.mode
    cfg["mode"] = mode
    if mode == "smoke":
        cfg["epochs"] = cfg["smoke_epochs"]
        cfg["batch_size"] = cfg["smoke_batch_size"]
        cfg["max_train_samples"] = cfg["smoke_train_samples"]
        cfg["max_eval_samples"] = cfg["smoke_eval_samples"]
    else:
        cfg["max_train_samples"] = None
        cfg["max_eval_samples"] = None

    for cli_key, cfg_key in (
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("lr", "lr"),
        ("seed", "seed"),
        ("num_workers", "n_workers"),
        ("accelerator", "accelerator"),
        ("max_train_samples", "max_train_samples"),
        ("max_eval_samples", "max_eval_samples"),
    ):
        value = getattr(args, cli_key, None)
        if value is not None:
            cfg[cfg_key] = value

    # `epochs` is how long the Trainer runs; `schedule_epochs` is the horizon
    # upstream's EWTA -> EWTA+NLL -> NLL and teacher-forcing schedules are defined
    # over. A smoke run executes the first epochs of a realistic schedule instead
    # of collapsing it into a single epoch.
    cfg["schedule_epochs"] = (
        cfg["smoke_schedule_epochs"] if mode == "smoke" else cfg["epochs"]
    )
    if int(cfg["schedule_epochs"]) < MIN_SCHEDULE_EPOCHS:
        raise SystemExit(
            f"schedule_epochs={cfg['schedule_epochs']} is too small; upstream's loss "
            f"schedule needs at least {MIN_SCHEDULE_EPOCHS} epochs."
        )
    return cfg


def resolve_splits(
    data_root: Path,
    dataset: str,
    n_samples: int,
    fallback: str,
    probe_only: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    paths = {s: split_indices_path(data_root, dataset, s) for s in SPLITS}
    missing = [str(p) for s, p in paths.items() if not p.exists()]

    if missing and probe_only:
        # --check-data only verifies that the arrays load and convert. Report the
        # missing split files instead of inventing indices.
        order = np.arange(min(n_samples, 4096), dtype=np.int64)
        info = {
            "source": "unresolved",
            "missing_paths": missing,
            "note": "--check-data inspected the leading samples; no split was invented.",
        }
        return {s: order for s in SPLITS}, info

    if not missing:
        indices = {s: np.load(p).astype(np.int64) for s, p in paths.items()}
        info = {
            "source": "neighformer_split_files",
            "paths": {s: str(p) for s, p in paths.items()},
            "sizes": {s: int(v.size) for s, v in indices.items()},
        }
        return indices, info

    if fallback != "sequential":
        raise SystemExit(
            "Missing split index files:\n  "
            + "\n  ".join(missing)
            + f"\n\nGenerate them first (e.g. {data_root / dataset / 'split.py'}), or rerun with "
            "--split-fallback sequential to use a deterministic 70/15/15 sequential split. "
            "The fallback is recorded in run_config.yaml; it is NOT the canonical split."
        )

    order = np.arange(n_samples, dtype=np.int64)
    n_train = int(n_samples * 0.70)
    n_val = int(n_samples * 0.15)
    indices = {
        "train": order[:n_train],
        "val": order[n_train:n_train + n_val],
        "test": order[n_train + n_val:],
    }
    info = {
        "source": "adapter_sequential_fallback",
        "warning": "Canonical split files were missing; a deterministic 70/15/15 "
                   "sequential split was generated by the adapter. Results are not "
                   "comparable to runs that use the canonical splits.",
        "missing_paths": missing,
        "sizes": {s: int(v.size) for s, v in indices.items()},
    }
    return indices, info


def subsample(indices: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or indices.size <= limit:
        return indices
    rng = np.random.default_rng(seed)
    picked = rng.choice(indices.size, size=limit, replace=False)
    return np.sort(indices[picked])


def make_epoch_logger() -> Any:
    """Log train_loss / val_ade / val_fde / val_nll once per epoch into train.log."""
    from lightning.pytorch.callbacks import Callback

    class EpochLogger(Callback):
        # on_train_epoch_end runs after the validation loop, so both the epoch's
        # train_loss and its val metrics are already in callback_metrics.
        def on_train_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            m = trainer.callback_metrics

            def get(key: str) -> float:
                v = m.get(key)
                return float(v) if v is not None else float("nan")

            LOGGER.info(
                "epoch %3d/%-3d  loss=%.4f  val_ade=%.4f  val_fde=%.4f  val_nll=%.4g",
                trainer.current_epoch + 1,
                trainer.max_epochs,
                get("train_loss"),
                get("val_ade"),
                get("val_fde"),
                get("val_nll"),
            )

    return EpochLogger()


def to_plain(obj: Any) -> Any:
    """Coerce numpy scalars, Paths and str/int subclasses into YAML/JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, str):
        return str(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    if obj is None:
        return None
    return str(obj)


def environment_info(upstream_dir: Path) -> dict[str, Any]:
    import torch
    import torch_geometric
    import lightning

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "lightning": lightning.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "upstream_dir": str(upstream_dir),
        "upstream_url": "https://github.com/westny/mtp-go.git",
        "upstream_commit": upstream_commit(upstream_dir),
    }
    try:
        import torchdiffeq

        info["torchdiffeq"] = torchdiffeq.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def feature_mapping(feature_mode: str) -> dict[str, Any]:
    nb_idx = feature_mode_indices(feature_mode)
    nb_names = feature_mode_names(feature_mode)
    node_names = ["x|dx", "y|dy", "vx|dvx", "vy|dvy", "ax|dax", "ay|day"] + nb_names[6:]
    return {
        "ego_node": {
            "source": "x_ego.npy",
            "channels": ["x", "y", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"],
            "frame": "relative to ego position at the last history step",
            "extra_channel_fill": EGO_EXTRA_FILL if len(nb_idx) > 6 else None,
        },
        "neighbor_nodes": {
            "source": "x_nb.npy",
            "selected_indices": nb_idx,
            "channels": nb_names,
            "frame": "ego-relative (dx, dy, dvx, ...)",
        },
        "node_feature_layout": node_names,
        "node_feature_dim": len(nb_idx),
        "edges": "per history step, fully connected + self loops over present nodes; "
                 "edge feature = Euclidean distance in the ego-relative frame",
        "future_edges": "last observed history graph reused for every future step "
                        "(neighbor futures are not part of the NeighFormer schema)",
        "targets": "ego node only: y.npy (+ y_vel.npy, y_acc.npy); "
                   "tar_real_mask is False for every neighbor node",
    }


def build_datasets(
    data_dir: Path,
    indices: dict[str, np.ndarray],
    feature_mode: str,
    cfg: dict[str, Any],
) -> dict[str, NeighFormerGraphDataset]:
    datasets: dict[str, NeighFormerGraphDataset] = {}
    for split in SPLITS:
        limit = cfg["max_train_samples"] if split == "train" else cfg["max_eval_samples"]
        idx = subsample(indices[split], limit, cfg["seed"])
        datasets[split] = NeighFormerGraphDataset(data_dir, idx, feature_mode, split=split)
    return datasets


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir: Path = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (EXPERIMENT_ROOT / output_dir).resolve()
    setup_logging(output_dir)

    command = " ".join(shlex.quote(a) for a in [sys.executable, *sys.argv])
    cfg = apply_overrides(load_config(args.config), args)

    data_root: Path = args.data_root.resolve()
    data_dir = dataset_dir(data_root, args.dataset)
    spec = DatasetSpec(
        dataset=args.dataset,
        feature_mode=args.feature_mode,
        split="train",
        data_dir=data_dir,
        split_indices_path=split_indices_path(data_root, args.dataset, "train"),
    )
    validate_dataset_spec(spec)

    upstream_dir = resolve_upstream_dir(args.upstream_dir)
    add_upstream_to_path(upstream_dir)

    LOGGER.info("mode          : %s", cfg["mode"])
    LOGGER.info("dataset       : %s (%s)", args.dataset, data_dir)
    LOGGER.info("feature mode  : %s -> %d node channels", args.feature_mode,
                len(feature_mode_indices(args.feature_mode)))
    LOGGER.info("upstream      : %s", upstream_dir)
    LOGGER.info("output dir    : %s", output_dir)

    # ---------------------------------------------------------------- dt check
    dt = float(cfg["dt"])
    dt_est = estimate_dt(data_dir)
    if dt_est is not None:
        LOGGER.info("dt configured=%.4fs, estimated from data=%.4fs", dt, dt_est)
        if abs(dt_est - dt) / dt > 0.15:
            LOGGER.warning(
                "Configured dt (%.4fs) differs from the value estimated from x_ego "
                "(%.4fs) by more than 15%%. The motion model integrates with the "
                "configured dt; set `dt` in the config to match the data.", dt, dt_est
            )

    n_total = int(np.load(data_dir / "x_ego.npy", mmap_mode="r").shape[0])
    indices, split_info = resolve_splits(
        data_root, args.dataset, n_total, args.split_fallback, probe_only=args.check_data
    )
    if "warning" in split_info:
        LOGGER.warning(split_info["warning"])
    if split_info["source"] == "unresolved":
        LOGGER.warning("Split index files are missing: %s", ", ".join(split_info["missing_paths"]))

    datasets = build_datasets(data_dir, indices, args.feature_mode, cfg)
    for split, ds in datasets.items():
        LOGGER.info("split %-5s : %d samples", split, ds.len())

    probe = datasets["train"][0] if datasets["train"].len() else None
    data_report = {
        "dataset": args.dataset,
        "feature_mode": args.feature_mode,
        "data_dir": str(data_dir),
        "splits": split_info,
        "feature_mapping": feature_mapping(args.feature_mode),
        "arrays": {s: datasets[s].describe() for s in SPLITS},
        "dt_configured": dt,
        "dt_estimated": dt_est,
        "node_channel_stats": datasets["train"].channel_stats(),
    }
    if probe is not None:
        data_report["example_graph"] = {
            "num_nodes": int(probe.num_nodes),
            "x": list(probe.x.shape),
            "y": list(probe.y.shape),
            "history_graphs": len(probe.edge_index),
            "future_graphs": len(probe.tar_edge_index),
            "edges_last_history_step": int(probe.edge_index[-1].shape[1]),
            "ego_target_steps": int(probe.tar_real_mask[0, :, 0].sum()),
            "neighbor_target_steps": int(probe.tar_real_mask[1:, :, 0].sum()),
        }
    (output_dir / "data_report.json").write_text(
        json.dumps(to_plain(data_report), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.check_data:
        LOGGER.info("--check-data: wrote %s", output_dir / "data_report.json")
        LOGGER.info("%s", json.dumps(data_report.get("example_graph", {}), indent=2))
        return 0

    # ------------------------------------------------------------------ model
    import torch
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint
    from torch_geometric.loader import DataLoader

    from base_mdn import LitEncoderDecoder  # upstream
    from models.gru_gnn import GRUGNNDecoder, GRUGNNEncoder  # upstream

    seed_everything(int(cfg["seed"]), workers=True)

    if cfg["motion_model"] in ROTATIONAL_MOTION_MODELS:
        raise SystemExit(
            f"motion_model={cfg['motion_model']} needs heading/vehicle-dimension inputs that "
            "the NeighFormer schema does not provide. Use one of: 2Xnode, neuralode, "
            "1Xint, 2Xint, 3Xint."
        )

    hp = SimpleNamespace(**{k: v for k, v in cfg.items() if not k.startswith("_")})
    hp.dataset = f"{args.dataset}-{args.feature_mode}"
    hp.epochs = int(cfg["schedule_epochs"])  # loss/teacher-forcing schedule horizon

    n_features = len(feature_mode_indices(args.feature_mode))
    static_f_dim = 2 * int(bool(cfg["n_ode_static"]))
    motion_model = build_motion_model(hp, dt, static_f_dim)
    max_length = datasets["train"].history_len + 1  # encoder emits T_h + 1 states

    encoder = GRUGNNEncoder(
        input_size=n_features,
        hidden_size=cfg["hidden_size"],
        n_mixtures=motion_model.mixtures,
        n_layers=cfg["n_gnn_layers"],
        gnn_layer=cfg["gnn_layer"],
        n_heads=cfg["n_heads"],
        static_f_dim=static_f_dim,
        init_static=cfg["init_static"],
        use_edge_features=cfg["use_edge_features"],
    )
    decoder = GRUGNNDecoder(
        motion_model,
        hidden_size=encoder.hidden_size,
        max_length=max_length,
        n_layers=cfg["n_gnn_layers"],
        n_heads=cfg["n_heads"],
        static_f_dim=static_f_dim,
        gnn_layer=cfg["gnn_layer"],
        init_static=cfg["init_static"],
    )
    model = make_lit_module_class(LitEncoderDecoder)(encoder, decoder, hp)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info("model         : %s / %s, %d trainable params",
                cfg["motion_model"], cfg["gnn_layer"], n_params)
    LOGGER.info("input size    : %d node channels, n_states=%d, mixtures=%d",
                n_features, motion_model.n_states, motion_model.mixtures)

    n_workers = int(cfg["n_workers"])
    loader_kwargs = dict(num_workers=n_workers, pin_memory=torch.cuda.is_available())
    if n_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(datasets["train"], batch_size=int(cfg["batch_size"]),
                              shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(datasets["val"], batch_size=int(cfg["batch_size"]),
                            shuffle=False, **loader_kwargs)
    test_loader = DataLoader(datasets["test"], batch_size=int(cfg["batch_size"]),
                             shuffle=False, **loader_kwargs)

    accelerator = cfg["accelerator"]
    if accelerator == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    ckpt_dir = output_dir / "checkpoints"
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir), filename="best", monitor="val_ade", mode="min",
        save_top_k=1, save_last=True,
    )
    resume_path = ckpt_dir / "last.ckpt"
    trainer = Trainer(
        max_epochs=int(cfg["epochs"]),
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=cfg["clip"],
        log_every_n_steps=int(cfg["log_interval"]),
        callbacks=[checkpoint_cb, make_epoch_logger()],
        logger=False,
        enable_checkpointing=True,
        default_root_dir=str(output_dir),
    )

    started = time.time()
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=str(resume_path) if (args.resume and resume_path.exists()) else None,
    )
    train_seconds = time.time() - started
    LOGGER.info("training finished in %.1fs", train_seconds)

    # ------------------------------------------------------------- evaluation
    device = torch.device("cuda" if accelerator == "gpu" else "cpu")
    metrics = {
        "model": "mtp_go",
        "dataset": args.dataset,
        "feature_mode": args.feature_mode,
        "mode": cfg["mode"],
        "epochs_run": int(trainer.current_epoch),
        "train_seconds": round(train_seconds, 2),
        "train_loss": float(trainer.callback_metrics.get("train_loss", float("nan"))),
        "node_feature_channels": n_features,
        "num_parameters": int(n_params),
        "val": evaluate(model, val_loader, device, dt),
        "test": evaluate(model, test_loader, device, dt),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(to_plain(metrics), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("val  ADE=%.4f FDE=%.4f RMSE=%.4f",
                metrics["val"].get("ADE", float("nan")),
                metrics["val"].get("FDE", float("nan")),
                metrics["val"].get("RMSE", float("nan")))
    LOGGER.info("test ADE=%.4f FDE=%.4f RMSE=%.4f",
                metrics["test"].get("ADE", float("nan")),
                metrics["test"].get("FDE", float("nan")),
                metrics["test"].get("RMSE", float("nan")))

    # ---------------------------------------------------------------- configs
    env = environment_info(upstream_dir)
    run_config = {
        "model": "mtp_go",
        "command": command,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cli": {
            "config": str(args.config),
            "dataset": args.dataset,
            "feature_mode": args.feature_mode,
            "data_root": str(data_root),
            "output_dir": str(output_dir),
            "mode": cfg["mode"],
            "split_fallback": args.split_fallback,
        },
        "effective_config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "source_config": cfg["_raw_config"],
        "data": data_report,
        "model_summary": {
            "motion_model": cfg["motion_model"],
            "n_states": int(motion_model.n_states),
            "n_mixtures": int(motion_model.mixtures),
            "input_size": n_features,
            "max_length": max_length,
            "static_f_dim": static_f_dim,
            "num_parameters": int(n_params),
        },
        "environment": env,
        "artifacts": {
            "checkpoint": str(ckpt_dir / "best.ckpt"),
            "metrics": str(output_dir / "metrics.json"),
            "log": str(output_dir / "train.log"),
            "data_report": str(output_dir / "data_report.json"),
        },
    }
    with (output_dir / "run_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(to_plain(run_config), f, sort_keys=False, allow_unicode=True)
    (output_dir / "environment.json").write_text(
        json.dumps(to_plain(env), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("artifacts written to %s", output_dir)
    if cfg["mode"] == "smoke":
        LOGGER.info("This was a SMOKE run. Use --mode full for real training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
