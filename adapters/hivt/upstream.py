"""Helpers for importing the official HiVT checkout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]


def resolve_upstream_dir(path: str | Path | None = None) -> Path:
    upstream = Path(path) if path is not None else EXPERIMENT_ROOT / "external" / "hivt"
    upstream = upstream.expanduser()
    if not upstream.is_absolute():
        upstream = (EXPERIMENT_ROOT / upstream).resolve()
    if not upstream.exists():
        raise FileNotFoundError(
            f"HiVT upstream checkout not found: {upstream}\n"
            "Clone it with: git clone https://github.com/ZikangZhou/HiVT.git external/hivt"
        )
    return upstream


def _patch_torchmetrics_compat() -> None:
    try:
        import torchmetrics
    except ImportError:
        return
    metric = torchmetrics.Metric
    if getattr(metric, "_sota_hivt_compat", False):
        return
    original_init = metric.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.pop("compute_on_step", None)
        return original_init(self, *args, **kwargs)

    metric.__init__ = patched_init
    metric._sota_hivt_compat = True


def add_upstream_to_path(path: str | Path | None = None) -> Path:
    upstream = resolve_upstream_dir(path)
    _patch_torchmetrics_compat()
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    _patch_temporal_data_compat()
    _patch_temporal_encoder_layer_compat()
    _patch_hivt_forward_compat()
    return upstream


def _patch_temporal_data_compat() -> None:
    try:
        import torch
        from torch_geometric.data import Data
        from utils import TemporalData
    except ImportError:
        return
    if getattr(TemporalData, "_sota_hivt_compat", False):
        return

    def patched_inc(self, key, value, *args, **kwargs):
        if key == "lane_actor_index":
            return torch.tensor([[self["lane_vectors"].size(0)], [self.num_nodes]])
        if key in {"agent_index", "av_index"}:
            return self.num_nodes
        if key in {"sample_index", "recording_id", "track_id", "frame_id", "seq_id"}:
            return 0
        return Data.__inc__(self, key, value, *args, **kwargs)

    TemporalData.__inc__ = patched_inc
    TemporalData._sota_hivt_compat = True


def _patch_temporal_encoder_layer_compat() -> None:
    try:
        from models.local_encoder import TemporalEncoderLayer
    except ImportError:
        return
    if getattr(TemporalEncoderLayer, "_sota_hivt_compat", False):
        return
    original_forward = TemporalEncoderLayer.forward

    def patched_forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        return original_forward(self, src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

    TemporalEncoderLayer.forward = patched_forward
    TemporalEncoderLayer._sota_hivt_compat = True


def _patch_hivt_forward_compat() -> None:
    try:
        import torch
        from models.hivt import HiVT
    except ImportError:
        return
    if getattr(HiVT, "_sota_hivt_compat", False):
        return

    original_forward = HiVT.forward

    def patched_forward(self, data):
        if self.rotate:
            return original_forward(self, data)
        # PyG 2.x drops None attributes assigned through __setitem__; the
        # official LocalEncoder still indexes data['rotate_mat'].
        data._store._mapping["rotate_mat"] = None
        local_embed = self.local_encoder(data=data)
        global_embed = self.global_interactor(data=data, local_embed=local_embed)
        y_hat, pi = self.decoder(local_embed=local_embed, global_embed=global_embed)
        return y_hat, pi

    HiVT.forward = patched_forward
    HiVT._sota_hivt_compat = True


def upstream_commit(path: str | Path | None = None) -> str | None:
    upstream = resolve_upstream_dir(path)
    try:
        return subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
