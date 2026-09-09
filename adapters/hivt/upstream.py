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
    _patch_hivt_validation_compat()
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


def _patch_hivt_validation_compat() -> None:
    try:
        import torch
        import torch.nn.functional as F
        from models.hivt import HiVT
    except ImportError:
        return
    if getattr(HiVT, "_sota_hivt_multi_agent_metrics", False):
        return

    def _mode_errors(y_hat, target, valid_mask):
        dist = torch.norm(y_hat[:, :, :, :2] - target.unsqueeze(0), p=2, dim=-1)
        valid = valid_mask.unsqueeze(0).float()
        counts = valid_mask.float().sum(dim=-1).clamp_min(1.0)
        ade = (dist * valid).sum(dim=-1) / counts.unsqueeze(0)
        last_valid = valid_mask.long().sum(dim=-1) - 1
        mode_idx = torch.arange(y_hat.shape[0], device=y_hat.device).unsqueeze(1)
        node_idx = torch.arange(y_hat.shape[1], device=y_hat.device).unsqueeze(0)
        fde = dist[mode_idx, node_idx, last_valid.unsqueeze(0)]
        return ade, fde

    def patched_validation_step(self, data, batch_idx):
        y_hat, pi = self(data)
        reg_mask = ~data["padding_mask"][:, self.historical_steps:]
        valid_steps = reg_mask.sum(dim=-1)
        cls_mask = valid_steps > 0
        if not bool(cls_mask.any()):
            return None

        y_hat_t = y_hat[:, cls_mask]
        y_t = data.y[cls_mask]
        mask_t = reg_mask[cls_mask]
        ade_modes, fde_modes = _mode_errors(y_hat_t, y_t, mask_t)
        best_mode = ade_modes.argmin(dim=0)
        y_hat_best = y_hat_t[best_mode, torch.arange(y_hat_t.shape[1], device=y_hat.device)]
        reg_loss = self.reg_loss(y_hat_best[mask_t], y_t[mask_t])
        soft_target = F.softmax(-ade_modes.t(), dim=-1).detach()
        cls_loss = self.cls_loss(pi[cls_mask], soft_target)
        loss = reg_loss + cls_loss

        min_ade = ade_modes.min(dim=0).values.mean()
        min_fde = fde_modes.min(dim=0).values.mean()
        miss = (fde_modes.min(dim=0).values > 2.0).float().mean()
        batch_size = int(cls_mask.sum().item())
        self.log("val_reg_loss", reg_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("val_minADE", min_ade, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("val_minFDE", min_fde, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("val_minMR", miss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    HiVT.validation_step = patched_validation_step
    HiVT._sota_hivt_multi_agent_metrics = True


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
