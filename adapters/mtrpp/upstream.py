"""Helpers for importing the official MTR checkout."""

from __future__ import annotations

import subprocess
import sys
import types
import importlib
from pathlib import Path
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
_STUBBED_CUDA_OPS = False


def resolve_upstream_dir(path: str | Path | None = None) -> Path:
    upstream = Path(path) if path is not None else EXPERIMENT_ROOT / "external" / "mtrpp"
    upstream = upstream.expanduser()
    if not upstream.is_absolute():
        upstream = (EXPERIMENT_ROOT / upstream).resolve()
    if not upstream.exists():
        raise FileNotFoundError(
            f"MTR upstream checkout not found: {upstream}\n"
            "Clone it with: git clone https://github.com/sshaoshuai/MTR.git external/mtrpp"
        )
    return upstream


def _install_easydict_fallback() -> None:
    try:
        import easydict  # noqa: F401
        return
    except ImportError:
        pass

    class EasyDict(dict):
        def __getattr__(self, name: str) -> Any:
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name: str, value: Any) -> None:
            self[name] = value

        def __delattr__(self, name: str) -> None:
            del self[name]

    module = types.ModuleType("easydict")
    module.EasyDict = EasyDict
    sys.modules["easydict"] = module


def _install_cuda_op_stubs() -> None:
    """Let MTR import on machines where its optional CUDA extensions are absent.

    The stubs are only a bridge to import the official modules. Training/eval can
    still use the original CUDA kernels when they are installed; otherwise
    ``patch_mtr_runtime`` switches the affected local-attention paths to the
    upstream global-attention implementation for smoke/debug runs.
    """
    global _STUBBED_CUDA_OPS

    for name in ("mtr.ops.attention.attention_cuda", "mtr.ops.knn.knn_cuda"):
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
        except ImportError:
            package = name.rsplit(".", 1)[0]
            for key in list(sys.modules):
                if key == package or key.startswith(package + "."):
                    sys.modules.pop(key, None)
            sys.modules[name] = types.ModuleType(name)
            _STUBBED_CUDA_OPS = True


def _patch_cuda_noop_for_cpu() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available() or getattr(torch.Tensor, "_sota_mtrpp_cuda_noop", False):
        return

    def tensor_cuda(self, device=None, non_blocking=False, memory_format=None):
        return self

    def module_cuda(self, device=None):
        return self

    torch.Tensor.cuda = tensor_cuda
    torch.nn.Module.cuda = module_cuda
    torch.Tensor._sota_mtrpp_cuda_noop = True


def _patch_encoder_global_attn_mask() -> None:
    """Fix an upstream shape bug in MTREncoder.apply_global_attn.

    Upstream permutes the padding mask as if it were 3D::

        x_mask_t = x_mask.permute(1, 0, 2)

    but x_mask is (batch_size, N), so this raises. The mask feeds
    MultiheadAttention's ``key_padding_mask``, which is batch-first even though
    ``src`` is seq-first, so it should not be permuted at all -- matching how
    MTRDecoder passes ``memory_key_padding_mask=~kv_mask`` unpermuted.

    The bug is invisible upstream because MTR++ always runs with local attention;
    only the global-attention fallback reaches this code.
    """
    try:
        from mtr.models.context_encoder.mtr_encoder import MTREncoder
        from mtr.models.utils.transformer import position_encoding_utils
    except ImportError:
        return
    if getattr(MTREncoder, "_sota_mtrpp_global_attn_mask_fix", False):
        return

    import torch

    def apply_global_attn(self, x, x_mask, x_pos):
        assert torch.all(x_mask.sum(dim=-1) > 0)

        batch_size, N, d_model = x.shape
        x_t = x.permute(1, 0, 2)
        x_pos_t = x_pos.permute(1, 0, 2)

        pos_embedding = position_encoding_utils.gen_sineembed_for_position(x_pos_t, hidden_dim=d_model)

        for k in range(len(self.self_attn_layers)):
            x_t = self.self_attn_layers[k](
                src=x_t,
                src_key_padding_mask=~x_mask,
                pos=pos_embedding,
            )
        return x_t.permute(1, 0, 2)  # (batch_size, N, d_model)

    MTREncoder.apply_global_attn = apply_global_attn
    MTREncoder._sota_mtrpp_global_attn_mask_fix = True


def _patch_global_attention_fallback(force: bool = False) -> None:
    if not (force or _STUBBED_CUDA_OPS):
        return
    try:
        from mtr.models.motion_decoder.mtr_decoder import MTRDecoder
    except ImportError:
        return
    if getattr(MTRDecoder, "_sota_mtrpp_global_attn_fallback", False):
        return

    original_build = MTRDecoder.build_transformer_decoder
    original_apply = MTRDecoder.apply_cross_attention

    def patched_build_transformer_decoder(self, in_channels, d_model, nhead, dropout=0.1,
                                          num_decoder_layers=1, use_local_attn=False):
        return original_build(
            self,
            in_channels=in_channels,
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            num_decoder_layers=num_decoder_layers,
            use_local_attn=False,
        )

    def patched_apply_cross_attention(self, kv_feature, kv_mask, kv_pos, query_content, query_embed,
                                      attention_layer, dynamic_query_center=None, layer_idx=0,
                                      use_local_attn=False, query_index_pair=None,
                                      query_content_pre_mlp=None, query_embed_pre_mlp=None):
        if use_local_attn and not getattr(attention_layer, "use_local_attn", False):
            use_local_attn = False
            query_index_pair = None
        return original_apply(
            self,
            kv_feature=kv_feature,
            kv_mask=kv_mask,
            kv_pos=kv_pos,
            query_content=query_content,
            query_embed=query_embed,
            attention_layer=attention_layer,
            dynamic_query_center=dynamic_query_center,
            layer_idx=layer_idx,
            use_local_attn=use_local_attn,
            query_index_pair=query_index_pair,
            query_content_pre_mlp=query_content_pre_mlp,
            query_embed_pre_mlp=query_embed_pre_mlp,
        )

    MTRDecoder.build_transformer_decoder = patched_build_transformer_decoder
    MTRDecoder.apply_cross_attention = patched_apply_cross_attention
    MTRDecoder._sota_mtrpp_global_attn_fallback = True


def add_upstream_to_path(path: str | Path | None = None, global_attention_fallback: bool = False) -> Path:
    upstream = resolve_upstream_dir(path)
    _install_easydict_fallback()
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    _install_cuda_op_stubs()
    _patch_cuda_noop_for_cpu()
    _patch_global_attention_fallback(force=global_attention_fallback)
    return upstream


def import_motion_transformer(path: str | Path | None = None, global_attention_fallback: bool = False):
    upstream = add_upstream_to_path(path, global_attention_fallback=global_attention_fallback)
    try:
        from mtr.config import cfg as mtr_global_cfg
        from mtr.models.model import MotionTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Unable to import official MTR. Install PyTorch and the upstream requirements; "
            "for full local-attention runs also build MTR CUDA ops with `python setup.py develop` "
            f"inside {upstream}."
        ) from exc
    _patch_encoder_global_attn_mask()
    _patch_global_attention_fallback(force=global_attention_fallback)
    return MotionTransformer, mtr_global_cfg, upstream


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


def using_cuda_op_stubs() -> bool:
    return _STUBBED_CUDA_OPS
