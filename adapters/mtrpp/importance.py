"""Importance helpers for the MTR adapter.

The functions here keep scalar interaction importance out of agent history
features and align it only with MTR local-attention query-key pairs.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def last_valid_importance_matrix(
    pair_features: np.ndarray,
    pair_valid: np.ndarray,
    keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return directed last-valid I for retained local agents.

    Args:
        pair_features: Full-scene directed pair features ``[A, A, T, D]`` with
            scalar importance stored at channel 9.
        pair_valid: Full-scene pair validity ``[A, A, T]``.
        keep: Local object index to original scene-agent index mapping.

    Returns:
        importance: ``[N, N]`` where ``importance[target_local, source_local]``
            is ``pair_features[target_original, source_original, last_t, 9]``.
        valid: ``[N, N]`` true only when that directed pair has at least one
            valid historical timestep.
        last_t: ``[N, N]`` selected historical timestep, or ``-1`` if invalid.
    """

    keep = np.asarray(keep, dtype=np.int64)
    n_obj = int(keep.size)
    importance = np.zeros((n_obj, n_obj), dtype=np.float32)
    valid_out = np.zeros((n_obj, n_obj), dtype=bool)
    last_t = np.full((n_obj, n_obj), -1, dtype=np.int64)
    if n_obj == 0:
        return importance, valid_out, last_t

    pair = np.asarray(pair_features, dtype=np.float32)
    valid = np.asarray(pair_valid, dtype=bool)
    if pair.ndim != 4 or pair.shape[-1] <= 9:
        raise ValueError(f"pair_features must have shape [A,A,T,D>=10], got {pair.shape}")
    if valid.shape != pair.shape[:3]:
        raise ValueError(f"pair_valid shape {valid.shape} does not match pair_features {pair.shape[:3]}")

    local_valid = valid[keep[:, None], keep[None, :]]
    has_valid = local_valid.any(axis=-1)
    if not bool(has_valid.any()):
        return importance, valid_out, last_t

    reverse_idx = np.argmax(local_valid[..., ::-1], axis=-1)
    selected_t = local_valid.shape[-1] - 1 - reverse_idx
    selected_t = np.where(has_valid, selected_t, -1).astype(np.int64)

    target_idx, source_idx = np.nonzero(has_valid)
    orig_target = keep[target_idx]
    orig_source = keep[source_idx]
    chosen_t = selected_t[target_idx, source_idx]
    importance[target_idx, source_idx] = pair[orig_target, orig_source, chosen_t, 9]
    valid_out[target_idx, source_idx] = True
    last_t[target_idx, source_idx] = chosen_t
    importance[~np.isfinite(importance)] = 0.0
    return importance, valid_out, last_t


def build_relative_importance_weights(
    *,
    torch_module: Any,
    agent_importance: Any,
    agent_importance_valid: Any,
    index_pair: Any,
    batch_offsets: Any,
    token_batch_idxs: Any,
    token_local_idxs: Any,
    num_objects: int,
    num_heads: int,
) -> Any:
    """Build ``[N, K, H]`` local-attention relative weights from agent I.

    ``index_pair`` stores key indices local to each batch element. The returned
    tensor is aligned with each existing ``query_token <- key_token`` slot and
    gives zero bias to invalid slots and every relation involving a map token.
    """

    torch = torch_module
    if agent_importance is None or agent_importance_valid is None:
        raise ValueError("Importance-enabled MTR requires agent_importance and agent_importance_valid.")

    if index_pair.numel() == 0:
        return index_pair.new_zeros((*index_pair.shape, int(num_heads)), dtype=agent_importance.dtype)

    device = index_pair.device
    num_objects = int(num_objects)
    num_heads = int(num_heads)
    index_pair_long = index_pair.long()
    valid_knn = index_pair_long >= 0

    query_batch = token_batch_idxs.long()
    key_global = batch_offsets.long()[query_batch].unsqueeze(1) + index_pair_long.clamp_min(0)
    key_token_local = token_local_idxs.long()[key_global]
    query_token_local = token_local_idxs.long().unsqueeze(1).expand_as(key_token_local)

    query_is_agent = query_token_local < num_objects
    key_is_agent = key_token_local < num_objects
    agent_pair = valid_knn & query_is_agent & key_is_agent

    safe_query = query_token_local.clamp(min=0, max=max(num_objects - 1, 0))
    safe_key = key_token_local.clamp(min=0, max=max(num_objects - 1, 0))
    safe_batch = query_batch.unsqueeze(1).expand_as(safe_query)

    pair_i = agent_importance[safe_batch, safe_query, safe_key]
    pair_valid = agent_importance_valid[safe_batch, safe_query, safe_key].bool()
    scalar_bias = torch.where(agent_pair & pair_valid, pair_i, torch.zeros_like(pair_i))
    scalar_bias = torch.nan_to_num(scalar_bias, nan=0.0, posinf=0.0, neginf=0.0)
    return scalar_bias.unsqueeze(-1).expand(-1, -1, num_heads).contiguous()


def relation_scope_counts(
    *,
    index_pair: Any,
    batch_offsets: Any,
    token_batch_idxs: Any,
    token_local_idxs: Any,
    num_objects: int,
) -> dict[str, int]:
    """Count local-attention relation types for diagnostics/tests."""

    index_pair_long = index_pair.long()
    valid_knn = index_pair_long >= 0
    if int(valid_knn.sum()) == 0:
        return {"agent_agent": 0, "agent_map": 0, "map_agent": 0, "map_map": 0}

    query_batch = token_batch_idxs.long()
    key_global = batch_offsets.long()[query_batch].unsqueeze(1) + index_pair_long.clamp_min(0)
    key_token_local = token_local_idxs.long()[key_global]
    query_token_local = token_local_idxs.long().unsqueeze(1).expand_as(key_token_local)
    query_agent = query_token_local < int(num_objects)
    key_agent = key_token_local < int(num_objects)

    return {
        "agent_agent": int((valid_knn & query_agent & key_agent).sum().item()),
        "agent_map": int((valid_knn & query_agent & ~key_agent).sum().item()),
        "map_agent": int((valid_knn & ~query_agent & key_agent).sum().item()),
        "map_map": int((valid_knn & ~query_agent & ~key_agent).sum().item()),
    }
