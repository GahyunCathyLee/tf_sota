"""Scene-first MTR++ baseline adapted to highD/exiD multi-agent arrays."""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def wrap_angle_torch(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def rotate_torch(points: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    c = torch.cos(heading)
    s = torch.sin(heading)
    x = points[..., 0]
    y = points[..., 1]
    return torch.stack([x * c - y * s, x * s + y * c], dim=-1)


def sine_position(points: torch.Tensor, d_model: int) -> torch.Tensor:
    half = max(1, d_model // 4)
    scale = torch.arange(half, device=points.device, dtype=points.dtype)
    scale = torch.pow(torch.tensor(10000.0, device=points.device, dtype=points.dtype), -scale / max(1, half))
    x = points[..., 0:1] * scale
    y = points[..., 1:2] * scale
    emb = torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=-1)
    if emb.shape[-1] < d_model:
        emb = F.pad(emb, (0, d_model - emb.shape[-1]))
    return emb[..., :d_model]


def pair_geometry(q_pos: torch.Tensor, q_heading: torch.Tensor, k_pos: torch.Tensor, k_heading: torch.Tensor) -> torch.Tensor:
    delta = k_pos.unsqueeze(-3) - q_pos.unsqueeze(-2)
    rel_pos = rotate_torch(delta, -q_heading.unsqueeze(-1))
    rel_heading = wrap_angle_torch(k_heading.unsqueeze(-2) - q_heading.unsqueeze(-1))
    return torch.cat([rel_pos, torch.sin(rel_heading).unsqueeze(-1), torch.cos(rel_heading).unsqueeze(-1)], dim=-1)


def gather_by_index(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    gather_index = index.clamp_min(0)
    while gather_index.ndim < values.ndim:
        gather_index = gather_index.unsqueeze(-1)
    gather_index = gather_index.expand(*index.shape, *values.shape[2:])
    return torch.gather(values, dim=1, index=gather_index)


class MaskedPolylineEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_channels: int, num_layers: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for _ in range(max(1, num_layers - 1)):
            layers.extend([nn.Linear(c_in, hidden_dim), nn.ReLU()])
            c_in = hidden_dim
        layers.append(nn.Linear(c_in, out_channels))
        self.mlp = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        feat = self.mlp(x)
        feat = feat.masked_fill(~mask.unsqueeze(-1), -1.0e4)
        pooled = feat.max(dim=-2).values
        return pooled.masked_fill(~mask.any(dim=-1, keepdim=True), 0.0)


class QueryCentricAttention(nn.Module):
    """Multi-head attention with query-frame relative geometry in score and value."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.rel_score = nn.Sequential(nn.Linear(4, d_model), nn.ReLU(), nn.Linear(d_model, num_heads))
        self.rel_value = nn.Sequential(nn.Linear(4, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))
        self.ffn_norm = nn.LayerNorm(d_model)
        self.last_attn_weights: torch.Tensor | None = None

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_pos: torch.Tensor,
        query_heading: torch.Tensor,
        key_pos: torch.Tensor,
        key_heading: torch.Tensor,
        query_mask: torch.Tensor,
        key_mask: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, qn, _ = query.shape
        kn = key_value.shape[1]
        q = self.q_proj(query).view(b, qn, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(b, kn, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(b, kn, self.num_heads, self.head_dim).transpose(1, 2)
        rel = pair_geometry(query_pos, query_heading, key_pos, key_heading)
        rel_score = self.rel_score(rel).permute(0, 3, 1, 2)
        rel_value = self.rel_value(rel).view(b, qn, kn, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores + rel_score
        valid = query_mask.unsqueeze(-1) & key_mask.unsqueeze(-2)
        if pair_mask is not None:
            valid = valid & pair_mask
        scores = scores.masked_fill(~valid.unsqueeze(1), -1.0e4)
        weights = torch.softmax(scores, dim=-1).masked_fill(~valid.unsqueeze(1), 0.0)
        weights = self.dropout(weights)
        self.last_attn_weights = weights.detach()
        context = torch.einsum("bhqk,bhkd->bhqd", weights, v)
        context = context + torch.einsum("bhqk,bhqkd->bhqd", weights, rel_value)
        context = context.transpose(1, 2).contiguous().view(b, qn, self.d_model)
        out = self.norm(query + self.out_proj(context))
        out = self.ffn_norm(out + self.ffn(out))
        return out.masked_fill(~query_mask.unsqueeze(-1), 0.0)


class MTRPPContextEncoder(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.model_cfg = config
        d_model = int(_cfg_get(config, "D_MODEL", 256))
        heads = int(_cfg_get(config, "NUM_ATTN_HEAD", 8))
        self.num_neighbors = int(_cfg_get(config, "NUM_OF_ATTN_NEIGHBORS", 16))
        self.agent_encoder = MaskedPolylineEncoder(
            int(_cfg_get(config, "NUM_INPUT_ATTR_AGENT")) + 1,
            int(_cfg_get(config, "NUM_CHANNEL_IN_MLP_AGENT", d_model)),
            d_model,
            int(_cfg_get(config, "NUM_LAYER_IN_MLP_AGENT", 3)),
        )
        self.map_encoder = MaskedPolylineEncoder(
            int(_cfg_get(config, "NUM_INPUT_ATTR_MAP", 9)),
            int(_cfg_get(config, "NUM_CHANNEL_IN_MLP_MAP", 64)),
            d_model,
            int(_cfg_get(config, "NUM_LAYER_IN_MLP_MAP", 5)),
        )
        self.layers = nn.ModuleList(
            [QueryCentricAttention(d_model, heads, float(_cfg_get(config, "DROPOUT_OF_ATTN", 0.1))) for _ in range(int(_cfg_get(config, "NUM_ATTN_LAYERS", 6)))]
        )
        self.num_out_channels = d_model
        self.call_count = 0
        self.last_pair_mask: torch.Tensor | None = None

    def _knn_pair_mask(self, pos: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(pos, pos)
        dist = dist.masked_fill(~valid.unsqueeze(1), 1.0e7).masked_fill(~valid.unsqueeze(2), 1.0e7)
        k = min(max(1, self.num_neighbors), pos.shape[1])
        idx = dist.topk(k=k, largest=False, dim=-1).indices
        mask = torch.zeros(valid.shape[0], pos.shape[1], pos.shape[1], dtype=torch.bool, device=pos.device)
        mask.scatter_(dim=-1, index=idx, value=True)
        return mask & valid.unsqueeze(1) & valid.unsqueeze(2)

    def forward(self, input_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        self.call_count += 1
        obj = input_dict["obj_trajs"]
        obj_mask = input_dict["obj_trajs_mask"].bool()
        map_poly = input_dict["map_polylines"]
        map_mask = input_dict["map_polylines_mask"].bool()
        obj_in = torch.cat([obj, obj_mask.unsqueeze(-1).type_as(obj)], dim=-1)
        obj_feature = self.agent_encoder(obj_in, obj_mask)
        map_feature = self.map_encoder(map_poly, map_mask)
        obj_valid = obj_mask.any(dim=-1)
        map_valid = map_mask.any(dim=-1)
        token_feature = torch.cat([obj_feature, map_feature], dim=1)
        token_mask = torch.cat([obj_valid, map_valid], dim=1)
        token_pos = torch.cat([input_dict["agent_token_pos"], input_dict["map_polylines_center"][..., 0:2]], dim=1)
        token_heading = torch.cat([input_dict["agent_token_heading"], input_dict["map_token_heading"]], dim=1)
        pair_mask = self._knn_pair_mask(token_pos, token_mask)
        self.last_pair_mask = pair_mask.detach()
        for layer in self.layers:
            token_feature = layer(
                token_feature,
                token_feature,
                token_pos,
                token_heading,
                token_pos,
                token_heading,
                token_mask,
                token_mask,
                pair_mask=pair_mask,
            )
        n_obj = obj.shape[1]
        return {
            "obj_feature": token_feature[:, :n_obj],
            "map_feature": token_feature[:, n_obj:],
            "obj_mask": obj_valid,
            "map_mask": map_valid,
            "obj_pos": input_dict["agent_token_pos"],
            "obj_heading": input_dict["agent_token_heading"],
            "map_pos": input_dict["map_polylines_center"][..., 0:2],
            "map_heading": input_dict["map_token_heading"],
        }


class MTRPPMotionDecoder(nn.Module):
    def __init__(self, in_channels: int, config: Any) -> None:
        super().__init__()
        self.model_cfg = config
        self.num_future_frames = int(_cfg_get(config, "NUM_FUTURE_FRAMES"))
        self.num_motion_modes = int(_cfg_get(config, "NUM_MOTION_MODES", 6))
        self.d_model = int(_cfg_get(config, "D_MODEL", in_channels))
        heads = int(_cfg_get(config, "NUM_ATTN_HEAD", 8))
        dropout = float(_cfg_get(config, "DROPOUT_OF_ATTN", 0.1))
        self.num_layers = int(_cfg_get(config, "NUM_DECODER_LAYERS", 6))
        self.in_proj_obj = nn.Linear(in_channels, self.d_model)
        self.in_proj_map = nn.Linear(in_channels, self.d_model)
        self.in_proj_focal = nn.Linear(in_channels, self.d_model)
        self.intent_mlp = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.ReLU(), nn.Linear(self.d_model, self.d_model))
        self.guided_layers = nn.ModuleList([QueryCentricAttention(self.d_model, heads, dropout) for _ in range(self.num_layers)])
        self.obj_cross_layers = nn.ModuleList([QueryCentricAttention(self.d_model, heads, dropout) for _ in range(self.num_layers)])
        self.map_cross_layers = nn.ModuleList([QueryCentricAttention(self.d_model, heads, dropout) for _ in range(self.num_layers)])
        self.fuse_layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.d_model * 4, self.d_model), nn.ReLU(), nn.Linear(self.d_model, self.d_model)) for _ in range(self.num_layers)]
        )
        self.cls_heads = nn.ModuleList([nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.ReLU(), nn.Linear(self.d_model, 1)) for _ in range(self.num_layers)])
        self.reg_heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.ReLU(), nn.Linear(self.d_model, self.num_future_frames * 7)) for _ in range(self.num_layers)]
        )
        self.dense_head = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.ReLU(), nn.Linear(self.d_model, self.num_future_frames * 4))
        self.forward_ret_dict: dict[str, Any] = {}
        self.last_guided_query: torch.Tensor | None = None
        self.last_intention_points_global: torch.Tensor | None = None
        self.last_guided_token_mask: torch.Tensor | None = None
        self.register_buffer("vehicle_intention_points", self._load_intention_points(config), persistent=False)

    def _load_intention_points(self, config: Any) -> torch.Tensor:
        path = Path(str(_cfg_get(config, "INTENTION_POINTS_FILE")))
        if not path.is_absolute():
            path = Path.cwd() / path
        with open(path, "rb") as f:
            payload = pickle.load(f)
        points = torch.as_tensor(payload.get("TYPE_VEHICLE", next(iter(payload.values()))), dtype=torch.float32)
        return points.view(-1, 2)

    def _build_queries(self, focal_pos: torch.Tensor, focal_heading: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local = self.vehicle_intention_points.to(focal_pos.device, focal_pos.dtype)
        local = local[: self.num_motion_modes]
        local = local.view(1, 1, self.num_motion_modes, 2).expand(focal_pos.shape[0], focal_pos.shape[1], -1, -1)
        global_points = focal_pos.unsqueeze(2) + rotate_torch(local, focal_heading.unsqueeze(-1))
        query = self.intent_mlp(sine_position(local, self.d_model))
        return query, local, global_points

    def _map_pair_mask(self, query_pos: torch.Tensor, map_pos: torch.Tensor, query_mask: torch.Tensor, map_mask: torch.Tensor) -> torch.Tensor:
        b, qn, _ = query_pos.shape
        pn = map_pos.shape[1]
        dist = torch.cdist(query_pos, map_pos).masked_fill(~map_mask.unsqueeze(1), 1.0e7)
        k = min(pn, int(_cfg_get(self.model_cfg, "NUM_BASE_MAP_POLYLINES", pn)))
        idx = dist.topk(k=max(1, k), largest=False, dim=-1).indices
        mask = torch.zeros(b, qn, pn, dtype=torch.bool, device=query_pos.device)
        mask.scatter_(-1, idx, True)
        return mask & query_mask.unsqueeze(-1) & map_mask.unsqueeze(1)

    def forward(self, input_dict: dict[str, Any], enc: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        obj_feature = self.in_proj_obj(enc["obj_feature"])
        map_feature = self.in_proj_map(enc["map_feature"])
        focal_idx = input_dict["focal_track_indices"].long()
        focal_mask = input_dict["focal_mask"].bool()
        focal_feature = self.in_proj_focal(gather_by_index(enc["obj_feature"], focal_idx))
        focal_pos = gather_by_index(enc["obj_pos"], focal_idx)
        focal_heading = gather_by_index(enc["obj_heading"].unsqueeze(-1), focal_idx).squeeze(-1)
        query, intention_local, intention_global = self._build_queries(focal_pos, focal_heading)
        b, m, k, _ = query.shape
        query = query + focal_feature.unsqueeze(2)
        q_mask = focal_mask.unsqueeze(-1).expand(b, m, k)
        flat_query = query.view(b, m * k, self.d_model)
        flat_mask = q_mask.reshape(b, m * k)
        flat_pos = intention_global.reshape(b, m * k, 2)
        flat_heading = focal_heading.unsqueeze(-1).expand(b, m, k).reshape(b, m * k)
        center_feature = focal_feature.unsqueeze(2).expand(b, m, k, self.d_model).reshape(b, m * k, self.d_model)
        pred_list = []
        dynamic_pos = flat_pos
        for layer_idx in range(self.num_layers):
            guided = self.guided_layers[layer_idx](flat_query, flat_query, dynamic_pos, flat_heading, dynamic_pos, flat_heading, flat_mask, flat_mask)
            obj_ctx = self.obj_cross_layers[layer_idx](
                guided,
                obj_feature,
                dynamic_pos,
                flat_heading,
                enc["obj_pos"],
                enc["obj_heading"],
                flat_mask,
                enc["obj_mask"],
            )
            map_ctx = self.map_cross_layers[layer_idx](
                guided,
                map_feature,
                dynamic_pos,
                flat_heading,
                enc["map_pos"],
                enc["map_heading"],
                flat_mask,
                enc["map_mask"],
                pair_mask=self._map_pair_mask(dynamic_pos, enc["map_pos"], flat_mask, enc["map_mask"]),
            )
            flat_query = self.fuse_layers[layer_idx](torch.cat([center_feature, guided, obj_ctx, map_ctx], dim=-1))
            flat_query = flat_query.masked_fill(~flat_mask.unsqueeze(-1), 0.0)
            pred_scores = self.cls_heads[layer_idx](flat_query).view(b, m, k)
            raw = self.reg_heads[layer_idx](flat_query).view(b, m, k, self.num_future_frames, 7)
            local_xy = raw[..., 0:2]
            global_xy = focal_pos[:, :, None, None, :] + rotate_torch(local_xy, focal_heading[:, :, None, None])
            local_vel = raw[..., 5:7]
            global_vel = rotate_torch(local_vel, focal_heading[:, :, None, None])
            pred_trajs = torch.cat([global_xy, raw[..., 2:5], global_vel], dim=-1)
            pred_scores = pred_scores.masked_fill(~focal_mask.unsqueeze(-1), -1.0e4)
            pred_trajs = pred_trajs.masked_fill(~focal_mask[:, :, None, None, None], 0.0)
            pred_list.append((pred_scores, pred_trajs))
            dynamic_pos = pred_trajs[..., -1, 0:2].reshape(b, m * k, 2)
        dense_raw = self.dense_head(obj_feature).view(obj_feature.shape[0], obj_feature.shape[1], self.num_future_frames, 4)
        dense_xy = enc["obj_pos"].unsqueeze(2) + rotate_torch(dense_raw[..., 0:2], enc["obj_heading"].unsqueeze(-1))
        dense_vel = rotate_torch(dense_raw[..., 2:4], enc["obj_heading"].unsqueeze(-1))
        dense_pred = torch.cat([dense_xy, dense_vel], dim=-1).masked_fill(~enc["obj_mask"][:, :, None, None], 0.0)
        self.forward_ret_dict = {
            "pred_list": pred_list,
            "intention_points": intention_global,
            "intention_points_local": intention_local,
            "focal_mask": focal_mask,
            "focal_gt_trajs": input_dict["focal_gt_trajs"],
            "focal_gt_mask": input_dict["focal_gt_mask"],
            "obj_trajs_future_state": input_dict["obj_trajs_future_state"],
            "obj_trajs_future_mask": input_dict["obj_trajs_future_mask"],
            "pred_dense_trajs": dense_pred,
        }
        self.last_guided_query = flat_query.detach().view(b, m, k, self.d_model)
        self.last_intention_points_global = intention_global.detach()
        self.last_guided_token_mask = flat_mask.detach()
        return {"pred_scores": pred_list[-1][0], "pred_trajs": pred_list[-1][1]}

    def get_loss(self) -> tuple[torch.Tensor, dict[str, float], dict[str, float]]:
        gt = self.forward_ret_dict["focal_gt_trajs"]
        gt_mask = self.forward_ret_dict["focal_gt_mask"].bool()
        focal_mask = self.forward_ret_dict["focal_mask"].bool()
        valid = focal_mask & gt_mask.any(dim=-1)
        tb: dict[str, float] = {}
        total = gt.new_tensor(0.0)
        for layer_idx, (scores, trajs) in enumerate(self.forward_ret_dict["pred_list"]):
            final_idx = gt_mask.long().sum(dim=-1).clamp_min(1) - 1
            gt_goal = gt.gather(2, final_idx[..., None, None].expand(*gt.shape[:2], 1, gt.shape[-1])).squeeze(2)[..., 0:2]
            mode_goal = trajs[..., -1, 0:2]
            nearest = torch.norm(mode_goal - gt_goal.unsqueeze(2), dim=-1).argmin(dim=-1)
            try:
                from mtr.utils import loss_utils

                reg_flat, _ = loss_utils.nll_loss_gmm_direct(
                    pred_scores=scores.reshape(-1, scores.shape[-1]),
                    pred_trajs=trajs[..., 0:5].reshape(-1, trajs.shape[2], trajs.shape[3], 5),
                    gt_trajs=gt[..., 0:2].reshape(-1, gt.shape[2], 2),
                    gt_valid_mask=gt_mask.reshape(-1, gt_mask.shape[-1]),
                    pre_nearest_mode_idxs=nearest.reshape(-1),
                    timestamp_loss_weight=None,
                    use_square_gmm=False,
                )
                reg = reg_flat.view_as(nearest)
            except Exception:
                chosen = trajs.gather(2, nearest[..., None, None, None].expand(*nearest.shape, 1, trajs.shape[-2], trajs.shape[-1])).squeeze(2)
                pos_loss = torch.norm(chosen[..., 0:2] - gt[..., 0:2], dim=-1)
                reg = (pos_loss * gt_mask.float()).sum(dim=-1) / gt_mask.float().sum(dim=-1).clamp_min(1.0)
            chosen = trajs.gather(2, nearest[..., None, None, None].expand(*nearest.shape, 1, trajs.shape[-2], trajs.shape[-1])).squeeze(2)
            vel_loss = F.l1_loss(chosen[..., 5:7], gt[..., 2:4], reduction="none").sum(dim=-1)
            vel_loss = (vel_loss * gt_mask.float()).sum(dim=-1) / gt_mask.float().sum(dim=-1).clamp_min(1.0)
            cls = F.cross_entropy(scores.view(-1, scores.shape[-1]), nearest.view(-1), reduction="none").view_as(nearest)
            layer_loss = ((reg + 0.5 * vel_loss + cls) * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
            total = total + layer_loss
            tb[f"loss_layer{layer_idx}"] = float(layer_loss.detach())
        total = total / max(1, len(self.forward_ret_dict["pred_list"]))
        dense = self.forward_ret_dict["pred_dense_trajs"]
        fut = self.forward_ret_dict["obj_trajs_future_state"]
        fut_mask = self.forward_ret_dict["obj_trajs_future_mask"].bool()
        dense_loss = torch.norm(dense[..., 0:2] - fut[..., 0:2], dim=-1)
        dense_loss = (dense_loss * fut_mask.float()).sum() / fut_mask.float().sum().clamp_min(1.0)
        total = total + 0.2 * dense_loss
        tb["loss_dense_prediction"] = float(dense_loss.detach())
        tb["loss"] = float(total.detach())
        return total, tb, dict(tb)


class MTRPPMotionTransformer(nn.Module):
    schema_version = "mtrpp_scene_first_v1"

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.model_cfg = config
        self.context_encoder = MTRPPContextEncoder(config.CONTEXT_ENCODER)
        self.motion_decoder = MTRPPMotionDecoder(self.context_encoder.num_out_channels, config.MOTION_DECODER)

    def forward(self, batch_dict: dict[str, Any]):
        input_dict = batch_dict["input_dict"]
        enc = self.context_encoder(input_dict)
        pred = self.motion_decoder(input_dict, enc)
        batch_dict.update(enc)
        batch_dict.update(pred)
        batch_dict["focal_objects_feature"] = gather_by_index(enc["obj_feature"], input_dict["focal_track_indices"].long())
        if self.training:
            return self.get_loss()
        return batch_dict

    def get_loss(self):
        return self.motion_decoder.get_loss()
