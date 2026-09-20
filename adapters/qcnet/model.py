"""QCNet model construction with highD/exiD dimI agent attributes."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
from torch_cluster import radius, radius_graph
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import dense_to_sparse, subgraph

from layers.attention_layer import AttentionLayer
from layers.fourier_embedding import FourierEmbedding
from modules.qcnet_agent_encoder import QCNetAgentEncoder
from predictors import QCNet
from utils import angle_between_2d_vectors, weight_init, wrap_angle

from adapters.common import feature_mode_names
from adapters.qcnet.multiagent_dataset import INTERACTION_EDGE_TYPE


class QCNetAgentEncoderWithAttrs(QCNetAgentEncoder):
    """Official QCNet agent encoder with optional continuous agent attrs.

    The base encoder embeds four continuous features per agent-time token:
    displacement norm, displacement angle, velocity norm, velocity angle. For
    Feature modes with side channels supply ``agent.attrs`` and this encoder
    appends them before the Fourier embedding.
    """

    def __init__(
        self,
        dataset: str,
        input_dim: int,
        hidden_dim: int,
        num_historical_steps: int,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        extra_agent_attr_dim: int,
    ) -> None:
        super().__init__(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.extra_agent_attr_dim = int(extra_agent_attr_dim)
        self.x_a_emb = FourierEmbedding(
            input_dim=4 + self.extra_agent_attr_dim,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.apply(weight_init)

    def forward(self, data: HeteroData, map_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mask = data["agent"]["valid_mask"][:, :self.num_historical_steps].contiguous()
        pos_a = data["agent"]["position"][:, :self.num_historical_steps, :self.input_dim].contiguous()
        motion_vector_a = torch.cat(
            [pos_a.new_zeros(data["agent"]["num_nodes"], 1, self.input_dim), pos_a[:, 1:] - pos_a[:, :-1]],
            dim=1,
        )
        head_a = data["agent"]["heading"][:, :self.num_historical_steps].contiguous()
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        pos_pl = data["map_polygon"]["position"][:, :self.input_dim].contiguous()
        orient_pl = data["map_polygon"]["orientation"].contiguous()

        vel = data["agent"]["velocity"][:, :self.num_historical_steps, :self.input_dim].contiguous()
        categorical_embs = [
            self.type_a_emb(data["agent"]["type"].long()).repeat_interleave(
                repeats=self.num_historical_steps, dim=0
            ),
        ]

        x_a = torch.stack(
            [
                torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]),
                torch.norm(vel[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=vel[:, :, :2]),
            ],
            dim=-1,
        )
        if self.extra_agent_attr_dim:
            attrs = data["agent"]["attrs"][:, :self.num_historical_steps, :self.extra_agent_attr_dim].contiguous()
            x_a = torch.cat([x_a, attrs], dim=-1)
        x_a = self.x_a_emb(continuous_inputs=x_a.view(-1, x_a.size(-1)), categorical_embs=categorical_embs)
        x_a = x_a.view(-1, self.num_historical_steps, self.hidden_dim)

        pos_t = pos_a.reshape(-1, self.input_dim)
        head_t = head_a.reshape(-1)
        head_vector_t = head_vector_a.reshape(-1, 2)
        mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)
        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[:, edge_index_t[1] - edge_index_t[0] <= self.time_span]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [
                torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]),
                rel_head_t,
                edge_index_t[0] - edge_index_t[1],
            ],
            dim=-1,
        )
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)

        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        mask_s = mask.transpose(0, 1).reshape(-1)
        pos_pl = pos_pl.repeat(self.num_historical_steps, 1)
        orient_pl = orient_pl.repeat(self.num_historical_steps)
        if isinstance(data, Batch):
            batch_s = torch.cat(
                [data["agent"]["batch"] + data.num_graphs * t for t in range(self.num_historical_steps)], dim=0
            )
            batch_pl = torch.cat(
                [data["map_polygon"]["batch"] + data.num_graphs * t for t in range(self.num_historical_steps)], dim=0
            )
        else:
            batch_s = torch.arange(self.num_historical_steps, device=pos_a.device).repeat_interleave(
                data["agent"]["num_nodes"]
            )
            batch_pl = torch.arange(self.num_historical_steps, device=pos_pl.device).repeat_interleave(
                data["map_polygon"]["num_nodes"]
            )
        edge_index_pl2a = radius(
            x=pos_s[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2a_radius,
            batch_x=batch_s,
            batch_y=batch_pl,
            max_num_neighbors=300,
        )
        edge_index_pl2a = edge_index_pl2a[:, mask_s[edge_index_pl2a[1]]]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]])
        r_pl2a = torch.stack(
            [
                torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a[:, :2]),
                rel_orient_pl2a,
            ],
            dim=-1,
        )
        r_pl2a = self.r_pl2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)

        edge_index_a2a = radius_graph(
            x=pos_s[:, :2], r=self.a2a_radius, batch=batch_s, loop=False, max_num_neighbors=300
        )
        edge_index_a2a = subgraph(subset=mask_s, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [
                torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
                rel_head_a2a,
            ],
            dim=-1,
        )
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)

        for i in range(self.num_layers):
            x_a = x_a.reshape(-1, self.hidden_dim)
            x_a = self.t_attn_layers[i](x_a, r_t, edge_index_t)
            x_a = x_a.reshape(-1, self.num_historical_steps, self.hidden_dim).transpose(0, 1).reshape(
                -1, self.hidden_dim
            )
            x_a = self.pl2a_attn_layers[i](
                (map_enc["x_pl"].transpose(0, 1).reshape(-1, self.hidden_dim), x_a), r_pl2a, edge_index_pl2a
            )
            x_a = self.a2a_attn_layers[i](x_a, r_a2a, edge_index_a2a)
            x_a = x_a.reshape(self.num_historical_steps, -1, self.hidden_dim).transpose(0, 1)
        return {"x_a": x_a}


class QCNetAgentEncoderWithInteractionImportance(QCNetAgentEncoder):
    """Official QCNet agent encoder with edge-level directed interaction I.

    ``data.multiagent.interaction.build_pair_features`` stores directed
    importance as ``pair_features[target_i, source_j, timestep, 9]``. QCNet's
    PyG social edges use ``edge_index=[source_j, target_i]`` after expanding
    agents across historical timesteps, so the lookup below intentionally maps
    ``edge_index_a2a[0] -> source`` and ``edge_index_a2a[1] -> target`` to
    ``pair_I[target, source, t]``.
    """

    def __init__(
        self,
        dataset: str,
        input_dim: int,
        hidden_dim: int,
        num_historical_steps: int,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
    ) -> None:
        super().__init__(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.r_a2a_emb = FourierEmbedding(input_dim=4, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)

    def _a2a_interaction_importance(self, data: HeteroData, edge_index_a2a: torch.Tensor) -> torch.Tensor:
        if edge_index_a2a.numel() == 0:
            return edge_index_a2a.new_zeros((0,), dtype=torch.float32)
        if INTERACTION_EDGE_TYPE not in data.edge_types:
            raise ValueError("use_interaction_importance=true requires edge-level interaction importance tensors")

        store = data[INTERACTION_EDGE_TYPE]
        required = {"edge_index", "time", "importance"}
        missing = required.difference(store.keys())
        if missing:
            raise ValueError(f"interaction importance edge store is missing keys: {sorted(missing)}")

        num_agents = int(data["agent"]["num_nodes"])
        src_flat = edge_index_a2a[0]
        dst_flat = edge_index_a2a[1]
        src_time = torch.div(src_flat, num_agents, rounding_mode="floor")
        dst_time = torch.div(dst_flat, num_agents, rounding_mode="floor")
        if not torch.equal(src_time, dst_time):
            raise ValueError("QCNet a2a edge source/target timesteps are not aligned")
        src_agent = src_flat.remainder(num_agents)
        dst_agent = dst_flat.remainder(num_agents)

        imp_edge_index = store["edge_index"].to(device=edge_index_a2a.device, dtype=torch.long)
        imp_time = store["time"].to(device=edge_index_a2a.device, dtype=torch.long).view(-1)
        imp = store["importance"].to(device=edge_index_a2a.device, dtype=torch.float32).view(-1)
        if imp_edge_index.size(1) != imp_time.numel() or imp.numel() != imp_time.numel():
            raise ValueError(
                "interaction importance tensors must have aligned edge_index/time/importance lengths"
            )
        if not torch.isfinite(imp).all():
            raise ValueError("NaN/Inf found in interaction importance")

        imp_src = imp_edge_index[0]
        imp_dst = imp_edge_index[1]
        imp_keys = (imp_time * num_agents + imp_src) * num_agents + imp_dst
        query_keys = (src_time * num_agents + src_agent) * num_agents + dst_agent
        order = torch.argsort(imp_keys)
        sorted_keys = imp_keys[order]
        pos = torch.searchsorted(sorted_keys, query_keys)
        in_bounds = pos < sorted_keys.numel()
        matched = torch.zeros_like(in_bounds, dtype=torch.bool)
        matched[in_bounds] = sorted_keys[pos[in_bounds]] == query_keys[in_bounds]
        if not bool(matched.all()):
            missing_count = int((~matched).sum().item())
            raise ValueError(
                f"missing interaction importance for {missing_count} QCNet a2a edges; "
                "pair_valid should match QCNet valid social edges"
            )
        return imp[order[pos]].to(dtype=torch.float32)

    def forward(self, data: HeteroData, map_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mask = data["agent"]["valid_mask"][:, :self.num_historical_steps].contiguous()
        pos_a = data["agent"]["position"][:, :self.num_historical_steps, :self.input_dim].contiguous()
        motion_vector_a = torch.cat(
            [pos_a.new_zeros(data["agent"]["num_nodes"], 1, self.input_dim), pos_a[:, 1:] - pos_a[:, :-1]],
            dim=1,
        )
        head_a = data["agent"]["heading"][:, :self.num_historical_steps].contiguous()
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        pos_pl = data["map_polygon"]["position"][:, :self.input_dim].contiguous()
        orient_pl = data["map_polygon"]["orientation"].contiguous()

        vel = data["agent"]["velocity"][:, :self.num_historical_steps, :self.input_dim].contiguous()
        categorical_embs = [
            self.type_a_emb(data["agent"]["type"].long()).repeat_interleave(
                repeats=self.num_historical_steps, dim=0
            ),
        ]

        x_a = torch.stack(
            [
                torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]),
                torch.norm(vel[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=vel[:, :, :2]),
            ],
            dim=-1,
        )
        x_a = self.x_a_emb(continuous_inputs=x_a.view(-1, x_a.size(-1)), categorical_embs=categorical_embs)
        x_a = x_a.view(-1, self.num_historical_steps, self.hidden_dim)

        pos_t = pos_a.reshape(-1, self.input_dim)
        head_t = head_a.reshape(-1)
        head_vector_t = head_vector_a.reshape(-1, 2)
        mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)
        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[:, edge_index_t[1] - edge_index_t[0] <= self.time_span]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [
                torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]),
                rel_head_t,
                edge_index_t[0] - edge_index_t[1],
            ],
            dim=-1,
        )
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)

        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        mask_s = mask.transpose(0, 1).reshape(-1)
        pos_pl = pos_pl.repeat(self.num_historical_steps, 1)
        orient_pl = orient_pl.repeat(self.num_historical_steps)
        if isinstance(data, Batch):
            batch_s = torch.cat(
                [data["agent"]["batch"] + data.num_graphs * t for t in range(self.num_historical_steps)], dim=0
            )
            batch_pl = torch.cat(
                [data["map_polygon"]["batch"] + data.num_graphs * t for t in range(self.num_historical_steps)], dim=0
            )
        else:
            batch_s = torch.arange(self.num_historical_steps, device=pos_a.device).repeat_interleave(
                data["agent"]["num_nodes"]
            )
            batch_pl = torch.arange(self.num_historical_steps, device=pos_pl.device).repeat_interleave(
                data["map_polygon"]["num_nodes"]
            )
        edge_index_pl2a = radius(
            x=pos_s[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2a_radius,
            batch_x=batch_s,
            batch_y=batch_pl,
            max_num_neighbors=300,
        )
        edge_index_pl2a = edge_index_pl2a[:, mask_s[edge_index_pl2a[1]]]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]])
        r_pl2a = torch.stack(
            [
                torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a[:, :2]),
                rel_orient_pl2a,
            ],
            dim=-1,
        )
        r_pl2a = self.r_pl2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)

        edge_index_a2a = radius_graph(
            x=pos_s[:, :2], r=self.a2a_radius, batch=batch_s, loop=False, max_num_neighbors=300
        )
        edge_index_a2a = subgraph(subset=mask_s, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        i_a2a = self._a2a_interaction_importance(data, edge_index_a2a).to(device=pos_s.device, dtype=pos_s.dtype)
        r_a2a = torch.stack(
            [
                torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
                rel_head_a2a,
                i_a2a,
            ],
            dim=-1,
        )
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)

        for i in range(self.num_layers):
            x_a = x_a.reshape(-1, self.hidden_dim)
            x_a = self.t_attn_layers[i](x_a, r_t, edge_index_t)
            x_a = x_a.reshape(-1, self.num_historical_steps, self.hidden_dim).transpose(0, 1).reshape(
                -1, self.hidden_dim
            )
            x_a = self.pl2a_attn_layers[i](
                (map_enc["x_pl"].transpose(0, 1).reshape(-1, self.hidden_dim), x_a), r_pl2a, edge_index_pl2a
            )
            x_a = self.a2a_attn_layers[i](x_a, r_a2a, edge_index_a2a)
            x_a = x_a.reshape(self.num_historical_steps, -1, self.hidden_dim).transpose(0, 1)
        return {"x_a": x_a}


def build_qcnet(model_args: dict, feature_mode: str, use_interaction_importance: bool = False) -> QCNet:
    model = QCNet(**model_args)
    extra_dim = max(0, len(feature_mode_names(feature_mode)) - 6)
    if use_interaction_importance and extra_dim:
        raise ValueError(
            "use_interaction_importance=true cannot be combined with legacy QCNet agent attrs "
            f"from feature_mode={feature_mode!r}; use feature_mode=baseline for edge-level I."
        )
    if use_interaction_importance:
        model.encoder.agent_encoder = QCNetAgentEncoderWithInteractionImportance(
            dataset=model.dataset,
            input_dim=model.input_dim,
            hidden_dim=model.hidden_dim,
            num_historical_steps=model.num_historical_steps,
            time_span=model.time_span,
            pl2a_radius=model.pl2a_radius,
            a2a_radius=model.a2a_radius,
            num_freq_bands=model.num_freq_bands,
            num_layers=model.num_agent_layers,
            num_heads=model.num_heads,
            head_dim=model.head_dim,
            dropout=model.dropout,
        )
        return model
    if extra_dim:
        model.encoder.agent_encoder = QCNetAgentEncoderWithAttrs(
            dataset=model.dataset,
            input_dim=model.input_dim,
            hidden_dim=model.hidden_dim,
            num_historical_steps=model.num_historical_steps,
            time_span=model.time_span,
            pl2a_radius=model.pl2a_radius,
            a2a_radius=model.a2a_radius,
            num_freq_bands=model.num_freq_bands,
            num_layers=model.num_agent_layers,
            num_heads=model.num_heads,
            head_dim=model.head_dim,
            dropout=model.dropout,
            extra_agent_attr_dim=extra_dim,
        )
    return model
