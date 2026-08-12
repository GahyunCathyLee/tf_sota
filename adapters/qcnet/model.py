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


class QCNetAgentEncoderWithAttrs(QCNetAgentEncoder):
    """Official QCNet agent encoder with optional continuous agent attrs.

    The base encoder embeds four continuous features per agent-time token:
    displacement norm, displacement angle, velocity norm, velocity angle. For
    ``dimI`` runs the dataset supplies ``agent.attrs[..., [dim, I]]`` and this
    encoder appends them before the Fourier embedding.
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


def build_qcnet(model_args: dict, feature_mode: str) -> QCNet:
    model = QCNet(**model_args)
    extra_dim = 2 if feature_mode == "dimI" else 0
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
