"""Small MTP-GO encoder shim for native edge features plus scalar ``I``."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_gru_gnn_encoder(
    *,
    input_size: int,
    hidden_size: int,
    n_mixtures: int,
    n_layers: int,
    gnn_layer: str,
    n_heads: int,
    static_f_dim: int,
    init_static: bool,
    use_edge_features: bool,
    edge_feature_dim: int,
) -> nn.Module:
    """Build the upstream GRUGNNEncoder, allowing edge feature dim > 1.

    Upstream exposes ``use_edge_features`` as a boolean and maps it to
    ``edge_dim=1``.  The +I experiment needs ``[distance, I]`` edge attributes,
    so this wrapper preserves the same encoder structure while widening only the
    edge feature dimension.
    """
    if int(edge_feature_dim) == 1:
        from models.gru_gnn import GRUGNNEncoder

        return GRUGNNEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            n_mixtures=n_mixtures,
            n_layers=n_layers,
            gnn_layer=gnn_layer,
            n_heads=n_heads,
            static_f_dim=static_f_dim,
            init_static=init_static,
            use_edge_features=use_edge_features,
        )
    return FlexibleEdgeGRUGNNEncoder(
        input_size=input_size,
        hidden_size=hidden_size,
        n_mixtures=n_mixtures,
        n_layers=n_layers,
        gnn_layer=gnn_layer,
        n_heads=n_heads,
        static_f_dim=static_f_dim,
        init_static=init_static,
        use_edge_features=use_edge_features,
        edge_feature_dim=edge_feature_dim,
    )


class FlexibleEdgeGRUGNNEncoder(nn.Module):
    """Upstream ``GRUGNNEncoder`` with configurable edge feature dimension."""

    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        n_heads: int = 3,
        n_layers: int = 1,
        n_mixtures: int = 7,
        static_f_dim: int = 6,
        dropout: float = 0.1,
        gnn_layer: str = "graphconv",
        init_static: bool = False,
        use_edge_features: bool = True,
        edge_feature_dim: int = 1,
    ) -> None:
        super().__init__()
        from models.gnn_layers import create_sequential_gnn
        from models.gru_gnn import GRUGNNCell

        self.hidden_size = hidden_size
        self.init_static = init_static
        self.dropout = nn.Dropout(p=dropout)
        self.static_feature_dim = static_f_dim

        if self.init_static:
            self.init_gnn = create_sequential_gnn(
                input_size=self.static_feature_dim,
                output_size=hidden_size,
                hidden_size=hidden_size,
                n_heads=n_heads,
                dropout=dropout,
                layers=n_layers,
                activation="elu",
                gnn_layer=gnn_layer,
            )
        else:
            self.init_state_param = nn.Parameter(torch.empty(hidden_size).uniform_(-1e-2, 1e-2))

        edge_dim = int(edge_feature_dim) if use_edge_features else None
        self.gru_cell = GRUGNNCell(
            input_size,
            hidden_size,
            n_heads,
            n_layers,
            dropout,
            gnn_layer,
            edge_dim=edge_dim,
        )
        self.mixture = nn.Linear(hidden_size, n_mixtures)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        if hasattr(self, "init_state_param"):
            nn.init.uniform_(self.init_state_param, -1e-2, 1e-2)
        nn.init.uniform_(self.mixture.bias, -std, std)
        nn.init.xavier_uniform_(self.mixture.weight)

    def init_hidden(self, data, batch_size: int) -> torch.Tensor:
        if self.init_static:
            from models.utils import extract_static_features

            init_gnn_input = extract_static_features(data)
            return self.init_gnn(init_gnn_input, data.edge_index[0], None)
        return self.init_state_param.repeat(batch_size, 1)

    def forward(self, data):
        x, edge_index, edge_features = data.x, data.edge_index, data.edge_features
        batch_size, _, _ = x.size()
        hidden = self.init_hidden(data, batch_size)
        output = [hidden]

        for x_i, ei_i, ef_i in zip(x.transpose(0, 1), edge_index, edge_features):
            hidden = self.gru_cell(x_i, ei_i, hidden, edge_attr=ef_i.to(torch.float32))
            output.append(hidden)
        output_t = torch.stack(output, dim=1)
        mixture_w = self.mixture(self.dropout(F.elu(hidden)))
        return output_t, mixture_w
