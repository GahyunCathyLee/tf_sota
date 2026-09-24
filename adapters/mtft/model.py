"""Faithful MTFT components adapted for batched highD/exiD samples."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 512) -> None:
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-math.log(1000.0) / hidden_dim))
        pe = torch.zeros(max_len, hidden_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[: x.shape[-2]].to(dtype=x.dtype, device=x.device)


def build_scale_masks(num_scales: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """Return paper scale masks where scale i observes steps divisible by i."""
    idx = torch.arange(seq_len, device=device)
    diff = idx[:, None] - idx[None, :]
    masks = []
    for scale in range(1, num_scales + 1):
        masks.append((diff.remainder(scale) == 0))
    return torch.stack(masks, dim=0)


class ScaleFeedForward(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class MultiScaleAttentionHead(nn.Module):
    """MTFT MAH: one hidden-sized stream per temporal scale/head."""

    def __init__(self, hidden_dim: int, num_scales: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.w_q = nn.Linear(hidden_dim, hidden_dim * num_scales)
        self.w_k = nn.Linear(hidden_dim, hidden_dim * num_scales)
        self.w_v = nn.Linear(hidden_dim, hidden_dim * num_scales)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
        scale_allowed: torch.Tensor,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.w_q(x).view(b, t, self.num_scales, self.hidden_dim).transpose(1, 2)
        k = self.w_k(x).view(b, t, self.num_scales, self.hidden_dim).transpose(1, 2)
        v = self.w_v(x).view(b, t, self.num_scales, self.hidden_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
        valid_keys = obs_mask[:, None, None, :]
        allowed = scale_allowed[None, :, :, :]
        scores = scores.masked_fill(~(allowed & valid_keys), torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        context = torch.matmul(self.dropout(attn), v)
        return self.norm(context + q)


class TemporalEncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_scales: int, dropout: float) -> None:
        super().__init__()
        self.mah = MultiScaleAttentionHead(hidden_dim, num_scales, dropout)
        self.ffn = ScaleFeedForward(hidden_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
        scale_allowed: torch.Tensor,
    ) -> torch.Tensor:
        return self.ffn(self.mah(x, obs_mask, scale_allowed))


class CRMF(nn.Module):
    """Continuity Representation-guided Multi-scale Fusion."""

    def __init__(self, hidden_dim: int, num_scales: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Sequential(
            nn.Linear(num_scales * hidden_dim, (num_scales * hidden_dim) // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear((num_scales * hidden_dim) // 2, hidden_dim),
        )

    def forward(
        self,
        motion: torch.Tensor,
        obs_mask: torch.Tensor,
        scale_allowed: torch.Tensor,
    ) -> torch.Tensor:
        # motion: (B, A, S, T, D), obs_mask: (B, A, T)
        b, a, s, t, d = motion.shape
        observed_keys = obs_mask[:, :, None, None, :] & scale_allowed[None, None, :, :, :]
        info_increment = observed_keys.float().sum(dim=-1)
        across_scores = info_increment.masked_fill(~obs_mask[:, :, None, :], torch.finfo(motion.dtype).min)
        across = torch.softmax(across_scores, dim=-1)
        across = torch.nan_to_num(across, nan=0.0)
        continuity = torch.sum(motion * across[..., None], dim=-2)

        q = self.q(continuity)
        k = self.k(motion)
        v = self.v(motion)
        scale_scores = torch.einsum("basd,bastd->bast", q, k) / math.sqrt(d)
        scale_scores = scale_scores.masked_fill(~obs_mask[:, :, None, :], torch.finfo(scale_scores.dtype).min)
        scale_attn = torch.softmax(scale_scores, dim=-1)
        scale_attn = torch.nan_to_num(scale_attn, nan=0.0)
        fused_per_scale = torch.sum(v * self.dropout(scale_attn)[..., None], dim=-2)
        return self.out(fused_per_scale.reshape(b, a, s * d))


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_scales: int, dropout: float) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_scales = num_scales
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pos = PositionalEncoding(hidden_dim)
        self.layers = nn.ModuleList([TemporalEncoderLayer(hidden_dim, num_scales, dropout) for _ in range(num_layers)])
        self.scale_reduce = nn.Linear(num_scales * hidden_dim, hidden_dim)
        self.crmf = CRMF(hidden_dim, num_scales, dropout)

    def forward(self, agents: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        b, t, a, f = agents.shape
        x = agents.permute(0, 2, 1, 3).reshape(b * a, t, f)
        mask = obs_mask.permute(0, 2, 1).reshape(b * a, t)
        h = self.input_proj(x) + self.pos(x).unsqueeze(0)
        h = h * mask[..., None].to(dtype=h.dtype)
        scale_allowed = build_scale_masks(self.num_scales, t, agents.device)

        multi = None
        for layer_idx, layer in enumerate(self.layers):
            multi = layer(h, mask, scale_allowed)
            if layer_idx < len(self.layers) - 1:
                h = self.scale_reduce(multi.transpose(1, 2).reshape(b * a, t, self.num_scales * self.hidden_dim))
                h = h * mask[..., None].to(dtype=h.dtype)

        assert multi is not None
        motion = multi.view(b, a, self.num_scales, t, self.hidden_dim)
        temporal = self.crmf(motion, obs_mask.permute(0, 2, 1), scale_allowed)
        return temporal


class GlobalInteraction(nn.Module):
    """Lightweight VectorNet-style fully connected global interaction."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1),
        )
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, temporal: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        b, a, d = temporal.shape
        src = temporal[:, :, None, :].expand(b, a, a, d)
        dst = temporal[:, None, :, :].expand(b, a, a, d)
        logits = self.score(torch.cat([src, dst], dim=-1)).squeeze(-1)
        valid = agent_mask[:, None, :] & agent_mask[:, :, None]
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.matmul(self.dropout(attn), self.value(temporal))
        return self.norm(temporal + out)


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_dim: int, future_steps: int, dropout: float) -> None:
        super().__init__()
        self.future_steps = int(future_steps)
        self.cell = nn.LSTMCell(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 4, 2),
        )

    def forward(self, encoded_target: torch.Tensor) -> torch.Tensor:
        h = encoded_target
        c = encoded_target
        states = []
        for _ in range(self.future_steps):
            h, c = self.cell(h, (h, c))
            states.append(h)
        return self.head(torch.stack(states, dim=1))


class MTFT(nn.Module):
    """Multi-scale Temporal Fusion Transformer for single-target prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 5,
        future_steps: int = 15,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.future_steps = int(future_steps)
        self.temporal = TemporalEncoder(input_dim, hidden_dim, num_layers, num_heads, dropout)
        self.interaction = GlobalInteraction(hidden_dim, dropout)
        self.decoder = LSTMDecoder(hidden_dim, future_steps, dropout)

    def forward(self, agents: torch.Tensor, obs_mask: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        temporal = self.temporal(agents, obs_mask)
        temporal = temporal * agent_mask[..., None].to(dtype=temporal.dtype)
        interacted = self.interaction(temporal, agent_mask)
        target_encoding = interacted[:, 0, :]
        return self.decoder(target_encoding)

