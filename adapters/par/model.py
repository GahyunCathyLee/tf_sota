"""PAR-style trajectory-token model for highD/exiD.

The official PAR car model uses a Llama causal LM over discretized trajectory
tokens.  This wrapper keeps that backbone while avoiding a hard dependency on
``trajdata.AgentBatch`` for the NeighFormer arrays.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class PARTrajectoryModel(nn.Module):
    """Causal LM over interleaved neighbour/ego acceleration tokens."""

    def __init__(
        self,
        vocab_size: int,
        pad_index: int,
        num_agents: int,
        side_dim: int = 0,
        transformer: dict[str, Any] | None = None,
        use_agent_embedding: bool = True,
        dropout_hidden: float = 0.0,
        dropout_attn: float = 0.0,
    ) -> None:
        super().__init__()
        from transformers import LlamaConfig, LlamaForCausalLM

        transformer = transformer or {}
        hidden = int(transformer.get("hsize", 128))
        config = LlamaConfig(
            vocab_size=int(vocab_size),
            hidden_size=hidden,
            intermediate_size=int(transformer.get("isize", hidden * 4)),
            num_hidden_layers=int(transformer.get("depth", 8)),
            num_attention_heads=int(transformer.get("heads", 8)),
            num_key_value_heads=int(transformer.get("heads", 8)),
            max_position_embeddings=int(transformer.get("max_position_embeddings", 4098)),
            use_cache=False,
        )
        if dropout_hidden > 0:
            config.hidden_dropout_prob = float(dropout_hidden)
        if dropout_attn > 0:
            config.attention_probs_dropout_prob = float(dropout_attn)
        self.vgpt = LlamaForCausalLM(config)
        self.pad_index = int(pad_index)
        self.num_agents = int(num_agents)
        self.side_dim = int(side_dim)
        self.agent_embedding = nn.Embedding(self.num_agents, hidden) if use_agent_embedding else None
        self.side_projection = nn.Linear(self.side_dim, hidden) if self.side_dim > 0 else None

    def input_embeddings(
        self,
        tokens: torch.Tensor,
        agent_ids: torch.Tensor,
        side: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeds = self.vgpt.get_input_embeddings()(tokens)
        if self.agent_embedding is not None:
            embeds = embeds + self.agent_embedding(agent_ids)
        if self.side_projection is not None:
            if side is None:
                raise ValueError("side-channel tensor is required for this PAR model")
            embeds = embeds + self.side_projection(side.to(dtype=embeds.dtype))
        return embeds

    def forward(
        self,
        tokens: torch.Tensor,
        agent_ids: torch.Tensor,
        side: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeds = self.input_embeddings(tokens, agent_ids, side)
        return self.vgpt(inputs_embeds=embeds, use_cache=False).logits

    def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        tokens = batch["tokens"]
        logits = self(tokens, batch["agent_ids"], batch.get("side"))
        target = tokens[:, 1:]
        mask = batch["loss_mask"][:, 1:].bool() & (target != self.pad_index)
        per_tok = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            ignore_index=self.pad_index,
            reduction="none",
        ).reshape_as(target)
        loss = per_tok[mask].mean() if mask.any() else per_tok.new_tensor(0.0)
        return loss, {"loss": float(loss.detach()), "tokens_supervised": int(mask.sum().detach().cpu())}

    @torch.no_grad()
    def generate_ego_future_tokens(
        self,
        batch: dict[str, torch.Tensor],
        obs_token_steps: int,
        future_len: int,
        multinomial_sampling: bool = False,
        generate_all_agents: bool = False,
    ) -> torch.Tensor:
        """Generate ego future acceleration tokens from observed social context."""
        tokens_full = batch["tokens"]
        agent_full = batch["agent_ids"]
        side_full = batch.get("side")
        num_agents = self.num_agents
        prefix_len = int(obs_token_steps) * num_agents
        seq = tokens_full[:, :prefix_len].clone()
        agent_seq = agent_full[:, :prefix_len].clone()
        side_seq = side_full[:, :prefix_len].clone() if side_full is not None else None
        generated = []

        if generate_all_agents:
            for _ in range(int(future_len)):
                for ag in range(num_agents):
                    logits = self(seq, agent_seq, side_seq)[:, -1]
                    if multinomial_sampling:
                        next_tok = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                    else:
                        next_tok = logits.argmax(dim=-1, keepdim=True)
                    next_agent = torch.full((seq.shape[0], 1), ag, dtype=agent_seq.dtype, device=seq.device)
                    if side_seq is not None:
                        next_pos = side_seq.shape[1]
                        next_side = side_full[:, next_pos : next_pos + 1].clone()
                        side_seq = torch.cat([side_seq, next_side], dim=1)
                    seq = torch.cat([seq, next_tok], dim=1)
                    agent_seq = torch.cat([agent_seq, next_agent], dim=1)
                    if ag == num_agents - 1:
                        generated.append(next_tok)
            return torch.cat(generated, dim=1)

        for _ in range(int(future_len)):
            pad_neighbors = torch.full(
                (seq.shape[0], num_agents - 1),
                self.pad_index,
                dtype=seq.dtype,
                device=seq.device,
            )
            neighbor_ids = torch.arange(num_agents - 1, dtype=agent_seq.dtype, device=seq.device)
            neighbor_ids = neighbor_ids.unsqueeze(0).expand(seq.shape[0], -1)
            seq_in = torch.cat([seq, pad_neighbors], dim=1)
            agent_in = torch.cat([agent_seq, neighbor_ids], dim=1)
            if side_seq is not None:
                side_pad = torch.zeros(seq.shape[0], num_agents - 1, side_seq.shape[-1], device=seq.device)
                side_in = torch.cat([side_seq, side_pad], dim=1)
            else:
                side_in = None

            logits = self(seq_in, agent_in, side_in)[:, -1]
            if multinomial_sampling:
                next_tok = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            else:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            ego_ids = torch.full((seq.shape[0], 1), num_agents - 1, dtype=agent_seq.dtype, device=seq.device)
            seq = torch.cat([seq_in, next_tok], dim=1)
            agent_seq = torch.cat([agent_in, ego_ids], dim=1)
            if side_seq is not None:
                side_ego = torch.full((seq.shape[0], 1, side_seq.shape[-1]), -1.0, device=seq.device)
                side_seq = torch.cat([side_in, side_ego], dim=1)
            generated.append(next_tok)
        return torch.cat(generated, dim=1)
