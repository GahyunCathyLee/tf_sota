"""Mask-aware evaluation on top of the upstream MTP-GO Lightning module.

Upstream computes metrics over every vehicle in the scene because its
preprocessing has ground-truth futures for all of them. Canonical NeighFormer
arrays only store the ego future, but PAR-style ``y_nb.npy`` arrays provide
fixed-ID neighbour futures. Metrics are computed over every node with at least
one valid future target.

Training is left untouched: ``training_step``/``encode_decode`` come straight
from upstream ``base_mdn.LitEncoderDecoder``.
"""

from __future__ import annotations

from typing import Any

import torch


def make_lit_module_class(base_cls: type) -> type:
    class TargetLitEncoderDecoder(base_cls):  # type: ignore[valid-type, misc]
        def _target_predictions(self, data):
            all_states, all_Ps, mixture_coeffs, dec_mask, target = self.encode_decode(data, 0)
            mask_all = dec_mask.bool()
            keep = mask_all.any(dim=-1)
            states = all_states[keep][..., :2]         # (B, T, m, 2)
            covs = all_Ps[keep][..., :2, :2]           # (B, T, m, 2, 2)
            pis = mixture_coeffs[keep]                 # (B, m)
            tgt = target[keep][..., :2]                # (B, T, 2)
            mask = mask_all[keep]                      # (B, T)
            return states, covs, pis, tgt, mask

        def validation_step(self, data, batch_idx):
            from losses import NLLMDNLoss  # upstream loss, on sys.path

            states, covs, pis, tgt, mask = self._target_predictions(data)
            batch_size = tgt.shape[0]

            nll = NLLMDNLoss()(states, covs, pis, tgt, mask)
            best = torch.argmax(pis, dim=-1)
            ml = states[torch.arange(batch_size, device=states.device), :, best]  # (B, T, 2)
            err = torch.linalg.norm(ml - tgt, dim=-1)                              # (B, T)
            valid = mask.float()
            counts = valid.sum(dim=-1).clamp_min(1.0)
            last_valid = mask.long().sum(dim=-1) - 1

            ade = (err * valid).sum(dim=-1).div(counts).mean()
            fde = err[torch.arange(batch_size, device=states.device), last_valid].mean()
            self.log_dict(
                {"val_ade": ade, "val_fde": fde, "val_nll": nll},
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
            return ade

        def test_step(self, data, batch_idx):
            states, _, pis, tgt, mask = self._target_predictions(data)
            batch_size = tgt.shape[0]
            best = torch.argmax(pis, dim=-1)
            ml = states[torch.arange(batch_size, device=states.device), :, best]
            err = torch.linalg.norm(ml - tgt, dim=-1)
            valid = mask.float()
            counts = valid.sum(dim=-1).clamp_min(1.0)
            last_valid = mask.long().sum(dim=-1) - 1
            self.log_dict(
                {
                    "test_ade": (err * valid).sum(dim=-1).div(counts).mean(),
                    "test_fde": err[torch.arange(batch_size, device=states.device), last_valid].mean(),
                },
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )

    return TargetLitEncoderDecoder


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: torch.device,
    dt: float,
    hz: float = 3.0,
    meta_lookup: Any = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Exact ADE/FDE/RMSE over every node with a valid future target.

    Metric definitions match `neighformer/src/metrics.py` — see
    `adapters/mtp_go/metrics.py`.
    """
    from losses import NLLMDNLoss  # upstream loss, on sys.path

    from adapters.mtp_go.metrics import MetricAccumulator

    nll_fn = NLLMDNLoss()
    model = model.to(device)
    model.eval()
    acc = MetricAccumulator(dt=dt, hz=hz)

    iterator = loader
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(loader, desc="Evaluating", dynamic_ncols=True, leave=True)
        except ImportError:
            pass

    for data in iterator:
        data = data.to(device)
        states, covs, pis, tgt, mask = model._target_predictions(data)
        b = tgt.shape[0]

        best = torch.argmax(pis, dim=-1)
        ml = states[torch.arange(b, device=states.device), :, best]   # (B, T, 2)

        nll_val = nll_fn(states, covs, pis, tgt, mask)
        labels = None
        if meta_lookup is not None and getattr(meta_lookup, "enabled", False):
            scene_labels = meta_lookup.lookup(data.sample_index.view(-1).cpu().numpy())
            target_rows = data.tar_real_mask[..., :2].all(dim=-1).any(dim=-1)
            node_graph = data.batch[target_rows].detach().cpu().numpy().reshape(-1)
            labels = [scene_labels[int(i)] if scene_labels is not None else None for i in node_graph]

        acc.update(
            ml,
            tgt,
            all_modes=states,
            valid_mask=mask,
            nll=float(nll_val) if torch.isfinite(nll_val) else None,
            labels=labels,
        )
        if progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(ADE=f"{acc.sum_ade / max(1, acc.n):.4f}")

    result = acc.result()
    if acc.has_scenario:
        result["_event_stats"] = dict(acc.event_stats)
        result["_state_stats"] = dict(acc.state_stats)
    return result
