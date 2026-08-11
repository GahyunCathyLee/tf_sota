"""Ego-only evaluation on top of the upstream MTP-GO Lightning module.

Upstream computes metrics over every vehicle in the scene because its
preprocessing has ground-truth futures for all of them. The NeighFormer schema
only stores the ego future, so all metrics here are restricted to the ego node
(node 0 of every scene graph, i.e. ``data.ptr[:-1]``).

Training is left untouched: ``training_step``/``encode_decode`` come straight
from upstream ``base_mdn.LitEncoderDecoder``.
"""

from __future__ import annotations

from typing import Any

import torch


def make_lit_module_class(base_cls: type) -> type:
    class EgoLitEncoderDecoder(base_cls):  # type: ignore[valid-type, misc]
        def _ego_prediction(self, data):
            all_states, all_Ps, mixture_coeffs, dec_mask, target = self.encode_decode(data, 0)
            ego = data.ptr[:-1]
            states = all_states[ego][..., :2]          # (B, T, m, 2)
            covs = all_Ps[ego][..., :2, :2]            # (B, T, m, 2, 2)
            pis = mixture_coeffs[ego]                  # (B, m)
            tgt = target[ego][..., :2]                 # (B, T, 2)
            mask = dec_mask[ego]                       # (B, T)
            return states, covs, pis, tgt, mask

        def validation_step(self, data, batch_idx):
            from losses import NLLMDNLoss  # upstream loss, on sys.path

            states, covs, pis, tgt, mask = self._ego_prediction(data)
            batch_size = tgt.shape[0]

            nll = NLLMDNLoss()(states, covs, pis, tgt, mask)
            best = torch.argmax(pis, dim=-1)
            ml = states[torch.arange(batch_size, device=states.device), :, best]  # (B, T, 2)
            err = torch.linalg.norm(ml - tgt, dim=-1)                              # (B, T)

            ade = err.mean()
            fde = err[:, -1].mean()
            self.log_dict(
                {"val_ade": ade, "val_fde": fde, "val_nll": nll},
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
            return ade

        def test_step(self, data, batch_idx):
            states, _, pis, tgt, _ = self._ego_prediction(data)
            batch_size = tgt.shape[0]
            best = torch.argmax(pis, dim=-1)
            ml = states[torch.arange(batch_size, device=states.device), :, best]
            err = torch.linalg.norm(ml - tgt, dim=-1)
            self.log_dict(
                {"test_ade": err.mean(), "test_fde": err[:, -1].mean()},
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )

    return EgoLitEncoderDecoder


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
    """Exact (sample-weighted) ADE/FDE/RMSE over the ego nodes of a loader.

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
        states, covs, pis, tgt, mask = model._ego_prediction(data)
        b = tgt.shape[0]

        best = torch.argmax(pis, dim=-1)
        ml = states[torch.arange(b, device=states.device), :, best]   # (B, T, 2)

        nll_val = nll_fn(states, covs, pis, tgt, mask)
        labels = None
        if meta_lookup is not None and getattr(meta_lookup, "enabled", False):
            labels = meta_lookup.lookup(data.sample_index.view(-1).cpu().numpy())

        acc.update(
            ml,
            tgt,
            all_modes=states,
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
