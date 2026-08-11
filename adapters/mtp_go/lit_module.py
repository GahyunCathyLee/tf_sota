"""Ego-only evaluation on top of the upstream MTP-GO Lightning module.

Upstream computes metrics over every vehicle in the scene because its
preprocessing has ground-truth futures for all of them. The NeighFormer schema
only stores the ego future, so all metrics here are restricted to the ego node
(node 0 of every scene graph, i.e. ``data.ptr[:-1]``).

Training is left untouched: ``training_step``/``encode_decode`` come straight
from upstream ``base_mdn.LitEncoderDecoder``.
"""

from __future__ import annotations

import math
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
def evaluate(model, loader, device: torch.device, dt: float, max_batches: int | None = None) -> dict[str, Any]:
    """Exact (sample-weighted) ADE/FDE/RMSE over the ego nodes of a loader."""
    from losses import NLLMDNLoss  # upstream loss, on sys.path

    nll_fn = NLLMDNLoss()
    model = model.to(device)
    model.eval()

    n_scenes = 0
    sum_err: torch.Tensor | None = None     # (T,) sum of L2 error
    sum_sq: torch.Tensor | None = None      # (T,) sum of squared L2 error
    sum_min_err: torch.Tensor | None = None  # (T,) sum of best-of-k L2 error
    nll_total = 0.0
    nll_batches = 0

    for bi, data in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        data = data.to(device)
        states, covs, pis, tgt, mask = model._ego_prediction(data)
        b = tgt.shape[0]

        best = torch.argmax(pis, dim=-1)
        ml = states[torch.arange(b, device=states.device), :, best]
        err = torch.linalg.norm(ml - tgt, dim=-1)                    # (B, T)

        all_err = torch.linalg.norm(states - tgt.unsqueeze(2), dim=-1)  # (B, T, m)
        min_err = all_err[torch.arange(b, device=states.device), :, all_err.mean(dim=1).argmin(dim=-1)]

        if sum_err is None:
            sum_err = torch.zeros(err.shape[1], dtype=torch.float64, device=err.device)
            sum_sq = torch.zeros_like(sum_err)
            sum_min_err = torch.zeros_like(sum_err)
        sum_err += err.double().sum(dim=0)
        sum_sq += (err.double() ** 2).sum(dim=0)
        sum_min_err += min_err.double().sum(dim=0)
        n_scenes += b

        nll_val = nll_fn(states, covs, pis, tgt, mask)
        if torch.isfinite(nll_val):
            nll_total += float(nll_val)
            nll_batches += 1

    if sum_err is None or n_scenes == 0:
        return {"num_scenes": 0}

    mean_err = (sum_err / n_scenes).cpu()
    mean_sq = (sum_sq / n_scenes).cpu()
    mean_min = (sum_min_err / n_scenes).cpu()
    horizon = [round((t + 1) * dt, 4) for t in range(int(mean_err.shape[0]))]
    step_rmse = torch.sqrt(mean_sq)

    metrics: dict[str, Any] = {
        "num_scenes": int(n_scenes),
        "dt_seconds": dt,
        "horizon_seconds": horizon,
        "ADE": float(mean_err.mean()),
        "FDE": float(mean_err[-1]),
        "RMSE": float(torch.sqrt(mean_sq.mean())),
        "minADE_k": float(mean_min.mean()),
        "minFDE_k": float(mean_min[-1]),
        "NLL": (nll_total / nll_batches) if nll_batches else float("nan"),
        "step_RMSE": [float(v) for v in step_rmse],
        "step_ADE": [float(v) for v in mean_err],
    }

    # RMSE at whole-second marks, snapped to the nearest available step.
    for sec in range(1, 6):
        if not horizon:
            break
        step = min(range(len(horizon)), key=lambda t: abs(horizon[t] - sec))
        if abs(horizon[step] - sec) > dt:  # that second is outside the horizon
            continue
        metrics[f"RMSE@{sec}s"] = float(step_rmse[step])
        metrics[f"RMSE@{sec}s_actual_seconds"] = horizon[step]

    if math.isnan(metrics["NLL"]):
        metrics.pop("NLL")
    return metrics
