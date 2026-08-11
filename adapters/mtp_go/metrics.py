"""Metrics and report tables, matching `neighformer/src/metrics.py` exactly.

The definitions below are deliberately identical to NeighFormer's so that
MTP-GO numbers can be put next to EncDecFormer numbers without a conversion:

    ade  = mean_samples( mean_t ||pred - y|| )
    fde  = mean_samples( ||pred_T - y_T|| )
    rmse = mean_samples( sqrt( mean_t ||pred - y||^2 ) )      <- per-sample sqrt
    rmse@Ns = sqrt( sum_samples ||pred_i - y_i||^2 / n ),  i = int(N * hz) - 1

`hz` is the reporting convention inherited from NeighFormer configs (3.0),
which is not exactly 1/dt (dt = 0.32 s -> 3.125 Hz). The index formula is kept
identical on purpose; the true time of each reported second is recorded
alongside it as `rmse_Ns_actual_seconds`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

EVAL_SECONDS = (1, 2, 3, 4, 5)


# ──────────────────────────────────────────────────────────────────────────────
# Per-sample metrics (call .mean() for a batch average)
# ──────────────────────────────────────────────────────────────────────────────

def ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(B, T, 2) -> (B,)"""
    return torch.norm(pred - target, dim=-1).mean(dim=-1)


def fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(B, T, 2) -> (B,)"""
    return torch.norm(pred[:, -1, :] - target[:, -1, :], dim=-1)


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(B, T, 2) -> (B,)"""
    return torch.norm(pred - target, dim=-1).pow(2).mean(dim=-1).sqrt()


# ──────────────────────────────────────────────────────────────────────────────
# Accumulator
# ──────────────────────────────────────────────────────────────────────────────

class MetricAccumulator:
    """Exact (sample-weighted) accumulation over an arbitrary number of batches."""

    def __init__(self, dt: float, hz: float = 3.0) -> None:
        self.dt = float(dt)
        self.hz = float(hz)
        self.n = 0
        self.sum_ade = 0.0
        self.sum_fde = 0.0
        self.sum_rmse = 0.0
        self.sum_min_ade = 0.0
        self.sum_min_fde = 0.0
        self.sum_nll = 0.0
        self.n_nll = 0
        self._step_abs: np.ndarray | None = None   # (T,) sum of L2 per step
        self._step_sq: np.ndarray | None = None    # (T,) sum of squared L2 per step
        # {label: [sum_ade, sum_fde, sum_rmse, count]}
        self.event_stats: dict[str, list] = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
        self.state_stats: dict[str, list] = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
        self.has_scenario = False

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        all_modes: torch.Tensor | None = None,
        nll: float | None = None,
        labels: list[dict[str, str] | None] | None = None,
    ) -> None:
        """
        pred      : (B, T, 2) most-likely trajectory
        target    : (B, T, 2)
        all_modes : (B, T, m, 2) every mixture component, for minADE/minFDE
        labels    : per-sample {"event_label": ..., "state_label": ...} or None
        """
        b = pred.shape[0]
        dist = torch.norm(pred - target, dim=-1)             # (B, T)
        a = dist.mean(dim=-1)                                # (B,)
        f = dist[:, -1]                                      # (B,)
        r = dist.pow(2).mean(dim=-1).sqrt()                  # (B,)

        self.sum_ade += float(a.sum())
        self.sum_fde += float(f.sum())
        self.sum_rmse += float(r.sum())
        self.n += b

        step_abs = dist.double().sum(dim=0).cpu().numpy()
        step_sq = dist.double().pow(2).sum(dim=0).cpu().numpy()
        if self._step_abs is None:
            self._step_abs = np.zeros_like(step_abs)
            self._step_sq = np.zeros_like(step_sq)
        self._step_abs += step_abs
        self._step_sq += step_sq

        if all_modes is not None:
            mode_dist = torch.norm(all_modes - target.unsqueeze(2), dim=-1)   # (B, T, m)
            best = mode_dist.mean(dim=1).argmin(dim=-1)                       # (B,)
            best_dist = mode_dist[torch.arange(b, device=pred.device), :, best]
            self.sum_min_ade += float(best_dist.mean(dim=-1).sum())
            self.sum_min_fde += float(best_dist[:, -1].sum())

        if nll is not None and math.isfinite(nll):
            self.sum_nll += float(nll)
            self.n_nll += 1

        if labels is not None:
            a_np, f_np, r_np = a.cpu().numpy(), f.cpu().numpy(), r.cpu().numpy()
            for i, lab in enumerate(labels):
                if lab is None or i >= b:
                    continue
                self.has_scenario = True
                for acc, key in ((self.event_stats, "event_label"),
                                 (self.state_stats, "state_label")):
                    name = lab.get(key) or "unknown"
                    acc[name][0] += float(a_np[i])
                    acc[name][1] += float(f_np[i])
                    acc[name][2] += float(r_np[i])
                    acc[name][3] += 1

    def result(self) -> dict[str, Any]:
        if self.n == 0 or self._step_abs is None:
            return {"n_samples": 0}
        n = float(self.n)
        step_rmse = np.sqrt(self._step_sq / n)
        step_ade = self._step_abs / n
        horizon = [round((t + 1) * self.dt, 4) for t in range(len(step_ade))]

        out: dict[str, Any] = {
            "n_samples": int(self.n),
            "ade": self.sum_ade / n,
            "fde": self.sum_fde / n,
            "rmse": self.sum_rmse / n,
            "dt_seconds": self.dt,
            "eval_hz": self.hz,
            "horizon_seconds": horizon,
            "step_rmse": [float(v) for v in step_rmse],
            "step_ade": [float(v) for v in step_ade],
        }
        if self.sum_min_ade:
            out["min_ade"] = self.sum_min_ade / n
            out["min_fde"] = self.sum_min_fde / n
        if self.n_nll:
            out["nll"] = self.sum_nll / self.n_nll

        for sec in EVAL_SECONDS:
            idx = int(sec * self.hz) - 1
            if 0 <= idx < len(step_rmse):
                out[f"rmse_{sec}s"] = float(step_rmse[idx])
                out[f"rmse_{sec}s_actual_seconds"] = horizon[idx]
            else:
                out[f"rmse_{sec}s"] = float("nan")
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Scenario labels
# ──────────────────────────────────────────────────────────────────────────────

def load_scenario_labels(path: Path) -> dict[tuple[int, int, int], dict[str, Any]] | None:
    """scenario_labels.csv -> {(recordingId, trackId, t0_frame): {...}}"""
    import pandas as pd

    path = Path(path)
    if not path.exists():
        print(f"[WARN] scenario_labels not found: {path} -> scenario breakdown disabled")
        return None
    df = pd.read_csv(path)
    required = {"recordingId", "trackId", "t0_frame"}
    if required - set(df.columns):
        print(f"[WARN] scenario_labels missing {required - set(df.columns)} -> disabled")
        return None
    if "event_label" not in df.columns and "state_label" not in df.columns:
        print("[WARN] scenario_labels has no event_label/state_label -> disabled")
        return None

    lut: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in df.itertuples(index=False):
        key = (int(row.recordingId), int(row.trackId), int(row.t0_frame))
        lut[key] = {
            "event_label": getattr(row, "event_label", None),
            "state_label": getattr(row, "state_label", None),
        }
    return lut


class SampleMetaLookup:
    """sample index -> (recordingId, trackId, frame) -> scenario label."""

    def __init__(self, data_dir: Path, labels_lut: dict | None) -> None:
        self.labels_lut = labels_lut
        self.meta: dict[str, np.ndarray] | None = None
        if labels_lut is None:
            return
        files = {
            "rec": data_dir / "meta_recordingId.npy",
            "track": data_dir / "meta_trackId.npy",
            "frame": data_dir / "meta_frame.npy",
        }
        if not all(p.exists() for p in files.values()):
            print(f"[WARN] meta_*.npy missing in {data_dir} -> scenario breakdown disabled")
            self.labels_lut = None
            return
        self.meta = {k: np.load(p, mmap_mode="r") for k, p in files.items()}

    @property
    def enabled(self) -> bool:
        return self.labels_lut is not None and self.meta is not None

    def lookup(self, sample_indices: np.ndarray) -> list[dict[str, str] | None] | None:
        if not self.enabled:
            return None
        out: list[dict[str, str] | None] = []
        for i in sample_indices:
            key = (
                int(self.meta["rec"][i]),
                int(self.meta["track"][i]),
                int(self.meta["frame"][i]),
            )
            out.append(self.labels_lut.get(key))
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Report tables (same layout as neighformer/evaluate.py)
# ──────────────────────────────────────────────────────────────────────────────

def _sep(widths, left="+", mid="+", right="+", fill="-") -> str:
    return left + mid.join(fill * w for w in widths) + right


def print_metrics(results: dict[str, Any]) -> None:
    c1 = 15
    ws = [c1, c1, c1]
    print()
    print(_sep(ws))
    print(f"|{'ADE':^{c1}}|{'FDE':^{c1}}|{'RMSE':^{c1}}|")
    print(_sep(ws))
    print(f"|{results['ade']:^{c1}.4f}|{results['fde']:^{c1}.4f}|{results['rmse']:^{c1}.4f}|")
    print(_sep(ws))

    c2 = 9
    inner = c2 * len(EVAL_SECONDS) + (len(EVAL_SECONDS) - 1)
    print()
    print(f"+{'-' * inner}+")
    print(f"|{'RMSE':^{inner}}|")
    print(_sep([c2] * len(EVAL_SECONDS)))
    print("|" + "|".join(f"{'@' + str(s) + 's':^{c2}}" for s in EVAL_SECONDS) + "|")
    print(_sep([c2] * len(EVAL_SECONDS)))
    vals = [results.get(f"rmse_{s}s", float("nan")) for s in EVAL_SECONDS]
    print("|" + "|".join(f"{v:^{c2}.4f}" for v in vals) + "|")
    print(_sep([c2] * len(EVAL_SECONDS)))

    extra = [(k, results[k]) for k in ("min_ade", "min_fde", "nll") if k in results]
    if extra:
        print("\n  " + "   ".join(f"{k}={v:.4f}" for k, v in extra))


def print_scenario_results(stats: dict[str, list], label_type: str) -> None:
    if not stats:
        return
    rows = sorted(stats.items(), key=lambda x: (x[0] == "unknown", x[0]))
    c_lbl = max(max(len(lbl) for lbl, _ in rows), len(label_type)) + 2
    c_n, c_m = 9, 11
    ws = [c_lbl, c_n, c_m, c_m, c_m]

    print(f"\n====== Scenario Results [{label_type}] ======")
    print(_sep(ws))
    print(f"|{label_type:^{c_lbl}}|{'n':^{c_n}}|{'ADE':^{c_m}}|{'FDE':^{c_m}}|{'RMSE':^{c_m}}|")
    print(_sep(ws))
    total_n = sum(v[3] for v in stats.values())
    for lbl, (sa, sf, sr, n) in rows:
        if n == 0:
            continue
        print(f"|{lbl:^{c_lbl}}|{n:^{c_n},}|{sa/n:^{c_m}.4f}|{sf/n:^{c_m}.4f}|{sr/n:^{c_m}.4f}|")
    print(_sep(ws))
    n_all = max(1, total_n)
    print(
        f"|{'Total':^{c_lbl}}|{total_n:^{c_n},}"
        f"|{sum(v[0] for v in stats.values())/n_all:^{c_m}.4f}"
        f"|{sum(v[1] for v in stats.values())/n_all:^{c_m}.4f}"
        f"|{sum(v[2] for v in stats.values())/n_all:^{c_m}.4f}|"
    )
    print(_sep(ws))


def print_latency(lat: dict[str, float], batch_size: int, warmup: int, iters: int) -> None:
    c = 15
    ws = [c, c, c]
    print()
    print(f"  Batch size : {batch_size}   Warmup : {warmup:,}   Measurement : {iters:,}")
    print()
    print(_sep(ws))
    print(f"|{'Avg (ms)':^{c}}|{'Min (ms)':^{c}}|{'Max (ms)':^{c}}|")
    print(_sep(ws))
    print(f"|{lat['avg_ms']:^{c}.2f}|{lat['min_ms']:^{c}.2f}|{lat['max_ms']:^{c}.2f}|")
    print(_sep(ws))
