"""Trajectory metrics for ego-only and multi-agent evaluation.

For single-agent predictors this reduces to the NeighFormer convention:

    ade  = mean_samples( mean_t ||pred - y|| )
    fde  = mean_samples( ||pred_T - y_T|| )
    rmse = mean_samples( sqrt( mean_t ||pred - y||^2 ) )      <- per-sample sqrt
    rmse@Ns = sqrt( sum_samples ||pred_i - y_i||^2 / n ),  i = int(N * hz) - 1

For multi-agent predictors, each valid target agent trajectory is treated as
one scored sample and metrics are averaged over all scored agents. Partially
observed futures are handled with a timestep mask: ADE/RMSE use valid future
steps and FDE uses the last valid future step for that agent.

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
        self.n_min = 0
        self.sum_nll = 0.0
        self.n_nll = 0
        self._step_abs: np.ndarray | None = None   # (T,) sum of L2 per step
        self._step_sq: np.ndarray | None = None    # (T,) sum of squared L2 per step
        self._step_count: np.ndarray | None = None  # (T,) valid agents per step
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
        valid_mask: torch.Tensor | None = None,
        nll: float | None = None,
        labels: list[dict[str, str] | None] | None = None,
    ) -> None:
        """
        pred      : (B, T, 2) most-likely trajectory
        target    : (B, T, 2)
        all_modes : (B, T, m, 2) every mixture component, for minADE/minFDE
        valid_mask: (B, T) valid target timesteps; defaults to all-valid
        labels    : per-sample {"event_label": ..., "state_label": ...} or None
        """
        if pred.ndim != 3 or target.ndim != 3:
            raise ValueError(f"pred/target must be (B, T, 2), got {pred.shape} and {target.shape}")
        if pred.shape != target.shape:
            raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")

        device = pred.device
        if valid_mask is None:
            valid_mask = torch.ones(pred.shape[:2], dtype=torch.bool, device=device)
        else:
            valid_mask = valid_mask.to(device=device, dtype=torch.bool)
        valid_agent = valid_mask.any(dim=-1)
        if not bool(valid_agent.any()):
            return

        pred = pred[valid_agent]
        target = target[valid_agent]
        valid_mask = valid_mask[valid_agent]
        if all_modes is not None:
            all_modes = all_modes[valid_agent]
        if labels is not None:
            labels = [lab for lab, keep in zip(labels, valid_agent.detach().cpu().tolist()) if keep]

        b = pred.shape[0]
        dist = torch.norm(pred - target, dim=-1)             # (B, T)
        valid_f = valid_mask.float()
        counts = valid_f.sum(dim=-1).clamp_min(1.0)
        masked_dist = dist * valid_f
        a = masked_dist.sum(dim=-1) / counts                 # (B,)
        r = (dist.pow(2) * valid_f).sum(dim=-1).div(counts).sqrt()
        last_valid = valid_mask.long().sum(dim=-1) - 1
        f = dist[torch.arange(b, device=device), last_valid]  # (B,)

        self.sum_ade += float(a.sum())
        self.sum_fde += float(f.sum())
        self.sum_rmse += float(r.sum())
        self.n += b

        step_abs = (dist.double() * valid_mask.double()).sum(dim=0).cpu().numpy()
        step_sq = (dist.double().pow(2) * valid_mask.double()).sum(dim=0).cpu().numpy()
        step_count = valid_mask.double().sum(dim=0).cpu().numpy()
        if self._step_abs is None:
            self._step_abs = np.zeros_like(step_abs)
            self._step_sq = np.zeros_like(step_sq)
            self._step_count = np.zeros_like(step_count)
        self._step_abs += step_abs
        self._step_sq += step_sq
        self._step_count += step_count

        if all_modes is not None:
            mode_dist = torch.norm(all_modes - target.unsqueeze(2), dim=-1)   # (B, T, m)
            mode_ade = (mode_dist * valid_f.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)
            best = mode_ade.argmin(dim=-1)                                    # (B,)
            best_dist = mode_dist[torch.arange(b, device=device), :, best]
            final_mode_dist = mode_dist[torch.arange(b, device=device), last_valid]
            self.sum_min_ade += float((best_dist * valid_f).sum(dim=-1).div(counts).sum())
            self.sum_min_fde += float(final_mode_dist.min(dim=-1).values.sum())
            self.n_min += b

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
        if self.n == 0 or self._step_abs is None or self._step_count is None:
            return {"n_samples": 0}
        n = float(self.n)
        step_rmse = np.full_like(self._step_sq, np.nan, dtype=np.float64)
        step_ade = np.full_like(self._step_abs, np.nan, dtype=np.float64)
        valid_steps = self._step_count > 0
        step_rmse[valid_steps] = np.sqrt(self._step_sq[valid_steps] / self._step_count[valid_steps])
        step_ade[valid_steps] = self._step_abs[valid_steps] / self._step_count[valid_steps]
        horizon = [round((t + 1) * self.dt, 4) for t in range(len(step_ade))]

        out: dict[str, Any] = {
            "n_samples": int(self.n),
            "n_agent_trajectories": int(self.n),
            "ade": self.sum_ade / n,
            "fde": self.sum_fde / n,
            "rmse": self.sum_rmse / n,
            "dt_seconds": self.dt,
            "eval_hz": self.hz,
            "horizon_seconds": horizon,
            "step_rmse": [float(v) for v in step_rmse],
            "step_ade": [float(v) for v in step_ade],
            "step_valid_count": [int(v) for v in self._step_count],
        }
        if self.n_min:
            n_min = float(self.n_min)
            out["min_ade"] = self.sum_min_ade / n_min
            out["min_fde"] = self.sum_min_fde / n_min
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
        self.meta_len = 0
        self._warned_bounds = False
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
        lengths = {k: int(v.shape[0]) for k, v in self.meta.items()}
        self.meta_len = min(lengths.values())
        if len(set(lengths.values())) > 1:
            print(
                "[WARN] meta_*.npy length mismatch "
                f"{lengths}; scenario lookup will use the first {self.meta_len:,} rows"
            )

    @property
    def enabled(self) -> bool:
        return self.labels_lut is not None and self.meta is not None and self.meta_len > 0

    def warn_if_incomplete(self, sample_indices: np.ndarray, context: str = "split") -> None:
        """Warn once if a split contains sample ids not covered by meta arrays."""
        if not self.enabled or self._warned_bounds:
            return
        sample_indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
        if sample_indices.size == 0:
            return
        bad = (sample_indices < 0) | (sample_indices >= self.meta_len)
        if not bad.any():
            return
        bad_values = sample_indices[bad]
        hi = self.meta_len - 1
        print(
            f"[WARN] scenario meta covers sample indices 0..{hi:,}, but {context} "
            f"contains {int(bad.sum()):,}/{sample_indices.size:,} out-of-range "
            f"indices (min={int(bad_values.min()):,}, max={int(bad_values.max()):,}). "
            "Those samples remain in overall metrics but are skipped in scenario breakdown."
        )
        self._warned_bounds = True

    def lookup(self, sample_indices: np.ndarray) -> list[dict[str, str] | None] | None:
        if not self.enabled:
            return None
        out: list[dict[str, str] | None] = []
        for i in sample_indices:
            i = int(i)
            if i < 0 or i >= self.meta_len:
                if not self._warned_bounds:
                    print(
                        f"[WARN] sample index {i:,} is outside scenario meta range "
                        f"0..{self.meta_len - 1:,}; skipping out-of-range scenario labels"
                    )
                    self._warned_bounds = True
                out.append(None)
                continue
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
