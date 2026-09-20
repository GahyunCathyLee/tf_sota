#!/usr/bin/env python3
"""Forensic audit of canonical final-I provenance.

This script intentionally does not define a new production policy. It compares
canonical ``x_nb[..., 9]`` against repository-supported candidate final-I
policies after the base state channels have been recovered.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from data.multiagent import preprocess as ma_pre  # noqa: E402
from data.multiagent.interaction import (  # noqa: E402
    DEFAULT_GAMMA,
    DEFAULT_TOP_N,
    GLOBAL_SLOT_WEIGHTS,
    SLOT_WEIGHTS_BY_LANE_LEVEL,
    SLOT_WEIGHTS_POST_LC,
    SLOT_WEIGHTS_PRE_LC,
    TOPN_SLOT_PRIORITY,
    compute_importance,
    slot_weight,
    target_context,
)

TOL = 1e-5
POLICIES = (
    "NO_BOOST",
    "NO_BOOST_TOP3",
    "GLOBAL",
    "GLOBAL_TOP3",
    "CONDITIONAL",
    "CONDITIONAL_TOP3",
)
CONTEXT_NAMES = {
    -1: "global",
    0: "LF_leftmost",
    1: "LF_middle",
    2: "LF_rightmost",
    10: "pre_LC0",
    11: "pre_LC1",
    12: "pre_LC2",
    13: "pre_LC3",
    20: "post_LC0",
    21: "post_LC1",
    22: "post_LC2",
    23: "post_LC3",
}


def _summ(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _policy_label(exact: list[str]) -> str:
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return "AMBIGUOUS"
    return "UNEXPLAINED"


def _context_code(kind: str, idx: int) -> int:
    if kind == "lane_following" and 0 <= idx <= 2:
        return int(idx)
    if kind == "pre_lc" and 0 <= idx <= 3:
        return 10 + int(idx)
    if kind == "post_lc" and 0 <= idx <= 3:
        return 20 + int(idx)
    return -1


def _build_context_codes(
    paths: ma_pre.DatasetPaths,
    arrays: dict[str, np.ndarray],
    target_hz: float,
    source_indices: np.ndarray,
) -> np.ndarray:
    n = int(source_indices.size)
    hist = int(np.load(paths.canonical_dir / "x_nb.npy", mmap_mode="r").shape[1])
    codes = np.full((n, hist), -1, dtype=np.int16)
    ctx_cache: dict[int, dict[str, Any]] = {}
    by_rec: dict[int, np.ndarray] = {}
    meta_rec = np.asarray(arrays["meta_recordingId"][source_indices])
    for rec_id in np.unique(meta_rec):
        by_rec[int(rec_id)] = np.flatnonzero(meta_rec == rec_id)

    for rec_i, (rec_id, indices) in enumerate(sorted(by_rec.items()), start=1):
        ctx = ctx_cache.get(rec_id)
        if ctx is None:
            ctx = (
                ma_pre._build_highd_context(paths.raw_dir, rec_id, target_hz)
                if paths.dataset == "highD"
                else ma_pre._build_exid_context(paths.raw_dir, rec_id, target_hz)
            )
            ctx_cache[rec_id] = ctx
        step = int(ctx["step"])
        for local_idx in indices:
            source_idx = int(source_indices[local_idx])
            ego_tid = int(arrays["meta_trackId"][source_idx])
            t0 = int(arrays["meta_frame"][source_idx])
            hist_frames = [t0 - (hist - 1 - t) * step for t in range(hist)]
            lane_ids = np.full(hist, -1, dtype=np.int32)
            levels = np.full(hist, -1, dtype=np.int16)
            row_map = ctx["frame_to_row"].get(ego_tid, {})
            for ti, frame in enumerate(hist_frames):
                row = row_map.get(int(frame))
                if row is not None:
                    lane_ids[ti] = int(ctx["lane_id"][row])
                    levels[ti] = int(ctx["lane_level"][row])
            for ti in range(hist):
                codes[local_idx, ti] = _context_code(*target_context(levels, ti, lane_ids=lane_ids))
        print(f"[context] {paths.dataset} recording {rec_id} ({rec_i}/{len(by_rec)}) samples={len(indices)}", flush=True)
    return codes


def _conditional_weights(codes: np.ndarray) -> np.ndarray:
    n, hist = codes.shape
    out = np.zeros((n, hist, 8), dtype=np.float32)
    for code in np.unique(codes):
        mask = codes == code
        if int(code) in (0, 1, 2):
            weights = SLOT_WEIGHTS_BY_LANE_LEVEL[int(code)]
        elif 10 <= int(code) <= 13:
            weights = SLOT_WEIGHTS_PRE_LC[int(code) - 10]
        elif 20 <= int(code) <= 23:
            weights = SLOT_WEIGHTS_POST_LC[int(code) - 20]
        else:
            weights = GLOBAL_SLOT_WEIGHTS
        out[mask] = weights
    return out


def _apply_top3_per_timestep(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(scores, dtype=np.float32)
    bsz, hist, slots = scores.shape
    priority = np.asarray([TOPN_SLOT_PRIORITY[i] for i in range(slots)], dtype=np.int16)
    for b in range(bsz):
        for t in range(hist):
            occupied = np.flatnonzero(valid[b, t])
            if occupied.size == 0:
                continue
            ordered = sorted(occupied.tolist(), key=lambda k: (-float(scores[b, t, k]), int(priority[k])))
            keep = ordered[:DEFAULT_TOP_N]
            out[b, t, keep] = scores[b, t, keep]
    return out


def _apply_top3_current_frame_slots(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(scores, dtype=np.float32)
    bsz, hist, slots = scores.shape
    priority = np.asarray([TOPN_SLOT_PRIORITY[i] for i in range(slots)], dtype=np.int16)
    t0 = hist - 1
    for b in range(bsz):
        occupied = np.flatnonzero(valid[b, t0])
        ordered = sorted(occupied.tolist(), key=lambda k: (-float(scores[b, t0, k]), int(priority[k])))
        keep = set(ordered[:DEFAULT_TOP_N])
        for k in keep:
            out[b, :, k] = np.where(valid[b, :, k], scores[b, :, k], 0.0)
    return out


def _apply_top3_history_positions(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(scores, dtype=np.float32)
    bsz, hist, slots = scores.shape
    priority = np.asarray([TOPN_SLOT_PRIORITY[i] for i in range(slots)], dtype=np.int16)
    for b in range(bsz):
        coords = np.argwhere(valid[b])
        if coords.size == 0:
            continue
        ordered = sorted(coords.tolist(), key=lambda tk: (-float(scores[b, tk[0], tk[1]]), int(priority[tk[1]]), int(tk[0])))
        for t, k in ordered[:DEFAULT_TOP_N]:
            out[b, t, k] = scores[b, t, k]
    return out


def _apply_top3_history_slot_max(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(scores, dtype=np.float32)
    bsz, hist, slots = scores.shape
    priority = np.asarray([TOPN_SLOT_PRIORITY[i] for i in range(slots)], dtype=np.int16)
    for b in range(bsz):
        slot_scores = []
        for k in range(slots):
            if bool(valid[b, :, k].any()):
                slot_scores.append((k, float(scores[b, :, k][valid[b, :, k]].max())))
        ordered = sorted(slot_scores, key=lambda ks: (-ks[1], int(priority[ks[0]])))
        keep = {k for k, _ in ordered[:DEFAULT_TOP_N]}
        for k in keep:
            out[b, :, k] = np.where(valid[b, :, k], scores[b, :, k], 0.0)
    return out


def _candidate_arrays(i_base: np.ndarray, valid: np.ndarray, cond_w: np.ndarray) -> dict[str, np.ndarray]:
    global_w = GLOBAL_SLOT_WEIGHTS.reshape(1, 1, 8)
    no_boost = i_base.astype(np.float32)
    global_scores = np.minimum(i_base * (1.0 + DEFAULT_GAMMA * global_w), 1.0).astype(np.float32)
    cond_scores = np.minimum(i_base * (1.0 + DEFAULT_GAMMA * cond_w), 1.0).astype(np.float32)
    return {
        "NO_BOOST": no_boost,
        "NO_BOOST_TOP3": _apply_top3_per_timestep(no_boost, valid),
        "GLOBAL": global_scores,
        "GLOBAL_TOP3": _apply_top3_per_timestep(global_scores, valid),
        "CONDITIONAL": cond_scores,
        "CONDITIONAL_TOP3": _apply_top3_per_timestep(cond_scores, valid),
        "GLOBAL_TOP3_CURRENT_FRAME_SLOTS": _apply_top3_current_frame_slots(global_scores, valid),
        "GLOBAL_TOP3_HISTORY_POSITIONS": _apply_top3_history_positions(global_scores, valid),
        "GLOBAL_TOP3_HISTORY_SLOT_MAX": _apply_top3_history_slot_max(global_scores, valid),
        "CONDITIONAL_TOP3_CURRENT_FRAME_SLOTS": _apply_top3_current_frame_slots(cond_scores, valid),
        "CONDITIONAL_TOP3_HISTORY_POSITIONS": _apply_top3_history_positions(cond_scores, valid),
        "CONDITIONAL_TOP3_HISTORY_SLOT_MAX": _apply_top3_history_slot_max(cond_scores, valid),
    }


def _load_splits(paths: ma_pre.DatasetPaths, n: int) -> dict[str, np.ndarray]:
    split_id = np.full(n, "unknown", dtype=object)
    masks = {}
    for split in ("train", "val", "test"):
        p = paths.split_dir / f"{split}_indices.npy"
        if p.exists():
            idx = np.load(p).astype(np.int64)
            masks[split] = idx
            split_id[idx] = split
    return {"ids": split_id, "indices": masks}


def _range_summary(labels: list[str], indices: np.ndarray, max_ranges: int = 30) -> dict[str, Any]:
    runs = []
    if len(labels) == 0:
        return {"num_runs": 0, "runs": []}
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append(
                {
                    "start_source_index": int(indices[start]),
                    "end_source_index": int(indices[i - 1]),
                    "label": labels[start],
                    "num_samples": int(i - start),
                }
            )
            start = i
    longest = sorted(runs, key=lambda r: r["num_samples"], reverse=True)[:max_ranges]
    first = runs[:max_ranges]
    return {"num_runs": len(runs), "first_runs": first, "longest_runs": longest}


def audit_dataset(dataset: str, source_root: Path, target_hz: float, chunk_size: int, max_samples: int | None) -> dict[str, Any]:
    start = time.perf_counter()
    paths = ma_pre.DatasetPaths(dataset, source_root)
    arrays = ma_pre._load_canonical_arrays(paths.canonical_dir)
    x_nb = np.load(paths.canonical_dir / "x_nb.npy", mmap_mode="r")
    nb_mask = np.load(paths.canonical_dir / "nb_mask.npy", mmap_mode="r")
    n_total = int(x_nb.shape[0])
    n = min(n_total, int(max_samples)) if max_samples is not None else n_total
    indices = np.arange(n, dtype=np.int64)
    print(f"[audit] {dataset}: samples={n:,}/{n_total:,}", flush=True)

    codes = _build_context_codes(paths, arrays, target_hz, indices)
    splits = _load_splits(paths, n_total)
    split_ids = splits["ids"][:n]
    meta_rec = np.asarray(arrays["meta_recordingId"][:n], dtype=np.int32)
    meta_track = np.asarray(arrays["meta_trackId"][:n], dtype=np.int32)
    meta_frame = np.asarray(arrays["meta_frame"][:n], dtype=np.int32)

    slot_total = 0
    sample_total = 0
    policy_slot_match = Counter()
    policy_sample_exact = Counter()
    policy_abs_sum = Counter()
    policy_zero_match = Counter()
    top3_zero_match = Counter()
    boost_nonzero_match = Counter()
    slot_best = Counter()
    label_by_sample: list[str] = []
    sample_details: list[dict[str, Any]] = []
    recording_counts: dict[int, Counter] = defaultdict(Counter)
    split_counts: dict[str, Counter] = defaultdict(Counter)
    split_slot_counts: dict[str, Counter] = defaultdict(Counter)
    context_counts: dict[str, Counter] = defaultdict(Counter)
    effective_nearest = Counter()
    effective_values: list[float] = []
    effective_examples: list[dict[str, Any]] = []
    ibase_reconstructed = 0
    ibase_reconstruct_total = 0
    top3_scopes = (
        "GLOBAL_TOP3",
        "GLOBAL_TOP3_CURRENT_FRAME_SLOTS",
        "GLOBAL_TOP3_HISTORY_POSITIONS",
        "GLOBAL_TOP3_HISTORY_SLOT_MAX",
        "CONDITIONAL_TOP3",
        "CONDITIONAL_TOP3_CURRENT_FRAME_SLOTS",
        "CONDITIONAL_TOP3_HISTORY_POSITIONS",
        "CONDITIONAL_TOP3_HISTORY_SLOT_MAX",
    )

    for lo in range(0, n, chunk_size):
        hi = min(n, lo + chunk_size)
        xb = np.asarray(x_nb[lo:hi], dtype=np.float32)
        valid = np.asarray(nb_mask[lo:hi], dtype=bool)
        canon_i = xb[..., 9]
        sx = xb[..., 6]
        sy = xb[..., 7]
        i_base = np.exp(-0.1 * (np.abs(sx) ** 1.5) - 0.1 * (sy ** 2.0)).astype(np.float32)
        cond_w = _conditional_weights(codes[lo:hi])
        cand = _candidate_arrays(i_base, valid, cond_w)
        occupied = valid
        occ_count = int(occupied.sum())
        if occ_count == 0:
            label_by_sample.extend(["UNEXPLAINED"] * (hi - lo))
            continue
        slot_total += occ_count
        sample_total += hi - lo

        # I_base audit: canonical I should always be <= plausible boosted/clipped values.
        ibase_reconstruct_total += occ_count
        ibase_reconstructed += int(np.count_nonzero(np.isfinite(i_base[occupied])))

        policy_matches = {}
        for name in POLICIES:
            diff = np.abs(cand[name] - canon_i)
            match = (diff <= TOL) & occupied
            policy_matches[name] = match
            policy_slot_match[name] += int(match.sum())
            policy_abs_sum[name] += float(diff[occupied].sum())
            policy_zero_match[name] += int((((cand[name] > TOL) == (canon_i > TOL)) & occupied).sum())
            per_sample = np.all((match | ~occupied).reshape(hi - lo, -1), axis=1)
            policy_sample_exact[name] += int(per_sample.sum())
            for split in np.unique(split_ids[lo:hi]):
                sm = np.asarray(split_ids[lo:hi] == split)[:, None, None]
                split_slot_counts[str(split)]["slots"] += int(np.count_nonzero(occupied & sm)) if name == POLICIES[0] else 0
                split_slot_counts[str(split)][name] += int(np.count_nonzero(match & sm))

        for name in top3_scopes:
            top3_zero_match[name] += int((((cand[name] > TOL) == (canon_i > TOL)) & occupied).sum())

        for base_name in ("NO_BOOST", "GLOBAL", "CONDITIONAL"):
            nz = (canon_i > TOL) & occupied
            boost_nonzero_match[base_name] += int((np.abs(cand[base_name] - canon_i)[nz] <= TOL).sum())

        stacked_diff = np.stack([np.abs(cand[name] - canon_i) for name in POLICIES], axis=-1)
        best_idx = np.argmin(stacked_diff, axis=-1)
        for pi, name in enumerate(POLICIES):
            slot_best[name] += int(np.count_nonzero((best_idx == pi) & occupied))

        for bi in range(hi - lo):
            exact = []
            sample_occ = occupied[bi]
            if not bool(sample_occ.any()):
                label = "UNEXPLAINED"
            else:
                for name in POLICIES:
                    if bool(np.all(policy_matches[name][bi][sample_occ])):
                        exact.append(name)
                label = _policy_label(exact)
            label_by_sample.append(label)
            recording_counts[int(meta_rec[lo + bi])][label] += 1
            split_counts[str(split_ids[lo + bi])][label] += 1
            if len(sample_details) < 40 and label in {"AMBIGUOUS", "UNEXPLAINED"}:
                sample_details.append(
                    {
                        "source_index": int(lo + bi),
                        "recordingId": int(meta_rec[lo + bi]),
                        "ego_trackId": int(meta_track[lo + bi]),
                        "t0_frame": int(meta_frame[lo + bi]),
                        "label": label,
                        "exact_policies": exact,
                        "occupied_slots": int(sample_occ.sum()),
                        "policy_match_counts": {
                            name: int(policy_matches[name][bi][sample_occ].sum()) for name in POLICIES
                        },
                    }
                )

        context_flat = codes[lo:hi][..., None].repeat(8, axis=2)
        for code in np.unique(context_flat[occupied]):
            ctx_name = CONTEXT_NAMES.get(int(code), f"context_{int(code)}")
            m = occupied & (context_flat == code)
            context_counts[ctx_name]["slots"] += int(m.sum())
            for name in POLICIES:
                context_counts[ctx_name][name] += int(policy_matches[name][m].sum())

        valid_eff = occupied & (canon_i > TOL) & (i_base > 1e-8) & (canon_i < 0.999999)
        if bool(valid_eff.any()):
            w_eff_full = np.full(canon_i.shape, np.nan, dtype=np.float32)
            w_eff_full[valid_eff] = ((canon_i[valid_eff] / i_base[valid_eff]) - 1.0) / DEFAULT_GAMMA
            w_eff = w_eff_full[valid_eff]
            finite = np.isfinite(w_eff)
            w_eff = w_eff[finite]
            effective_values.extend(w_eff[: max(0, 200000 - len(effective_values))].astype(float).tolist())
            coords = np.argwhere(valid_eff)
            for coord_i, (b, t, k) in enumerate(coords[:2000]):
                if len(effective_examples) >= 40:
                    break
                wv = float(((canon_i[b, t, k] / i_base[b, t, k]) - 1.0) / DEFAULT_GAMMA)
                candidates = {
                    "zero": 0.0,
                    "global": float(GLOBAL_SLOT_WEIGHTS[k]),
                    "conditional": float(cond_w[b, t, k]),
                }
                nearest = min(candidates, key=lambda name: abs(wv - candidates[name]))
                if abs(wv - candidates[nearest]) > 1e-3:
                    effective_examples.append(
                        {
                            "source_index": int(lo + b),
                            "recordingId": int(meta_rec[lo + b]),
                            "slot": int(k),
                            "history_timestep": int(t),
                            "canonical_I": float(canon_i[b, t, k]),
                            "I_base": float(i_base[b, t, k]),
                            "w_effective": wv,
                            "nearest": nearest,
                            "nearest_value": candidates[nearest],
                            "context": CONTEXT_NAMES.get(int(codes[lo + b, t]), str(int(codes[lo + b, t]))),
                        }
                    )
            zero_dist = np.abs(w_eff_full)
            global_dist = np.abs(w_eff_full - GLOBAL_SLOT_WEIGHTS.reshape(1, 1, 8))
            cond_dist = np.abs(w_eff_full - cond_w)
            dist_stack = np.stack([zero_dist, global_dist, cond_dist], axis=-1)
            nearest_idx = np.argmin(dist_stack, axis=-1)
            nearest_dist = np.min(dist_stack, axis=-1)
            for ni, name in enumerate(("zero", "global", "conditional")):
                nm = valid_eff & (nearest_idx == ni)
                effective_nearest[name] += int(np.count_nonzero(nm))
                effective_nearest[f"{name}_within_1e-3"] += int(np.count_nonzero(nm & (nearest_dist <= 1e-3)))

        print(f"[audit] {dataset} chunk {lo:,}-{hi:,}", flush=True)

    labels_arr = np.asarray(label_by_sample, dtype=object)
    label_counts = Counter(label_by_sample)
    recording_summary = {}
    for rec, ctr in sorted(recording_counts.items()):
        total = sum(ctr.values())
        dominant, dom_count = ctr.most_common(1)[0]
        recording_summary[str(rec)] = {
            "num_samples": int(total),
            "dominant_policy": dominant,
            "dominant_fraction": float(dom_count / total) if total else 0.0,
            "unexplained_fraction": float(ctr.get("UNEXPLAINED", 0) / total) if total else 0.0,
            "counts": dict(ctr),
        }

    split_summary = {}
    for split, ctr in sorted(split_counts.items()):
        total = sum(ctr.values())
        slot_ctr = split_slot_counts.get(split, Counter())
        split_slots = int(slot_ctr.get("slots", 0))
        split_summary[split] = {
            "num_samples": int(total),
            "sample_class_counts": dict(ctr),
            "unexplained_fraction": float(ctr.get("UNEXPLAINED", 0) / total) if total else 0.0,
            "slot_policy_match_rates": {
                name: float(slot_ctr.get(name, 0) / split_slots) if split_slots else 0.0 for name in POLICIES
            },
        }

    context_summary = {}
    for ctx, ctr in sorted(context_counts.items()):
        slots = int(ctr.get("slots", 0))
        context_summary[ctx] = {
            "slots": slots,
            "policy_slot_match_rates": {
                name: float(ctr.get(name, 0) / slots) if slots else 0.0 for name in POLICIES
            },
        }

    result = {
        "dataset": dataset,
        "num_samples": int(n),
        "num_slots": int(slot_total),
        "runtime_seconds": float(time.perf_counter() - start),
        "tolerance": TOL,
        "ibase_reproduction": {
            "finite_I_base_slots": int(ibase_reconstructed),
            "occupied_slots": int(ibase_reconstruct_total),
            "finite_fraction": float(ibase_reconstructed / max(1, ibase_reconstruct_total)),
        },
        "candidate_policies": list(POLICIES),
        "slot_level": {
            "policy_match_rates": {name: float(policy_slot_match[name] / max(1, slot_total)) for name in POLICIES},
            "policy_mae": {name: float(policy_abs_sum[name] / max(1, slot_total)) for name in POLICIES},
            "best_policy_counts": dict(slot_best),
            "best_policy_rates": {name: float(slot_best[name] / max(1, slot_total)) for name in POLICIES},
        },
        "sample_level": {
            "num_samples": int(sample_total),
            "exact_policy_counts": dict(policy_sample_exact),
            "exact_policy_rates": {name: float(policy_sample_exact[name] / max(1, sample_total)) for name in POLICIES},
            "classification_counts": dict(label_counts),
            "classification_rates": {k: float(v / max(1, sample_total)) for k, v in label_counts.items()},
            "examples": sample_details,
        },
        "recording_level": recording_summary,
        "source_index_ranges": _range_summary(label_by_sample, indices),
        "split_level": split_summary,
        "context_level": context_summary,
        "effective_weight": {
            "nearest_counts": dict(effective_nearest),
            "sampled_value_summary": _summ(np.asarray(effective_values, dtype=np.float64)),
            "unmatched_examples": effective_examples,
        },
        "boost_and_top3": {
            "nonzero_boost_value_match_rates": {
                name: float(boost_nonzero_match[name] / max(1, int(np.count_nonzero(np.asarray(x_nb[:n, ..., 9]) > TOL))))
                for name in ("NO_BOOST", "GLOBAL", "CONDITIONAL")
            },
            "zero_mask_match_rates": {name: float(policy_zero_match[name] / max(1, slot_total)) for name in POLICIES},
            "top3_scope_zero_mask_rates": {name: float(top3_zero_match[name] / max(1, slot_total)) for name in top3_scopes},
        },
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=EXPERIMENT_ROOT.parent / "neighformer" / "data")
    parser.add_argument("--dataset", choices=["highD", "exiD", "both"], default="both")
    parser.add_argument("--target-hz", type=float, default=3.0)
    parser.add_argument("--chunk-size", type=int, default=25000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", type=Path, default=EXPERIMENT_ROOT / "data" / "multiagent" / "final_i_provenance_audit.json")
    args = parser.parse_args(argv)

    datasets = ["highD", "exiD"] if args.dataset == "both" else [args.dataset]
    out = {}
    for dataset in datasets:
        out[dataset] = audit_dataset(
            dataset,
            args.source_root.expanduser().resolve(),
            args.target_hz,
            args.chunk_size,
            args.max_samples,
        )
        print(json.dumps({dataset: {
            "slot_level": out[dataset]["slot_level"]["policy_match_rates"],
            "sample_level": out[dataset]["sample_level"]["classification_rates"],
        }}, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
