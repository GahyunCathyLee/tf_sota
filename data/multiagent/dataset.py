#!/usr/bin/env python3
"""Lightweight memmap Dataset facade for production multi-agent arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ARRAY_NAMES = (
    "agent_ids",
    "x_agents",
    "obs_valid",
    "y_agents",
    "future_valid",
    "scored_agent_mask",
    "distances_t0",
    "agent_length",
    "agent_width",
    "agent_type",
    "heading",
    "lateral_velocity",
    "lane_id",
    "lane_level",
    "lane_offset",
    "lane_width",
    "semantic_role_t0",
    "candidate_count",
    "retained_count",
    "truncated_count",
    "recordingId",
    "ego_trackId",
    "t0_frame",
    "source_index",
    "ego_index",
)


class MultiAgentDataset:
    """Read samples from a directory created by ``data/multiagent/preprocess.py``."""

    def __init__(self, root: str | Path, indices: np.ndarray | None = None, mmap_mode: str = "r") -> None:
        self.root = Path(root)
        report_path = self.root / "preprocess_report.json"
        if not report_path.exists():
            raise FileNotFoundError(report_path)
        self.report = json.loads(report_path.read_text(encoding="utf-8"))
        self.arrays = {
            name: np.load(self.root / f"{name}.npy", mmap_mode=mmap_mode)
            for name in ARRAY_NAMES
            if (self.root / f"{name}.npy").exists()
        }
        n = int(self.arrays["agent_ids"].shape[0])
        self.indices = np.arange(n, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, Any]:
        idx = int(self.indices[item])
        return {name: arr[idx] for name, arr in self.arrays.items()}

    def take(self, count: int) -> "MultiAgentDataset":
        return MultiAgentDataset(self.root, self.indices[: int(count)])
