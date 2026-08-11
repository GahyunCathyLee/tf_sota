#!/usr/bin/env python3
"""Generate and optionally execute the unified SOTA experiment matrix."""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = Path(os.environ.get("SOTA_WORK_ROOT", EXPERIMENT_ROOT)).resolve()
MATRIX_PATH = EXPERIMENT_ROOT / "configs" / "matrix.yaml"
REGISTRY_PATH = EXPERIMENT_ROOT / "model_registry.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def command_for(job: dict[str, str], matrix: dict[str, Any]) -> list[str]:
    model = job["model"]
    dataset = job["dataset"]
    feature_mode = job["feature_mode"]
    adapter = EXPERIMENT_ROOT / matrix["paths"]["adapter_root"] / model / "train.py"
    model_cfg = EXPERIMENT_ROOT / "configs" / "models" / f"{model}.yaml"
    output_dir = (
        EXPERIMENT_ROOT
        / matrix["paths"]["output_root"]
        / model
        / dataset
        / feature_mode
    )

    return [
        sys.executable,
        str(adapter),
        "--config",
        str(model_cfg),
        "--dataset",
        dataset,
        "--feature-mode",
        feature_mode,
        "--data-root",
        str(Path(matrix["paths"]["canonical_data_root"]).resolve()),
        "--output-dir",
        str(output_dir),
    ]


def adapter_state(model: str, matrix: dict[str, Any]) -> str:
    adapter = EXPERIMENT_ROOT / matrix["paths"]["adapter_root"] / model / "train.py"
    if adapter.exists():
        return "ready"
    return "missing"


def expand_jobs(args: argparse.Namespace, matrix: dict[str, Any]) -> list[dict[str, str]]:
    models = [args.model] if args.model else matrix["models"]
    datasets = [args.dataset] if args.dataset else matrix["datasets"]
    feature_modes = [args.feature_mode] if args.feature_mode else matrix["feature_modes"]

    return [
        {"model": model, "dataset": dataset, "feature_mode": feature_mode}
        for model, dataset, feature_mode in itertools.product(models, datasets, feature_modes)
    ]


def write_job_files(jobs: list[dict[str, str]], matrix: dict[str, Any]) -> None:
    run_root = EXPERIMENT_ROOT / matrix["paths"]["output_root"]
    run_root.mkdir(parents=True, exist_ok=True)

    commands_path = run_root / "commands.txt"
    csv_path = run_root / "jobs.csv"

    with commands_path.open("w", encoding="utf-8") as f:
        for job in jobs:
            f.write(" ".join(shlex.quote(x) for x in command_for(job, matrix)) + "\n")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "dataset", "feature_mode", "adapter_state"],
        )
        writer.writeheader()
        for job in jobs:
            row = dict(job)
            row["adapter_state"] = adapter_state(job["model"], matrix)
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=None, default=None)
    parser.add_argument("--dataset", choices=["highD", "exiD"], default=None)
    parser.add_argument("--feature-mode", choices=["baseline", "dimI"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing-adapter", action="store_true")
    args = parser.parse_args()

    matrix = load_yaml(MATRIX_PATH)
    registry = load_yaml(REGISTRY_PATH)["models"]
    jobs = expand_jobs(args, matrix)
    write_job_files(jobs, matrix)

    for job in jobs:
        model = job["model"]
        if model not in registry:
            raise SystemExit(f"Unknown model in matrix: {model}")
        state = adapter_state(model, matrix)
        cmd = command_for(job, matrix)
        print(f"[job] {model} {job['dataset']} {job['feature_mode']} adapter={state}")
        print("      " + " ".join(shlex.quote(x) for x in cmd))

        if args.dry_run:
            continue
        if state == "missing" and not args.allow_missing_adapter:
            raise SystemExit(
                f"Adapter is missing for {model}. Create adapters/{model}/train.py first."
            )
        subprocess.run(cmd, cwd=WORK_ROOT, check=True)


if __name__ == "__main__":
    main()
