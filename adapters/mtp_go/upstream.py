"""Locate and import the upstream MTP-GO codebase.

The adapter never vendors upstream model code. It puts the upstream checkout on
``sys.path`` and imports the original encoder/decoder/motion models from it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]

# Search order for the upstream checkout.
CANDIDATE_DIRS = (
    EXPERIMENT_ROOT / "external" / "mtp_go",
    Path.home() / "mtp-go",
)

REQUIRED_FILES = ("base_mdn.py", "losses.py", "models/gru_gnn.py", "models/motion_models.py")


def resolve_upstream_dir(explicit: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        p = Path(explicit)
        candidates.append(p if p.is_absolute() else (EXPERIMENT_ROOT / p))
    candidates.extend(CANDIDATE_DIRS)

    for cand in candidates:
        cand = cand.expanduser()
        if all((cand / rel).exists() for rel in REQUIRED_FILES):
            return cand.resolve()

    raise FileNotFoundError(
        "Could not locate an MTP-GO checkout. Tried:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nClone https://github.com/westny/mtp-go.git into external/mtp_go "
        "(or symlink an existing checkout there)."
    )


def add_upstream_to_path(upstream_dir: Path) -> None:
    path = str(upstream_dir)
    if path not in sys.path:
        sys.path.insert(0, path)


def upstream_commit(upstream_dir: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(upstream_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        commit = out.stdout.strip()
    except Exception:
        return "unknown"
    try:
        dirty = subprocess.run(
            ["git", "-C", str(upstream_dir), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout.strip()
    except Exception:
        dirty = ""
    return f"{commit}-dirty" if dirty else commit


def build_motion_model(cfg: SimpleNamespace, dt: float, static_f_dim: int) -> Any:
    """Mirror the motion-model selection in upstream ``train.py``."""
    from models.motion_models import (  # noqa: PLC0415  (upstream import needs sys.path)
        Curvature,
        CurviLinear,
        DoubleIntegrator,
        FirstOrderNeuralODE,
        KinematicSingleTrack,
        SecondOrderNeuralODE,
        SingleIntegrator,
        TripleIntegrator,
        Unicycle,
    )

    name = cfg.motion_model
    if name == "1Xint":
        return SingleIntegrator(solver=cfg.ode_solver, dt=dt, mixtures=cfg.n_mixtures)
    if name == "2Xint":
        return DoubleIntegrator(solver=cfg.ode_solver, dt=dt, mixtures=cfg.n_mixtures)
    if name == "3Xint":
        return TripleIntegrator(solver=cfg.ode_solver, dt=dt, mixtures=cfg.n_mixtures)
    if name == "singletrack":
        return KinematicSingleTrack(solver=cfg.ode_solver, dt=dt, mixtures=cfg.n_mixtures)
    if name == "unicycle":
        return Unicycle(solver=cfg.ode_solver, dt=dt, mixtures=cfg.n_mixtures)
    if name == "curvature":
        return Curvature(solver=cfg.ode_solver, dt=dt, mixtures=cfg.n_mixtures)
    if name == "curvilinear":
        return CurviLinear(solver=cfg.ode_solver, dt=dt, mixtures=cfg.n_mixtures, u1_lim=cfg.u1_lim)
    if name == "neuralode":
        return FirstOrderNeuralODE(
            solver=cfg.ode_solver,
            dt=dt,
            mixtures=cfg.n_mixtures,
            static_f_dim=static_f_dim,
            n_hidden=cfg.n_ode_hidden,
            n_layers=cfg.n_ode_layers,
        )
    if name == "2Xnode":
        return SecondOrderNeuralODE(
            solver=cfg.ode_solver,
            dt=dt,
            mixtures=cfg.n_mixtures,
            static_f_dim=static_f_dim,
            n_hidden=cfg.n_ode_hidden,
            n_layers=cfg.n_ode_layers,
        )
    raise ValueError(f"Unsupported motion_model: {name}")


ROTATIONAL_MOTION_MODELS = ("singletrack", "unicycle", "curvature", "curvilinear")
