# MTP-GO Adapter

Runs the upstream MTP-GO model (GRU-GNN encoder/decoder + neural-ODE motion
model, MDN output) on the canonical NeighFormer highD/exiD npy data, for the
shared `baseline` vs `dimI` comparison.

Status: **smoke-verified** on highD and exiD, both feature modes, with canonical
splits. Full training has not been run.

## Files

| File | Role |
| --- | --- |
| `train.py` | Training entry point (equivalent of `neighformer/train.py`) |
| `evaluate.py` | Evaluation entry point (equivalent of `neighformer/evaluate.py`) |
| `dataset.py` | NeighFormer npy → MTP-GO PyG scene graphs |
| `metrics.py` | Metric definitions and report tables, identical to `neighformer/src/metrics.py` |
| `lit_module.py` | Ego-only validation/test metrics on top of upstream `LitEncoderDecoder` |
| `upstream.py` | Locates the upstream checkout, imports its modules, builds the motion model |

No upstream model code is copied. `external/mtp_go` is a symlink to a
`westny/mtp-go` checkout (currently `/home/gahyun/mtp-go`); the adapter puts it
on `sys.path` and imports `models.gru_gnn`, `models.motion_models`, `base_mdn`
and `losses` from there. The resolved directory and its commit hash are written
into `run_config.yaml` / `environment.json` on every run.

## train.py

Environment: `conda activate tf` (torch 2.9.1+cu128, PyG 2.7.0, lightning 2.6.1,
torchdiffeq 0.2.5).

```bash
# smoke (default mode): 4 epochs over 2048 train / 1024 eval samples
python adapters/mtp_go/train.py --config configs/models/mtp_go.yaml \
  --dataset highD --feature-mode baseline

# full training (opt-in)
python adapters/mtp_go/train.py --config configs/models/mtp_go.yaml \
  --dataset highD --feature-mode dimI --mode full

# data conversion report only, no training
python adapters/mtp_go/train.py ... --check-data
```

### Seed sweep

`configs/mtp_go/` holds one config per run, named exactly like
`neighformer/configs/`: `<dataset><feature><seed_index>.yaml`, where feature
`0` = baseline and `1` = dimI, and the seed index is
`1`→42, `2`→1234, `3`→3407, `4`→0, `5`→777. 2 datasets × 2 feature modes ×
5 seeds = 20 configs.

```bash
python adapters/mtp_go/train.py --config configs/mtp_go/highD1-1.yaml --mode full
```

`dataset` and `feature_mode` come from the config, so `--dataset` /
`--feature-mode` become optional (CLI still wins if given).

Configs support a `base:` key, resolved relative to the config's own directory
and then the repo root, and merged section by section. The chain is:

```text
configs/models/mtp_go.yaml   model, optimiser, dt, data root   <- change hyperparameters HERE
  configs/mtp_go/_base.yaml  output/ckpt/tensorboard layout keyed by {exp_tag}
    configs/mtp_go/highD1-1.yaml   exp_tag, dataset, feature_mode, seed  (4 keys)
```

Each per-run file restates only what changes, so a hyperparameter edit is a
one-line change in `configs/models/mtp_go.yaml` and cannot silently diverge
between seeds. Each run gets its own directory:

```text
runs/mtp_go/<dataset>/<feature_mode>/<exp_tag>/   metrics.json, run_config.yaml, logs
ckpts/mtp_go/<exp_tag>/                           best.ckpt, last.ckpt
tensorboard/mtp_go/<exp_tag>/
```

Run the whole sweep:

```bash
for c in configs/mtp_go/highD*.yaml configs/mtp_go/exiD*.yaml; do
  [ "$(basename "$c")" = "_base.yaml" ] && continue
  python adapters/mtp_go/train.py --config "$c" --mode full
done
```

### Paths

Every path is `CLI > config > default`, and relative paths resolve against the
`sota_experiments` root. Path strings may contain `{dataset}`, `{feature_mode}`
and `{exp_tag}` placeholders, so one config can serve the whole matrix without
runs overwriting each other.

| CLI | Config key | Default |
| --- | --- | --- |
| `--data-root` | `data.root` | `data` (expects `<root>/<dataset>/dimI` and `<root>/<dataset>/splits`) |
| `--output-dir` | `training.output_dir` | `runs/mtp_go/<dataset>/<feature-mode>` |
| `--ckpt-dir` | `training.ckpt_dir` | `<output-dir>/checkpoints`; when set, `<ckpt-dir>/<exp-tag>/` |
| `--tensorboard-dir` | `training.tensorboard_dir` | `<output-dir>/tensorboard`; always `<dir>/<exp-tag>/` |
| `--exp-tag` | `exp_tag` | `mtp_go_<dataset>_<feature-mode>` |
| `--dataset` | `dataset` | required from one of the two |
| `--feature-mode` | `feature_mode` | required from one of the two |
| `--config` | — | required |

Other overrides: `--epochs`, `--batch-size`, `--lr`, `--seed`, `--num-workers`,
`--accelerator {auto,gpu,cpu}`, `--max-train-samples`, `--max-eval-samples`,
`--upstream-dir`, `--resume`, `--no-tensorboard`, `--split-fallback`.

Like `neighformer/train.py`, it logs one line per epoch
(`loss / val_ade / val_fde / val_nll`), writes TensorBoard scalars, keeps
`best.ckpt` (monitored on `val_ade`) plus `last.ckpt`, and evaluates the **best**
checkpoint at the end — not the last epoch.

## evaluate.py

Rebuilds the model from the config stored inside the checkpoint, so only
`--ckpt` is required.

```bash
python adapters/mtp_go/evaluate.py --ckpt runs/mtp_go/highD/dimI/checkpoints/best.ckpt
python adapters/mtp_go/evaluate.py --ckpt .../best.ckpt --split val
python adapters/mtp_go/evaluate.py --ckpt .../best.ckpt --scenario
python adapters/mtp_go/evaluate.py --ckpt .../best.ckpt --measure-time --iters 200
python adapters/mtp_go/evaluate.py --ckpt .../best.ckpt --data-root ./data   # moved data
```

| Flag | Meaning |
| --- | --- |
| `--split {train,val,test}` | default `test` |
| `--data-root` | override the root stored in the checkpoint (different machine / Colab) |
| `--scenario` | per event / state breakdown from `scenario_labels.csv` |
| `--batch-size`, `--num-workers`, `--device` | override the checkpoint's values |
| `--max-samples` | evaluate only the first N samples |
| `--measure-time`, `--warmup`, `--iters` | single-sample latency, batch size 1 |
| `--output-json` | write the metrics to JSON |

Output tables have the same layout as `neighformer/evaluate.py`. Note that
MTP-GO's single-sample latency is ~89 ms on an A6000 — the 15-step rollout
propagates an EKF covariance through `vmap(jacrev)` at every step — so
NeighFormer's default `10,000` iterations takes ~15 minutes. Use `--iters`.

## Metric compatibility with NeighFormer

`metrics.py` reproduces `neighformer/src/metrics.py` exactly so the two models
can be compared directly:

```text
ade     = mean_samples( mean_t ||pred - y|| )
fde     = mean_samples( ||pred_T - y_T|| )
rmse    = mean_samples( sqrt( mean_t ||pred - y||^2 ) )     # per-sample sqrt
rmse@Ns = sqrt( sum_samples ||pred_i - y_i||^2 / n ),  i = int(N * hz) - 1
```

`hz` (`data.hz`, default 3.0) is NeighFormer's reporting convention and is not
exactly `1/dt` (dt = 0.32 s → 3.125 Hz). The index formula is kept identical on
purpose; the true time of each reported second is stored next to it as
`rmse_Ns_actual_seconds` (1s→0.96s, 2s→1.92s, 3s→2.88s, 4s→3.84s, 5s→4.8s).
MTP-GO is a mixture-density model, so the reported trajectory is the
most-likely mixture component; `min_ade` / `min_fde` over the 8 components are
reported as extras.

## Colab

Data at `./data/highD` and `./data/exiD` with the same layout as
`neighformer/data`. `configs/models/mtp_go_colab.yaml` is preconfigured for it.

```bash
!git clone https://github.com/GahyunCathyLee/sota_experiments.git
%cd sota_experiments
# upstream model code (the fork matches the commit recorded in run_config.yaml)
!git clone https://github.com/GahyunCathyLee/mtp-go.git external/mtp_go

!pip -q install torch_geometric lightning torchdiffeq
from google.colab import drive; drive.mount('/content/drive')

# data -> ./data/highD/{dimI,splits}, ./data/exiD/{dimI,splits}
!python adapters/mtp_go/train.py --config configs/models/mtp_go_colab.yaml \
    --dataset highD --feature-mode dimI --mode full
```

The Colab config keeps checkpoints, TensorBoard and metrics on Google Drive, so
a runtime reset does not lose them; rerun the same command with `--resume` to
continue from `last.ckpt`. Only `models/`, `base_mdn.py` and `losses.py` are
imported from the upstream checkout, and those files are unmodified upstream
code, so a fresh clone works.

## Feature mapping

The NeighFormer schema and upstream's own `highD-imp` preprocessing already use
the same node convention, so the mapping is direct:

| MTP-GO scene graph | NeighFormer source |
| --- | --- |
| node 0 (ego) | `x_ego[i]` → `[x, y, vx, vy, ax, ay]`, positions relative to the ego position at the last history step |
| node j+1 (neighbour) | `x_nb[i, :, k, idx]`, ego-relative `[dx, dy, dvx, dvy, dax, day]` (+ `dim`, `I` in dimI mode) |
| node presence | `nb_mask`; only slots present at some history step become nodes, and rows are zeroed at steps where the slot is absent |
| `edge_index` / `edge_features` | rebuilt per history step: fully connected + self-loops over the nodes present at that step, edge feature = Euclidean distance in the ego-relative frame (same as upstream `_build_edges`) |
| `y` | node 0 only: `y.npy` (+ `y_vel.npy`, `y_acc.npy`) → `[x, y, vx, vy, ax, ay]`; the 2Xnode motion model consumes the first 4 |
| `tar_real_mask` | True for node 0 only |

Feature modes select neighbour channels from `x_nb`:

- `baseline` → indices `[0,1,2,3,4,5]` → **6** node channels → `encoder.input_size = 6`
- `dimI` → indices `[0,1,2,3,4,5,8,9]` → **8** node channels → `encoder.input_size = 8`

In `dimI` mode the ego row has no `dim`/`I` of its own, so both extra channels
are filled with `-1.0`, mirroring upstream's `inp[0, :, 6] = -1.0` dummy
importance. Every run writes `data_report.json` with per-channel min/max/mean
and non-zero fraction over the neighbour rows, which shows that `dim` (0..4) and
`I` (0..1) actually reach the model in `dimI` mode and are absent in `baseline`.

## Deliberate deviations from upstream

1. **Future graph topology.** Upstream builds future edges from ground-truth
   future neighbour positions. The NeighFormer schema stores no neighbour
   futures, so the last observed history graph is reused for all `T_f` decoder
   steps.
2. **Ego-only supervision and metrics.** Only node 0 has a target, so
   `tar_real_mask` is False for every neighbour node and all reported metrics are
   ego-only (`data.ptr[:-1]`). Upstream's `test_step` is not used: it computes a
   miss rate with a hard-coded `i < 20` horizon check that divides by zero at
   `T_f = 15`. `lit_module.evaluate` replaces it with exact, sample-weighted
   aggregation.
3. **Variable node count.** Neighbour slots absent at every history step are
   dropped instead of being kept as zero-filled isolated nodes. Ego stays at
   index 0 so `data.ptr` still selects it.
4. **Static features are placeholders.** Vehicle type / length / width are not in
   the NeighFormer schema. `v_type` is a constant one-hot and `dim` is zeros;
   they only reach the model when `init_static` or `n_ode_static` is enabled
   (both default to False). Motion models needing heading or wheelbase
   (`singletrack`, `unicycle`, `curvature`, `curvilinear`) are rejected up front.
5. **Loss schedule horizon.** Upstream derives its EWTA → EWTA+NLL → NLL and
   teacher-forcing schedules from `epochs`. The adapter separates
   `schedule_epochs` (the schedule horizon) from `epochs` (how long the Trainer
   runs) so a smoke run executes the first epochs of a realistic schedule rather
   than collapsing it. `schedule_epochs` must be ≥ 16: at 8 or fewer,
   `epochs // 8 == 1` fails upstream's `wta_epochs > 1` guard, the pure-EWTA
   phase is skipped and the warm-up weight starts at 2.0, which gives the NLL
   term a **negative** coefficient (observed loss ≈ -3e9).

## dt

NeighFormer preprocesses at `target_hz = 3.0` over 25 Hz raw data, i.e. a stride
of 8 frames → **dt = 0.32 s** (T_h = 6 → 1.92 s history, T_f = 15 → 4.8 s
horizon). The motion model integrates with the configured `runtime.dt`; the
adapter re-estimates dt from `x_ego` displacement / velocity at startup and warns
on a >15% mismatch. Because the horizon is 4.8 s, `metrics.json` reports
`RMSE@1s..5s` snapped to the nearest available step and records the true time in
`RMSE@Ns_actual_seconds` (1s→0.96s, 2s→1.92s, 3s→2.88s, 4s→3.84s, 5s→4.8s), plus
the full `step_RMSE` / `step_ADE` arrays.

## Splits

Splits come from `<data-root>/<dataset>/splits/{train,val,test}_indices.npy`.

- highD: 682,333 / 96,885 / 194,618 (70/10/20).
- exiD: 471,091 / 67,347 / 135,062 (70/10/20), generated with
  `neighformer/data/exiD/split.py`.

If the files are missing the adapter refuses to run and points at `split.py`.
`--split-fallback sequential` opts into a deterministic 70/15/15 sequential
split; it logs a warning and records the fallback in `run_config.yaml`. Numbers
produced that way are not comparable to canonical-split runs.

## Output artifacts

Written to `--output-dir` (checkpoints and TensorBoard go to their own
directories when `--ckpt-dir` / `--tensorboard-dir` are given):

- `<ckpt-dir>/best.ckpt` (monitored on `val_ade`), `last.ckpt`, and a copy of
  `run_config.yaml` so a checkpoint stays self-describing
- `tensorboard/<exp-tag>/` — train/val loss and val ADE/FDE/NLL scalars
- `metrics.json` — ade, fde, rmse, min_ade/min_fde over the 8 mixture components,
  nll, `step_rmse`, `step_ade`, `rmse_1..5s`, for both val and test
- `run_config.yaml` — exact command, effective config, feature mapping, split
  provenance, model summary, environment, upstream commit
- `environment.json` — Python / torch / CUDA / PyG / Lightning / torchdiffeq versions
- `data_report.json` — array shapes, example scene graph, per-channel stats
- `train.log`

## Verification performed

- `python -m py_compile` on all adapter files.
- `scripts/inspect_data.py` for highD baseline.
- highD and exiD, `baseline` and `dimI` smoke runs: loss decreases, artifacts
  written, 6 vs 8 input channels confirmed (146,843 vs 147,995 parameters).
- `evaluate.py` on highD and exiD checkpoints, including `--scenario`
  (event / state breakdown) and `--measure-time`.
- Path resolution: config-only, CLI-only, and `{dataset}`/`{feature_mode}`
  placeholders; Colab-style `./data/<dataset>` layout.
- Learning check on 512 highD baseline samples, 40 epochs:
  val ADE 16.97 → 1.03 m, FDE 43.62 → 2.64 m, NLL 4794 → 13.
