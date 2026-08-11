# MTP-GO Adapter

Runs the upstream MTP-GO model (GRU-GNN encoder/decoder + neural-ODE motion
model, MDN output) on the canonical NeighFormer highD/exiD npy data, for the
shared `baseline` vs `dimI` comparison.

Status: **smoke-verified** on highD (`baseline`, `dimI`) and exiD
(`baseline`, `dimI`, sequential-split fallback). Full training has not been run.

## Files

| File | Role |
| --- | --- |
| `train.py` | CLI entrypoint: data → model → training → metrics/artifacts |
| `dataset.py` | NeighFormer npy → MTP-GO PyG scene graphs |
| `lit_module.py` | Ego-only validation/test metrics on top of upstream `LitEncoderDecoder` |
| `upstream.py` | Locates the upstream checkout, imports its modules, builds the motion model |

No upstream model code is copied. `external/mtp_go` is a symlink to a
`westny/mtp-go` checkout (currently `/home/gahyun/mtp-go`); the adapter puts it
on `sys.path` and imports `models.gru_gnn`, `models.motion_models`, `base_mdn`
and `losses` from there. The resolved directory and its commit hash are written
into `run_config.yaml` / `environment.json` on every run.

## Commands

Environment: `conda activate tf` (torch 2.9.1+cu128, PyG 2.7.0, lightning 2.6.1,
torchdiffeq 0.2.5).

```bash
# smoke (default mode): 4 epochs over 2048 train / 1024 eval samples
python adapters/mtp_go/train.py --config configs/models/mtp_go.yaml \
  --dataset highD --feature-mode baseline \
  --data-root /home/gahyun/neighformer/data \
  --output-dir runs/mtp_go/highD/baseline

python adapters/mtp_go/train.py --config configs/models/mtp_go.yaml \
  --dataset highD --feature-mode dimI \
  --data-root /home/gahyun/neighformer/data \
  --output-dir runs/mtp_go/highD/dimI

# data conversion report only, no training
python adapters/mtp_go/train.py ... --check-data

# full training (opt-in)
python adapters/mtp_go/train.py ... --mode full
```

Useful overrides: `--epochs`, `--batch-size`, `--lr`, `--seed`, `--num-workers`,
`--accelerator {auto,gpu,cpu}`, `--max-train-samples`, `--max-eval-samples`,
`--upstream-dir`, `--resume`.

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

- highD: present.
- **exiD: missing.** The adapter refuses to run and points at
  `/home/gahyun/neighformer/data/exiD/split.py`. Passing
  `--split-fallback sequential` opts into a deterministic 70/15/15 sequential
  split; it logs a warning and records the fallback in `run_config.yaml`. exiD
  numbers produced this way are not comparable to canonical-split runs.

## Output artifacts

Written to `--output-dir`:

- `checkpoints/best.ckpt` (monitored on `val_ade`) and `checkpoints/last.ckpt`
- `metrics.json` — ADE, FDE, RMSE, minADE/minFDE over the 8 mixture components,
  NLL, `step_RMSE`, `step_ADE`, `RMSE@1..5s`, for both val and test
- `run_config.yaml` — exact command, effective config, feature mapping, split
  provenance, model summary, environment, upstream commit
- `environment.json` — Python / torch / CUDA / PyG / Lightning / torchdiffeq versions
- `data_report.json` — array shapes, example scene graph, per-channel stats
- `train.log`

## Verification performed

- `python -m py_compile` on all adapter files.
- `scripts/inspect_data.py` for highD baseline.
- highD `baseline` and `dimI` smoke runs: loss decreases, artifacts written,
  6 vs 8 input channels confirmed (146,843 vs 147,995 parameters).
- exiD `baseline` and `dimI`: arrays load and convert (`--check-data`); the
  missing-split error path was verified, and smoke training was verified with
  `--split-fallback sequential`.
- Learning check on 512 highD baseline samples, 40 epochs:
  val ADE 16.97 → 1.03 m, FDE 43.62 → 2.64 m, NLL 4794 → 13.
