# SOTA Trajectory Prediction Experiments

This directory is the integration layer for running the same highD/exiD
experiments across multiple trajectory prediction models.

The current research question is:

- Compare `baseline` vs `dimI` input features.
- Use highD and exiD as the training/evaluation datasets.
- Keep model-specific code in one repository through `external/<model_name>/`
  plus thin adapters under `adapters/<model_name>/`.

## Target Models

Primary candidates:

- HiVT
- QCNet
- MTR++
- PAR
- SIMPL
- HPTR
- GameFormer
- QCNeXt
- MotionLM
- MTP-GO
- Trajectron++

See `model_registry.yaml` for upstream URLs, code availability, and the
recommended implementation strategy for each model.

## Feature Modes

The initial experiment matrix is intentionally narrow:

- `baseline`: neighbor channels dx, dy, dvx, dvy, dax, day
- `dimI`: neighbor channels dx, dy, dvx, dvy, dax, day, dim, I

The canonical data source is `/home/gahyun/neighformer/data/<dataset>/dimI`.
Feature modes are selected from `x_nb.npy`; they are not separate data folders.

## Repository Layout

```text
sota_experiments/
  README.md
  model_registry.yaml
  configs/
    matrix.yaml
    models/<model>.yaml
  adapters/
    common_schema.md
    <model>/              # model-specific dataset/model glue goes here
  external/
    <model>/              # cloned upstream repositories, ignored by git
  scripts/
    clone_upstreams.sh
    run_matrix.py
  runs/
    commands.txt
    jobs.csv
```

## Workflow

1. Inspect the canonical NeighFormer data:

```bash
python scripts/inspect_data.py --dataset highD --feature-mode baseline
python scripts/inspect_data.py --dataset highD --feature-mode dimI
python scripts/inspect_data.py --dataset exiD --feature-mode dimI
```

The highD split files are currently present. If exiD reports missing split files,
generate or restore `/home/gahyun/neighformer/data/exiD/splits/*.npy` before
split-based training.

2. Clone the open-source upstream repositories:

```bash
bash scripts/clone_upstreams.sh
```

3. Generate the baseline/dimI experiment matrix:

```bash
python scripts/run_matrix.py --dry-run
```

4. Implement one adapter at a time under `adapters/<model>/`.

Use the handoff prompts in `prompts/<model>.md` when starting a new Codex
session for a model. The common prompt and done definition are:

- `prompts/common_adapter_prompt.md`
- `prompts/adapter_done_definition.md`

5. Run smoke training first:

```bash
python scripts/run_matrix.py --model hivt --dataset highD --feature-mode baseline --dry-run
```

## Notes

This layer does not vendor upstream code directly. It keeps each official codebase
under `external/` and records the local changes needed to read the unified
highD/exiD feature format.

Models with no clear official training code, such as MotionLM, are tracked as
`reimplement` targets rather than clone-first targets.
