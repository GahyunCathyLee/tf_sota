# Common Codex Prompt: SOTA Adapter Implementation

You are working with two sibling directories:

- Experiment root: `/home/gahyun/sota_experiments`
- NeighFormer data/code root: `/home/gahyun/neighformer`

The research goal is to compare `baseline` vs `dimI` input features for
multi-agent vehicle trajectory prediction on highD and exiD. The NeighFormer
repository contains mmap/npy data:

```text
/home/gahyun/neighformer/data/highD/dimI/x_ego.npy
/home/gahyun/neighformer/data/highD/dimI/x_nb.npy
/home/gahyun/neighformer/data/exiD/dimI/x_ego.npy
/home/gahyun/neighformer/data/exiD/dimI/x_nb.npy
```

The shared schema is documented in:

```text
adapters/common_schema.md
```

Important shape contract:

```text
x_ego   : [N, T_h, 6]
x_nb    : [N, T_h, K, 10]
nb_mask : [N, T_h, K]
y       : [N, T_f, 2]
T_h = 6
T_f = 15
K = 8
baseline neighbor indices = [0, 1, 2, 3, 4, 5]
dimI neighbor indices     = [0, 1, 2, 3, 4, 5, 8, 9]
```

Your task for each model:

1. Inspect the upstream repository under `external/<model>/`
   or the sibling checkout mentioned in `model_registry.yaml`.
2. Identify the original dataset class, training entrypoint, config format,
   model input shape, target shape, loss, and evaluation code.
3. Implement `adapters/<model>/train.py`.
4. Keep model-specific glue in `adapters/<model>/`.
5. Do not rewrite the whole upstream model unless the model has no public code.
6. Preserve the experiment CLI:

```bash
cd /home/gahyun/sota_experiments
python adapters/<model>/train.py \
  --config configs/models/<model>.yaml \
  --dataset highD \
  --feature-mode baseline \
  --data-root /home/gahyun/neighformer/data \
  --output-dir runs/<model>/highD/baseline
```

The adapter must support:

- `--dataset highD|exiD`
- `--feature-mode baseline|dimI`
- `--data-root <path>`
- `--output-dir <path>`
- a smoke mode or config path that can run one epoch or a few batches locally

Minimum output artifacts:

- checkpoint or saved model state
- `metrics.json`
- `run_config.yaml` or equivalent copied config
- a short log under the output directory

Validation checklist:

1. `python -m py_compile` passes for new adapter files.
2. `python scripts/inspect_data.py --dataset highD --feature-mode baseline` works in the selected env.
3. highD baseline smoke run starts, computes a loss, and writes artifacts.
4. highD dimI smoke run starts with the correct input channel count.
5. exiD baseline smoke run loads arrays.
6. exiD dimI smoke run loads arrays.
7. If exiD split files are missing, document that and do not invent split indices silently.

Do not run full training until smoke tests pass.
