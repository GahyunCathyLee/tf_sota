# Adapter Done Definition

An adapter is considered ready for full experiments only when all items below are
true.

## Data

- Reads `highD` and `exiD`.
- Reads `baseline` and `dimI`.
- Confirms ego history has 6 channels.
- Confirms neighbor `baseline` selects 6 channels and neighbor `dimI` selects 8 channels.
- Handles missing or padded neighbors correctly.
- Preserves ego target as `[T_f, 2]`.

## Model

- Uses the upstream model architecture unless a reimplementation is explicitly
  required.
- Has a clearly documented mapping from common NeighFormer npy features to upstream model
  inputs.
- Does not silently drop `dim` or `I` in `dimI` mode.
- Can overfit or at least reduce loss on a tiny subset.

## Training

- Supports smoke training with a small number of batches.
- Supports checkpoint save and resume if upstream code provides it.
- Writes configs and metrics into `runs/<model>/<dataset>/<feature_mode>/`.

## Evaluation

- Reports ADE and FDE at minimum.
- Reports RMSE and step RMSE if practical.
- Uses the same target units and horizon as the NeighFormer-preprocessed experiments.

## Reproducibility

- Records upstream commit hash when an upstream repo is used.
- Records Python, PyTorch, CUDA, and key dependency versions.
- Includes the exact command used for the run.
