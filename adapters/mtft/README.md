# MTFT Adapter

This adapter implements MTFT (Multi-scale Temporal Fusion Transformer for Incomplete Vehicle Trajectory Prediction) as a single-target predictor on the existing NeighFormer highD/exiD `dimI` arrays.

## Architecture

The model follows the MTFT paper and official `FMSTF` code:

- MLP input projection from historical coordinates to hidden dimension.
- Sinusoidal positional encoding.
- Multi-scale Attention Head (MAH) with five scale-specific temporal masks. Scale `i` permits attention between timesteps whose index difference is divisible by `i`.
- Four temporal encoder layers by default. Intermediate layers reduce the five scale streams back to hidden size; the final layer keeps multi-scale streams for CRMF.
- Continuity Representation-guided Multi-scale Fusion (CRMF), using the observation matrix from sequence validity and scale masks to build information-increment weights across time.
- VectorNet-style global interaction over the target and valid surrounding vehicles.
- LSTMCell future decoder producing one deterministic target trajectory.

## Dataset Mapping

The source arrays are:

- `x_ego`: `(N, 6, 6)` with `[x, y, vx, vy, ax, ay]`
- `x_nb`: `(N, 6, 8, 10)` with `[dx, dy, dvx, dvy, dax, day, s_x, s_y, dim, I]`
- `nb_mask`: `(N, 6, 8)` where `True` means the neighbor slot exists
- `y`: `(N, 15, 2)` future target positions

The processed coordinates are ego-centered in the last observed target frame and are measured in meters. MTFT inputs use:

- target: `x_ego[..., 0:2]`
- neighbor: `x_ego[..., 0:2] + x_nb[..., 0:2]`

Predictions and metrics stay in the same ego-centered meter frame.

## Baseline vs +I

Baseline input shape is `(B, 6, 9, 2)`: target plus 8 neighbors, with only historical `[x, y]`.

`+I` input shape is `(B, 6, 9, 3)`: `[x, y, I]`. The scalar `I` enters only in `adapters/mtft/dataset.py` when constructing the MTFT input tensor. Neighbor `I` comes from `x_nb[..., 9]`; target `I` is set to `0`. No other dimI channels, labels, losses, attention masks, or decoder outputs use `I`.

## Masks

`nb_mask` controls neighbor validity. Invalid neighbor slots are masked in global interaction and do not contribute as real vehicles. Observation masks are separate from agent masks and support MTFT's incomplete-trajectory mechanism. By default:

```yaml
missing:
  enabled: false
```

so the main baseline and `+I` experiments use complete historical trajectories. When enabled, random missing observations are generated reproducibly from the configured seed and are applied only to historical inputs.

## Commands

Training:

```bash
python train.py --config configs/mtft/highD0-1.yaml
python train.py --config configs/mtft/highD2-1.yaml
python train.py --config configs/mtft/exiD0-1.yaml
python train.py --config configs/mtft/exiD2-1.yaml
```

Colab-style overrides:

```bash
python train.py \
  --config configs/mtft/exiD2-1.yaml \
  --epochs 100 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --num-workers 2 \
  --lr 1e-4 \
  --weight-decay 0.0 \
  --grad-clip 1.0 \
  --ckpt-dir /content/drive/MyDrive/TrajectoryPrediction/ckpts/mtft \
  --output-dir /content/drive/MyDrive/TrajectoryPrediction/runs/mtft/{dataset}/{feature_mode}/{exp_tag}
```

Other supported train overrides include `--data-root`, `--split-root`,
`--exp-tag`, `--seed`, `--device`, `--amp/--no-amp`, `--hidden-dim`,
`--num-layers`, `--num-heads`, and `--dropout`.

Evaluation:

```bash
python evaluate.py --ckpt ckpts/mtft/highD0-1/best.pt
python evaluate.py --checkpoint ckpts/mtft/highD2-1/best.pt --scenario
```

Useful validation commands:

```bash
python train.py --config configs/mtft/highD0-1.yaml --epochs 0 --check-data --forward-smoke --max-train-samples 128 --max-eval-samples 64
python train.py --config configs/mtft/highD2-1.yaml --epochs 0 --check-data --forward-smoke --tiny-overfit --max-train-samples 128 --max-eval-samples 64 --overfit-steps 80
```

Checkpoints are saved under `ckpts/mtft/{exp_tag}/`.

Scenario breakdown is available in evaluation with `--scenario`; it uses
`scenario_labels.csv` from the same dimI directory unless
`--scenario-labels <path>` is provided.

## Deviations

The original paper reports experiments with different sampling rates and horizons. This adapter uses the existing processed highD/exiD setup: 6 history steps and 15 future steps. The training loss is deterministic trajectory MSE, matching the official code's regression objective. Numerical paper reproduction is intentionally not attempted.
