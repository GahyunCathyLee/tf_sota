# Trajectron++ Adapter

This adapter keeps the local Trajectron++ model/training code intact and changes
only the input path: NeighFormer highD/exiD npy windows are converted into
Trajectron++ `Environment` pickles.

The selected upstream checkout is `../trajectronPP`, which already exposes
configurable neighbor relative input features. The adapter uses:

- `baseline`: `dx, dy, dvx, dvy, dax, day`
- `dimI`: `dx, dy, dvx, dvy, dax, day, dim, I`

Smoke:

```bash
conda run -n trajectron++ python train.py \
  --config configs/models/trajectronpp.yaml \
  --dataset highD \
  --feature-mode baseline \
  --data-root /home/gahyun/neighformer/data \
  --device cpu
```

Evaluate:

```bash
conda run -n trajectron++ python evaluate.py \
  --model trajectronpp \
  --ckpt runs/trajectronpp/highD/baseline/checkpoints/best.pt \
  --split test \
  --data-root /home/gahyun/neighformer/data
```

Full training should be run on Colab/GPU with `--mode full` and an
environment-compatible Trajectron++ checkout at `../trajectronPP` or via
`--upstream-dir`.
