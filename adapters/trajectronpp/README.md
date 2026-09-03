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
  --config configs/trajectronpp/highD0-1.yaml \
  --mode smoke \
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

The full sweep configs live in `configs/trajectronpp/`. They default to
`mode: full`, so this starts a real run:

```bash
conda run -n trajectron++ python train.py \
  --config configs/trajectronpp/highD1-1.yaml \
  --data-root /home/gahyun/neighformer/data \
  --device cuda:0
```

To avoid slow Colab preprocessing, build the Trajectron++ pickle files once on a
faster/local machine:

```bash
conda run -n trajectron++ python train.py \
  --config configs/trajectronpp/highD1-1.yaml \
  --mode preprocess \
  --data-root /home/gahyun/neighformer/data
```

Then upload at least `processed/train_env.pkl` and `processed/val_env.pkl` for
that run, and train on Colab with:

```bash
python train.py \
  --config configs/trajectronpp/highD1-1.yaml \
  --processed-dir /content/drive/MyDrive/path/to/processed \
  --reuse-processed \
  --device cuda:0
```
