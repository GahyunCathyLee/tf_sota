# PAR Adapter

Runs a PAR-style trajectory-token model on the canonical NeighFormer highD/exiD
arrays.

## Upstream

Official checkout:

```bash
git clone https://github.com/neerjathakkar/PAR.git external/par
```

The adapter records the upstream commit in `runtime.json` and checkpoints.

## Mapping

PAR's NuScenes car task is an autoregressive Llama model over discretized xy
trajectory tokens. highD/exiD use the same idea:

- positions are ego-centric NeighFormer coordinates
- acceleration tokens use PAR's first-order / second-order binning
- sequence order is neighbour slots first, ego last, repeated by token timestep
- `data/par/preprocess.py` fixes each neighbour slot to the vehicle ID present
  at `t0_frame` and tracks that ID through history/future
- when `y_nb.npy` exists, loss is computed on ego and neighbour future tokens
- ADE/FDE/RMSE are still reported on ego future for comparability

`dimI` uses continuous side-channel embeddings. For each fixed neighbour slot,
`[dim, I]` from the t0 slot is projected and added to that neighbour's token
embeddings. The ego token receives `[-1, -1]`; unavailable neighbour slots use
zeros. `baseline` has no side-channel projection.

## PAR Preprocessing

Do not modify `../neighformer`. Build a repo-local PAR data root instead:

```bash
python data/par/preprocess.py --dataset both \
  --source-data-root /home/gahyun/neighformer/data \
  --output-root data/par
```

This creates `data/par/<dataset>/dimI` with symlinks to the canonical arrays and
new PAR arrays:

```text
nb_ids.npy
nb_attr.npy
nb_attr_mask.npy
x_nb_abs.npy
x_nb_abs_mask.npy
y_nb.npy
y_nb_mask.npy
```

The seed-sweep configs default to `data.root: data/par`, so full PAR experiments
use these neighbour-future labels once preprocessing has been run. Passing
`--data-root /home/gahyun/neighformer/data` still works as an ego-only fallback.

## Training

Smoke run:

```bash
python adapters/par/train.py --config configs/par/highD0-1.yaml --mode smoke \
  --data-root data/par
```

Full run:

```bash
python adapters/par/train.py --config configs/par/highD0-1.yaml --mode full \
  --data-root data/par
```

Evaluation:

```bash
python adapters/par/evaluate.py --ckpt ckpts/par/highD0-1/best.pt --split test \
  --data-root data/par
```

The seed-sweep configs live in `configs/par/` and mirror
`configs/mtp_go/`: each per-run file changes only `dataset`, `feature_mode`,
`exp_tag`, and `runtime.seed`.

## Colab

For the fair-comparison multi-agent +I runs, put the generated multi-agent data
under the repo as:

```text
data/highD_multiagent/{train_full,val_full,test_full}
data/exiD_multiagent/{train_full,val_full,test_full}
```

Then run this in Colab:

```bash
!git clone https://github.com/GahyunCathyLee/sota_experiments.git
%cd sota_experiments
!git clone https://github.com/neerjathakkar/PAR.git external/par

!pip -q install pyyaml transformers accelerate sentencepiece
from google.colab import drive
drive.mount('/content/drive')
```

Example `exiD2-1` baseline+I training:

```bash
!python adapters/par/train.py \
  --config configs/models/par_colab.yaml \
  --dataset exiD \
  --feature-mode I \
  --exp-tag exiD2-1 \
  --mode full \
  --multiagent \
  --use-importance \
  --log-interval 10
```

Logs and checkpoints are written to Google Drive:

```text
/content/drive/MyDrive/TrajectoryPrediction/sota_experiments/runs/par/exiD/I/exiD2-1/train.log
/content/drive/MyDrive/TrajectoryPrediction/sota_experiments/ckpts/par/exiD2-1/
```

To continue after a runtime reset:

```bash
!python adapters/par/train.py \
  --config configs/models/par_colab.yaml \
  --dataset exiD \
  --feature-mode I \
  --exp-tag exiD2-1 \
  --mode full \
  --multiagent \
  --use-importance \
  --log-interval 10 \
  --resume /content/drive/MyDrive/TrajectoryPrediction/sota_experiments/ckpts/par/exiD2-1/last.pt
```
