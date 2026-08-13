# SIMPL Adapter

Runs the official SIMPL model from `external/simpl` on the canonical
NeighFormer highD/exiD npy data.

Status: implemented with a map-light pseudo-lane adapter. Use
`configs/simpl/<run>.yaml` to avoid colliding with the QCNet top-level configs.

## Usage

```bash
python train.py --config configs/simpl/exiD1-4.yaml
python evaluate.py --ckpt ckpts/simpl/exiD1-4/best.pt --scenario
```

For smoke runs:

```bash
python train.py --config configs/simpl/highD0-1.yaml \
  --data-root /home/gahyun/neighformer/data \
  --max-train-samples 64 --max-eval-samples 32 \
  --epochs 1 --batch-size 8 --num-workers 0
```

## Mapping

Actor node 0 is the ego/focal vehicle. Neighbor nodes are reconstructed from
`x_nb[..., 0:6]` by adding relative position/velocity/acceleration to the ego
history. Only the ego node has future supervision.

- `baseline`: actor input channels are `[x, y, vx, vy, ax, ay]`
- `dimI`: actor input channels are `[x, y, vx, vy, ax, ay, dim, I]`

highD/exiD do not include lane maps, so each scene receives one straight
pseudo-lane token. The official SIMPL network is otherwise imported unchanged.
