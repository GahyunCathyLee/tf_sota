# HiVT Adapter

This adapter runs the official HiVT implementation from `external/hivt` on the
NeighFormer highD/exiD numpy splits.

## Source

```bash
git clone https://github.com/ZikangZhou/HiVT.git external/hivt
```

The adapter imports the upstream `models.hivt.HiVT` class directly and applies
small runtime compatibility patches for the current PyTorch Geometric,
PyTorch Lightning, and torchmetrics versions.

## Data

- `baseline`: history node features are `[dx, dy, vx, vy, ax, ay]`.
- `dimI`: appends neighbor `[dim, I]`, giving node dim 8.
- `dx, dy` follow the official HiVT convention: frame-to-frame history
  displacement, with zero at beginning-of-sequence steps.
- `positions` stay as scene coordinates for actor/actor edges and future target
  construction.
- Future loss is ego-only for the canonical NeighFormer arrays; neighbor future
  steps are masked unless future arrays are added later.
- highD uses the SIMPL lane graph cache at
  `{data_root}/highD/simpl_lane_graph`; exiD falls back to a pseudo straight
  lane until an exiD map cache exists.

## Commands

```bash
python train.py --config configs/hivt/highD0-1.yaml --data-root /path/to/neighformer/data --mode smoke
python train.py --config configs/hivt/highD1-1.yaml --data-root /path/to/neighformer/data --mode check-data
python evaluate.py --ckpt ckpts/hivt/highD0-1/best.pt --data-root /path/to/neighformer/data --max-samples 512
```

Per-run configs live in `configs/hivt/` and mirror the existing seed/features:
`0 = baseline`, `1 = dimI`, seed indices `1..5`.
