# SIMPL Adapter

Runs the official SIMPL model from `external/simpl` on the canonical
NeighFormer highD/exiD npy data.

Status: implemented with real highD/exiD lane graph caches, with a straight
pseudo-lane fallback if the cache is missing. Use `configs/simpl/<run>.yaml` to
avoid colliding with the QCNet top-level configs.

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

## Lane Graph Cache

Build once after the NeighFormer npy data is available:

```bash
python scripts/build_simpl_lane_graph.py \
  --dataset all \
  --data-root /content/drive/MyDrive/TrajectoryPrediction/neighformer/data \
  --map-dir data/exiD/maps \
  --mmap-name dimI
```

The cache is written to:

```text
<data-root>/highD/simpl_lane_graph/
<data-root>/exiD/simpl_lane_graph/
```

Those folders are portable. On Colab, upload/copy them together with the npy
data under the same `--data-root`. If the cache lives somewhere else, override:

```bash
python train.py --config configs/simpl/exiD1-4.yaml \
  --data-root /content/drive/MyDrive/TrajectoryPrediction/neighformer/data \
  --lane-cache-root /content/drive/MyDrive/TrajectoryPrediction/lane_graph/exiD
```

## Mapping

Actor node 0 is the ego/focal vehicle. Neighbor nodes are reconstructed from
`x_nb[..., 0:6]` by adding relative position/velocity/acceleration to the ego
history. Only the ego node has future supervision.

- `baseline`: actor input channels are `[x, y, vx, vy, ax, ay]`
- `dimI`: actor input channels are `[x, y, vx, vy, ax, ay, dim, I]`

highD lane caches are derived from `upperLaneMarkings` and `lowerLaneMarkings`
after applying the same upper-direction coordinate normalization used by the
NeighFormer highD preprocessing. exiD lane caches are derived from Lanelet2 OSM
relations and per-sample ego headings reconstructed from raw tracks.

The official SIMPL network is otherwise imported unchanged.
