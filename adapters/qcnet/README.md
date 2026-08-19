# QCNet Adapter

Runs the official QCNet code from `external/qcnet` on the canonical NeighFormer
highD/exiD npy data.

Status: data conversion and config sweep are implemented. Local full training
was not smoke-verified because this machine's active environment is missing
QCNet/PyG extension dependencies (`torch_cluster`, `torch_scatter`,
`torchvision`).

## Files

| File | Role |
| --- | --- |
| `train.py` | Config-driven training wrapper around official `predictors.QCNet` |
| `evaluate.py` | Checkpoint-driven ADE/FDE/RMSE evaluation with `--scenario` support |
| `dataset.py` | NeighFormer npy -> QCNet Argoverse2-style `HeteroData` |
| `upstream.py` | Locates and imports `external/qcnet` |

Repo-root `train.py` and `evaluate.py` dispatch to this adapter, so Colab can
use the same command style as NeighFormer:

```bash
python train.py --config configs/qcnet/exiD1-4.yaml
python evaluate.py --ckpt ckpts/qcnet/exiD1-4/best.pt --scenario
```

## Configs

`configs/qcnet/highD0-1.yaml` ... `configs/qcnet/exiD1-5.yaml` mirror NeighFormer naming:

- feature `0` = baseline (`dx,dy,dvx,dvy,dax,day`)
- feature `1` = dimI (`dx,dy,dvx,dvy,dax,day,dim,I`)
- seed index `1=42`, `2=1234`, `3=3407`, `4=0`, `5=777`

Shared QCNet hyperparameters live in `configs/models/qcnet.yaml`.

## Data Mapping

Agent node 0 is the ego/focal vehicle. Neighbor nodes are reconstructed from
`x_nb[..., 0:6]` by adding relative position/velocity to the ego history. Only
the ego node has `predict_mask=True` for the future horizon, matching the
NeighFormer metric target.

For `dimI`, the adapter replaces QCNet's official agent encoder with an
equivalent adapter subclass whose agent-token Fourier input is extended from
4 continuous values to 6: the original motion features plus `[dim, I]`.

For highD, the adapter reads SIMPL lane graph caches from
`<data-root>/highD/simpl_lane_graph` and converts nearby lane segments to
QCNet's `map_polygon`/`map_point` schema. Build that cache with:

```bash
python scripts/build_simpl_lane_graph.py --dataset highD --data-root /path/to/data
```

If the cache is missing, or for exiD, the adapter falls back to the original
single pseudo-lane token so QCNet's map attention path remains valid.

## Dependencies

The official QCNet import path requires PyTorch, PyG, PyTorch Lightning,
`torch_cluster`, `torch_scatter`, and `torchvision`.

For Colab, after installing a compatible PyTorch/PyG stack:

```bash
pip install -q pytorch-lightning torchvision
pip install -q torch-geometric
# Install torch_cluster / torch_scatter wheels matching the Colab torch+CUDA build.
```

Data-only validation can be run without importing the QCNet model:

```bash
python train.py --config configs/qcnet/exiD1-4.yaml --check-data
```
