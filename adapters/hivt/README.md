# HiVT Adapter

Goal: convert NeighFormer highD/exiD npy samples into HiVT temporal graph data.

First implementation tasks:

1. Clone upstream into `external/hivt`.
2. Reuse HiVT model code and replace the Argoverse dataset class.
3. Feed `baseline` or `dimI` agent features into the node feature encoder.
4. Keep the shared metrics from `adapters/common.py`.
