# HPTR Adapter

Goal: convert highD/exiD agents into HPTR heterogeneous polyline inputs.

First implementation tasks:

1. Clone upstream into `external/hptr`.
2. Create agent polylines from NeighFormer npy history.
3. Add `dim` and `I` as vehicle polyline attributes.
4. Run local smoke tests with reduced batch size before cloud training.
