# GameFormer Adapter

Goal: run the interaction prediction component of GameFormer on highD/exiD.

First implementation tasks:

1. Clone upstream into `external/gameformer`.
2. Use `interaction_prediction` instead of planning code.
3. Build WOMD-like scene samples from highD/exiD NeighFormer npy data.
4. Keep full runs for A100-class GPU unless smoke profiling says otherwise.
