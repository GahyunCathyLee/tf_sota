# SIMPL Adapter

Goal: run SIMPL as a lightweight multi-agent baseline on highD/exiD.

First implementation tasks:

1. Clone upstream into `external/simpl`.
2. Replace Argoverse feature loading with the common NeighFormer npy loader.
3. Adjust input channel dimensions for `baseline` and `dimI`.
