# MTR++ Adapter

Goal: use the MTR codebase as the implementation base for MTR++-style
multi-agent experiments.

First implementation tasks:

1. Clone upstream into `external/mtrpp`.
2. Convert highD/exiD NeighFormer npy samples into MTR scene info records.
3. Extend object trajectory attributes for `dimI`.
4. Prefer A100-class GPU for full training.
