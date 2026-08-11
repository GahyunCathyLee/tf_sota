# QCNet Adapter

Goal: run QCNet marginal prediction on highD/exiD with `baseline` and `dimI`.

First implementation tasks:

1. Clone upstream into `external/qcnet`.
2. Build a highD/exiD scenario adapter compatible with QCNet data modules.
3. Append `dim` and `I` to agent token features for `dimI`.
4. Validate one-epoch smoke training before full GPU runs.
