# Trajectron++ Adapter

Goal: make Trajectron++ train on highD/exiD with `baseline` and `dimI`.

First implementation tasks:

1. Clone upstream into `external/trajectronpp` or reuse
   `../Trajectron-plus-plus`.
2. Fix highD/exiD environment standardization.
3. Define VEHICLE state schemas for `baseline` and `dimI`.
4. Profile local training; move full training to A100 if runtime is too high.
