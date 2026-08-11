# MotionLM Adapter

Goal: implement a compact MotionLM-style model for highD/exiD.

No official public training repository was found during the initial scan, so this
is tracked as a reimplementation target.

First implementation tasks:

1. Build a trajectory tokenizer for multi-agent future motion.
2. Add side-channel embeddings for `baseline` and `dimI`.
3. Implement an autoregressive transformer with shared ADE/FDE/RMSE evaluation.
