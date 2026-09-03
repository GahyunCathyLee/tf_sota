# MotionLM Adapter

Goal: integrate MotionLM for highD/exiD while preserving the original model as
much as possible.

No official public MotionLM training repository was found during the latest scan.
Do not add a `*-style` reimplementation here unless the experiment plan changes;
this adapter should wait for an official or faithful upstream implementation.

First implementation tasks:

1. Find or obtain original MotionLM training/model code.
2. Inspect its dataset format and tokenization.
3. Adapt only the input/data conversion path for highD/exiD baseline and dimI.
