# Codex Prompt: Implement MotionLM-Style Adapter

Work in `/home/gahyun`. Experiment root: `/home/gahyun/sota_experiments`. NeighFormer data/code root: `/home/gahyun/neighformer`.

Read first:

- `prompts/common_adapter_prompt.md`
- `prompts/adapter_done_definition.md`
- `model_registry.yaml`
- `configs/models/motionlm.yaml`
- `adapters/motionlm/README.md`

Goal: implement a compact MotionLM-style trajectory-token language model for
highD/exiD.

Important constraint:

- No official public MotionLM training code was found in the initial scan.
- Treat this as a reimplementation inspired by the paper, not an official-code
  reproduction.

Implementation steps:

1. Inspect the MotionLM paper/project description if needed.
2. Design a simple tokenization strategy for future trajectory deltas.
3. Encode observed multi-agent history from the common NeighFormer npy input.
4. Add `dim` and `I` as side-channel embeddings in `dimI`.
5. Implement a compact autoregressive transformer that can run smoke tests
   locally.
6. Report clearly that this is not official MotionLM unless official code is
   later found and integrated.

Validation target:

- Tokenizer round-trip or reconstruction sanity check.
- highD baseline one-batch training.
- highD dimI confirms side-channel use.
