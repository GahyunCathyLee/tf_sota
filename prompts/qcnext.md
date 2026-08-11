# Codex Prompt: Implement QCNeXt Adapter

Work in `/home/gahyun`. Experiment root: `/home/gahyun/sota_experiments`. NeighFormer data/code root: `/home/gahyun/neighformer`.

Read first:

- `prompts/common_adapter_prompt.md`
- `prompts/adapter_done_definition.md`
- `model_registry.yaml`
- `configs/models/qcnext.yaml`
- `adapters/qcnext/README.md`
- `prompts/qcnet.md`

Goal: implement QCNeXt-style joint multi-agent prediction after QCNet works.

Important constraint:

- Do not start QCNeXt before the QCNet highD/exiD adapter has a passing smoke
  run.
- No separate standalone official QCNeXt repo was found in the initial scan.
  Treat this as a QCNet-derived implementation.

Implementation steps:

1. Reuse the QCNet upstream checkout and highD/exiD data adapter.
2. Identify what must change for joint multi-agent prediction.
3. Add a joint decoder/head while preserving baseline/dimI input handling.
4. Evaluate ego-only metrics for paper comparability and optionally joint-agent
   metrics for model diagnostics.
5. Keep full training on A100 unless local profiling is acceptable.

Validation target:

- QCNet adapter already passes.
- QCNeXt smoke run produces joint predictions.
- Ego ADE/FDE/RMSE are written in the shared format.
