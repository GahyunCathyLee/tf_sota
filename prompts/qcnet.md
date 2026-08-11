# Codex Prompt: Implement QCNet Adapter

Work in `/home/gahyun`. Experiment root: `/home/gahyun/sota_experiments`. NeighFormer data/code root: `/home/gahyun/neighformer`.

Read first:

- `prompts/common_adapter_prompt.md`
- `prompts/adapter_done_definition.md`
- `model_registry.yaml`
- `configs/models/qcnet.yaml`
- `adapters/qcnet/README.md`

Goal: implement QCNet highD/exiD experiments with `baseline` and `dimI`.

Upstream:

- Official repo: `https://github.com/ZikangZhou/QCNet.git`
- Expected local path under experiment root: `external/qcnet`

Implementation steps:

1. Clone upstream if missing.
2. Inspect data modules for Argoverse/Waymo, feature encoders, query-centric
   scene construction, and metrics.
3. Add a highD/exiD scenario adapter that creates agent tokens from the common
   NeighFormer npy input.
4. Use `nb_mask` and distance-based edges for interactions.
5. Append `dim` and `I` to agent attributes in `dimI` mode and adjust embedding
   dimensions.
6. Start with marginal trajectory prediction before attempting QCNeXt-style
   joint prediction.
7. Add smoke config for tiny subsets.

Validation target:

- highD baseline and dimI smoke runs reach a loss value.
- exiD data loads.
- Output metrics are written to `metrics.json`.
