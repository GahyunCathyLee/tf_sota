# Codex Prompt: Implement GameFormer Adapter

Work in `/home/gahyun`. Experiment root: `/home/gahyun/sota_experiments`. NeighFormer data/code root: `/home/gahyun/neighformer`.

Read first:

- `prompts/common_adapter_prompt.md`
- `prompts/adapter_done_definition.md`
- `model_registry.yaml`
- `configs/models/gameformer.yaml`
- `adapters/gameformer/README.md`

Goal: use GameFormer's interaction prediction component for highD/exiD.

Upstream:

- Official repo: `https://github.com/MCZhi/GameFormer.git`
- Expected local path under experiment root: `external/gameformer`

Implementation steps:

1. Clone upstream if missing.
2. Inspect the `interaction_prediction` code path before the planning path.
3. Identify expected Waymo/nuPlan-like scene tensors.
4. Create a converter from common NeighFormer npy samples to the closest upstream scene
   tensor format.
5. Append `dim` and `I` to agent state features in `dimI`.
6. Add smoke mode and keep full training for stronger GPUs.

Validation target:

- Interaction prediction path imports.
- highD baseline scene conversion works.
- A one-batch smoke run computes loss.
