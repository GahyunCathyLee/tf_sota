# Codex Prompt: Implement HiVT Adapter

Work in `/home/gahyun`. Experiment root: `/home/gahyun/sota_experiments`. NeighFormer data/code root: `/home/gahyun/neighformer`.

Read first:

- `prompts/common_adapter_prompt.md`
- `prompts/adapter_done_definition.md`
- `model_registry.yaml`
- `configs/models/hivt.yaml`
- `adapters/hivt/README.md`

Goal: implement `adapters/hivt/train.py` for highD/exiD
`baseline` and `dimI`.

Upstream:

- Official repo: `https://github.com/ZikangZhou/HiVT.git`
- Expected local path under experiment root: `external/hivt`

Implementation steps:

1. Clone upstream if missing.
2. Inspect HiVT's original Argoverse dataset class, collate function, model
   constructor, Lightning/training entrypoint, and metric code.
3. Build a highD/exiD dataset wrapper that converts `x_ego`, `x_nb`, `nb_mask`,
   and `y` into HiVT temporal graph data.
4. Use `nb_mask` and distance-based edges to create interaction edges when possible.
5. Ensure input embedding dimensions differ for 6-channel neighbor `baseline` and
   8-channel neighbor `dimI`.
6. Disable or stub lane/map features only if the upstream allows agent-only
   training; otherwise create minimal highway lane polylines from available data.
7. Add smoke training with a tiny subset.

Validation target:

- highD baseline smoke run computes loss.
- highD dimI smoke run confirms 8 selected neighbor channels reach the model.
- exiD files load without shape errors.
