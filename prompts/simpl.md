# Codex Prompt: Implement SIMPL Adapter

Work in `/home/gahyun`. Experiment root: `/home/gahyun/sota_experiments`. NeighFormer data/code root: `/home/gahyun/neighformer`.

Read first:

- `prompts/common_adapter_prompt.md`
- `prompts/adapter_done_definition.md`
- `model_registry.yaml`
- `configs/models/simpl.yaml`
- `adapters/simpl/README.md`

Goal: implement SIMPL as a lightweight SOTA-style baseline on highD/exiD.

Upstream:

- Official repo: `https://github.com/HKUST-Aerial-Robotics/SIMPL.git`
- Expected local path under experiment root: `external/simpl`

Implementation steps:

1. Clone upstream if missing.
2. Inspect the original feature extraction and model input format.
3. Replace Argoverse feature loading with the common NeighFormer npy loader.
4. Map each agent history to the SIMPL agent feature format.
5. Wire 6-channel neighbor `baseline` and 8-channel neighbor `dimI` input dimensions.
6. Keep the first implementation map-light or map-free if possible.
7. Add smoke training.

Validation target:

- Fast smoke run on highD baseline.
- Correct channel count in highD dimI.
- exiD baseline loads.
