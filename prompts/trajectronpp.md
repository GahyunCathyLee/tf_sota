# Codex Prompt: Implement Trajectron++ Adapter

Work in `/home/gahyun`. Experiment root: `/home/gahyun/sota_experiments`. NeighFormer data/code root: `/home/gahyun/neighformer`.

Read first:

- `prompts/common_adapter_prompt.md`
- `prompts/adapter_done_definition.md`
- `model_registry.yaml`
- `configs/models/trajectronpp.yaml`
- `adapters/trajectronpp/README.md`

Goal: make Trajectron++ train and evaluate on highD/exiD with `baseline` and
`dimI`.

Local context:

- Sibling checkouts exist at `../Trajectron-plus-plus` and `../trajectronPP`.
- Previous inspection showed import can work in the `trajectron++` conda env.
- A likely blocker is highD preprocessing with `standardization=None`.
- Another required change is defining the VEHICLE state schema for `baseline`
  and `dimI`.

Implementation steps:

1. Inspect both sibling checkouts and choose the healthier one.
2. Inspect environment/data classes, preprocessing, config JSON/YAML, train
   entrypoint, and evaluation code.
3. Fix or replace the highD/exiD preprocessing path so `Environment` has valid
   standardization stats.
4. Define state channels:
   - `baseline`: position/velocity/acceleration features needed by Trajectron++
     plus the common dynamic channels.
   - `dimI`: the same plus `dim` and `I`.
5. Keep the common horizon: 6 history frames, 15 future frames.
6. Implement `adapters/trajectronpp/train.py` as a wrapper
   around upstream train/eval.
7. Add local smoke mode. Full training may require A100/Colab GPU.

Validation target:

- Preprocessing creates a usable environment pickle for highD baseline.
- Training starts and reaches the first loss update.
- `dimI` does not silently drop `dim` or `I`.
