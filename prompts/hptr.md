# Codex Prompt: Implement HPTR Adapter

Work in `/home/gahyun`. Experiment root: `/home/gahyun/sota_experiments`. NeighFormer data/code root: `/home/gahyun/neighformer`.

Read first:

- `prompts/common_adapter_prompt.md`
- `prompts/adapter_done_definition.md`
- `model_registry.yaml`
- `configs/models/hptr.yaml`
- `adapters/hptr/README.md`

Goal: run HPTR on highD/exiD with `baseline` and `dimI`.

Upstream:

- Official repo: `https://github.com/zhejz/HPTR.git`
- Expected local path under experiment root: `external/hptr`

Implementation steps:

1. Clone upstream if missing.
2. Inspect dataset format, heterogeneous polyline construction, relative pose
   encoding, model input dimensions, and training script.
3. Convert each highD/exiD sample into agent polylines.
4. Add `dim` and `I` as vehicle polyline attributes in `dimI`.
5. Decide whether simple highway lane polylines are needed for map inputs.
6. Build a smoke training path with reduced batch size.

Validation target:

- highD baseline converter works.
- dimI features are present in the encoded vehicle polyline.
- Training reaches one loss update.
