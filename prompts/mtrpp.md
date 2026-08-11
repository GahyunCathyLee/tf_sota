# Codex Prompt: Implement MTR++ Adapter

Work in `/home/gahyun`. Experiment root: `/home/gahyun/sota_experiments`. NeighFormer data/code root: `/home/gahyun/neighformer`.

Read first:

- `prompts/common_adapter_prompt.md`
- `prompts/adapter_done_definition.md`
- `model_registry.yaml`
- `configs/models/mtrpp.yaml`
- `adapters/mtrpp/README.md`

Goal: use the MTR codebase as the base for MTR++-style highD/exiD experiments.

Upstream:

- Official MTR repo: `https://github.com/sshaoshuai/MTR.git`
- Expected local path under experiment root: `external/mtrpp`

Implementation steps:

1. Clone upstream if missing.
2. Inspect Waymo preprocessing, scene info format, object feature tensors, and
   train/eval scripts.
3. Create a converter from common NeighFormer npy files to MTR-style scene info files.
4. Add `dim` and `I` to object trajectory attributes for `dimI`.
5. Keep map features minimal or create highway-lane placeholders if required.
6. Implement `train.py` wrapper to call upstream training on generated scene
   infos.
7. Add a local smoke run with very small batch size.

Validation target:

- Converter produces scene info files for highD baseline.
- Model starts training on a tiny subset.
- Full training is marked A100-recommended.
