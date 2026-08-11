# Codex Prompt: Implement PAR Adapter

Work in `/home/gahyun`. Experiment root: `/home/gahyun/sota_experiments`. NeighFormer data/code root: `/home/gahyun/neighformer`.

Read first:

- `prompts/common_adapter_prompt.md`
- `prompts/adapter_done_definition.md`
- `model_registry.yaml`
- `configs/models/par.yaml`
- `adapters/par/README.md`

Goal: adapt PAR to highD/exiD trajectory prediction.

Upstream:

- Official repo: `https://github.com/neerjathakkar/PAR.git`
- Expected local path under experiment root: `external/par`

Implementation steps:

1. Clone upstream if missing.
2. Inspect the NuScenes car trajectory task and any generic sequence modeling
   utilities.
3. Create a highD/exiD datamodule from common NeighFormer npy.
4. Decide and document whether `dim` and `I` are:
   - continuous side-channel embeddings, or
   - discretized tokens.
5. Implement the simpler continuous side-channel path first unless upstream
   strongly favors tokenization.
6. Add smoke training.

Validation target:

- highD baseline smoke run computes loss.
- highD dimI uses the extra two channels.
- exiD loads.
