# Codex Prompts for SOTA Model Adapters

Use these files to start a new adapter implementation session with Codex.

Recommended order:

1. `mtp_go.md`
2. `hivt.md`
3. `trajectronpp.md`
4. `qcnet.md`
5. `simpl.md`
6. `mtrpp.md`
7. `hptr.md`
8. `gameformer.md`
9. `par.md`
10. `qcnext.md`
11. `motionlm.md`

Each model prompt assumes:

- Experiment root: `/home/gahyun/sota_experiments`
- NeighFormer data/code root: `/home/gahyun/neighformer`

The shared experiment contract lives in:

- `adapters/common_schema.md`
- `configs/matrix.yaml`
- `model_registry.yaml`

For a new Codex session, paste one model prompt and ask Codex to continue until
the smoke test passes.
