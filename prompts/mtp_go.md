# Codex Prompt: Implement MTP-GO Adapter

Work in `/home/gahyun`. Experiment root: `/home/gahyun/sota_experiments`. NeighFormer data/code root: `/home/gahyun/neighformer`.

Read first:

- `prompts/common_adapter_prompt.md`
- `prompts/adapter_done_definition.md`
- `model_registry.yaml`
- `configs/models/mtp_go.yaml`
- `adapters/mtp_go/README.md`

Goal: implement `adapters/mtp_go/train.py` so MTP-GO can run
`baseline` and `dimI` experiments on highD and exiD.

Local context:

- A sibling checkout exists at `../mtp-go`.
- The official upstream is `https://github.com/westny/mtp-go.git`.
- Previous inspection found highD/highD-imp hooks in the repo, but the current
  local environment was missing torch/PyG/Lightning.

Implementation steps:

1. Inspect `../mtp-go` first, especially `train.py`, `datamodule.py`,
   dataset conversion code, model input feature definitions, and requirements.
2. Decide whether to reuse `../mtp-go` directly or clone/copy into
   `external/mtp_go`.
3. Create a thin adapter that maps:
   - `baseline` to the MTP-GO highD-style feature set without importance.
   - `dimI` to a feature set containing dx, dy, dvx, dvy, dax, day, dim, I.
4. Add exiD support by reusing the same common NeighFormer npy schema.
5. Keep a smoke mode that limits batches/epochs.
6. Write `metrics.json`, `run_config.yaml`, and dependency info into the output
   directory.

Smoke commands to make work:

```bash
python adapters/mtp_go/train.py --config configs/models/mtp_go.yaml --dataset highD --feature-mode baseline --data-root /home/gahyun/neighformer/data --output-dir runs/mtp_go/highD/baseline
python adapters/mtp_go/train.py --config configs/models/mtp_go.yaml --dataset highD --feature-mode dimI --data-root /home/gahyun/neighformer/data --output-dir runs/mtp_go/highD/dimI
```

Stop after smoke training works. Do not start full training automatically.
