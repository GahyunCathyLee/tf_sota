# data/

Dataset root for the experiment matrix. Expected layout, identical to
`neighformer/data/`:

```text
data/
  highD/
    preprocess.py       raw CSV -> mmap/npy      (versioned)
    split.py            train/val/test indices   (versioned)
    scenario_label.py   scenario_labels.csv      (versioned)
    dimI/               x_ego.npy, x_nb.npy, ... (ignored, ~6.8 GB)
    splits/             *_indices.npy            (ignored)
    raw/                original CSVs            (ignored)
  exiD/
    ... same structure
```

`.gitignore` versions `data/**/*.py` and ignores everything else, so the
pipeline is reproducible without pushing gigabytes of arrays.

## Provenance

These scripts are **copies** of `neighformer/data/{highD,exiD}/*.py`, taken from
neighformer commit `7138966` (2026-05-26). They are vendored so that a Colab
runtime can regenerate arrays and splits without checking out neighformer.

They are copies, not a submodule, so they can drift. If preprocessing changes on
the neighformer side, re-copy them and note it here — arrays produced by
different versions of these scripts are not comparable.

`neighformer/data/NGSIM/` is not vendored: NGSIM is not part of the
`configs/matrix.yaml` experiment matrix (highD and exiD only).

## PAR Extras

PAR needs fixed-identity neighbour futures to keep its multi-agent
autoregressive token structure. Do not modify `../neighformer`; build a
repo-local PAR data root instead:

```bash
python data/par/preprocess.py --dataset both \
  --source-data-root /home/gahyun/neighformer/data \
  --output-root data/par
```

This symlinks the canonical highD/exiD arrays and splits, then writes PAR-only
arrays under `data/par/<dataset>/dimI/`: `nb_ids.npy`, `nb_attr.npy`,
`x_nb_abs.npy`, `y_nb.npy`, and their masks. The usual `baseline` vs `dimI`
feature selection still comes from `x_nb.npy`; PAR uses `nb_attr.npy` only for
the continuous `[dim, I]` side-channel in `dimI` mode.

## Regenerating

```bash
# arrays: raw CSV -> data/<dataset>/dimI/*.npy
python data/highD/preprocess.py --help
python data/exiD/preprocess.py --help

# scenario labels (needed by evaluate.py --scenario)
python data/exiD/scenario_label.py

# splits: 70/10/20, stratified by event x state at track level
python data/exiD/split.py \
  --label_file data/exiD/dimI/scenario_labels.csv \
  --output_dir data/exiD/splits
```

Current split sizes:

| dataset | train | val | test |
| --- | --- | --- | --- |
| highD | 682,333 | 96,885 | 194,618 |
| exiD | 471,091 | 67,347 | 135,062 |

Note that these scripts default to paths relative to the **neighformer** project
root (e.g. `data/exiD/mmap/scenario_labels.csv`), so pass `--label_file` /
`--output_dir` explicitly as shown above.
