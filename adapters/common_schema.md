# Common highD/exiD Experiment Schema

The canonical data source is the NeighFormer mmap/npy preprocessing output:

```text
/home/gahyun/neighformer/data/<dataset>/dimI/
```

Supported datasets:

```text
highD
exiD
```

Required files:

```text
x_ego.npy       float32 [N, T_h, 6]
x_nb.npy        float32 [N, T_h, K, 10]
nb_mask.npy     bool    [N, T_h, K]
y.npy           float32 [N, T_f, 2]
x_last_abs.npy  float32 [N, 2]
```

Optional files:

```text
y_vel.npy                float32 [N, T_f, 2]
y_acc.npy                float32 [N, T_f, 2]
meta_recordingId.npy     int32   [N]
meta_trackId.npy         int32   [N]
meta_frame.npy           int32   [N]
scenario_labels.csv
```

Default dimensions:

```text
T_h = 6
T_f = 15
K   = 8 neighbors
```

Ego feature order:

```text
x_ego[..., 0] = x
x_ego[..., 1] = y
x_ego[..., 2] = xVelocity
x_ego[..., 3] = yVelocity
x_ego[..., 4] = xAcceleration
x_ego[..., 5] = yAcceleration
```

Neighbor raw feature order:

```text
x_nb[..., 0] = dx
x_nb[..., 1] = dy
x_nb[..., 2] = dvx
x_nb[..., 3] = dvy
x_nb[..., 4] = dax
x_nb[..., 5] = day
x_nb[..., 6] = s_x
x_nb[..., 7] = s_y
x_nb[..., 8] = dim
x_nb[..., 9] = I
```

Feature modes are defined by selecting neighbor channels from `x_nb.npy`:

```text
baseline = [0, 1, 2, 3, 4, 5]        # dx, dy, dvx, dvy, dax, day
dimI     = [0, 1, 2, 3, 4, 5, 8, 9]  # dx, dy, dvx, dvy, dax, day, dim, I
```

Split files are expected at:

```text
/home/gahyun/neighformer/data/<dataset>/splits/train_indices.npy
/home/gahyun/neighformer/data/<dataset>/splits/val_indices.npy
/home/gahyun/neighformer/data/<dataset>/splits/test_indices.npy
```

Current local note: highD split files are present. exiD arrays are present, but
exiD split files may need to be generated from
`/home/gahyun/neighformer/data/exiD/split.py` before split-based training.

Adapter responsibilities:

1. Load NeighFormer mmap/npy files from the shared data root.
2. Apply the selected feature mode without modifying the source arrays.
3. Convert ego history, neighbor history, neighbor mask, and future target into
   the upstream model's expected scene format.
4. Preserve the target convention `[N, T_f, 2]` in meters.
5. Use `x_last_abs.npy` when an upstream model requires absolute coordinates.
6. Report ADE/FDE/RMSE using the shared evaluator contract.
7. Keep all model-specific glue inside `adapters/<model>/` or a clearly
   documented patch file for `external/<model>/`.
