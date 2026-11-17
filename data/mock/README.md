# Mock camera frames (Grasp)

Configured by `config/mock.yaml` (paths relative to repo root).

**Current source (scene + wrist):**

`data/capture/336222075436_20261116_203908_347_{color,depth,meta}`

Uploads always include **color_b64 + depth_b64** (uint16 mm PNG) + color_intrinsics.

Legacy copies under `data/mock/d435_*` / `d405_*` are unused unless `mock.yaml` points back to them.
