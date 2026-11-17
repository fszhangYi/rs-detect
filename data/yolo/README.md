# YOLO 纸板数据集目录

```text
data/yolo/
├── dataset.yaml          # Ultralytics 数据配置（类别 cardboard）
├── images/train|val/     # jpg/png
└── labels/train|val/     # YOLO txt：class cx cy w h（归一化 0–1）
```

## 标注

推荐用配套工具 **[fast-label](../../../fast-label)**（与 `rs-detect` 同级目录）：浏览器画框 → 一键导出到本目录。

- 检测框即可（当前推理 `detect_yolo` 支持 box→mask；有 seg 权重则用 mask）。
- 类别名：`cardboard`（id `0`），与 `dataset.yaml` / `config/yolo.yaml` 的 `class_filter` 对齐。

## 训练

```bash
# 需已 ./scripts/setup_yolo_env.sh，且 images/labels 已标注
./scripts/train_yolo_cardboard.sh
# 可选: ./scripts/train_yolo_cardboard.sh --epochs 80 --device 0 --batch 32
```

脚本会训练并把 `best.pt` 安装到 `models/cardboard_yolo.pt`。之后 `config/yolo.yaml` 用 `method: yolo`（或 `auto` 且权重存在）。
