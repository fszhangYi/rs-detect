# rs-detect

D435（场景）+ D405（腕部）双相机感知工具：扫描分路、稳定出流、拍照/录像、SDK 内参、用户外参。

## 技术栈

- **核心**：Python 3.8+、`pyrealsense2`、OpenCV
- **桌面**：PyQt5 + **PyQt-Fluent-Widgets**
- **Web**：FastAPI + **Ant Design**（`web/public`）
- **版本**：**Git** 管控；开发过程见 `docs/DEVLOG.md`

## 当前里程碑（实现口径）

| 里程碑 | 内容 |
|--------|------|
| **M1** | 自动扫描 RealSense；D405→腕部待选区、D435→场景待选区；稳定出流；拍照；录像 |
| **M2** | 内参由 RealSense SDK 自动获取；外参由用户输入并保存 |

> 算法总方案仍见 `docs/suction_cardboard_plan.md`；长期开发排期见 `docs/development_plan.md`。

## 快速开始

```bash
cd /home/ubuntu/autodl-tmp/rs-detect
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# Web UI（推荐先验证）
./scripts/run_web.sh
# 浏览器打开 http://127.0.0.1:8787/

# 桌面 Fluent UI（需图形界面）
./scripts/run_desktop.sh
```

无实物相机时自动提供 **Mock D405/D435**，便于联调 UI 与存图/录像。

## 目录

```text
rs-detect/
├── config/           # cameras.yaml, extrinsics.yaml
├── data/captures/    # 拍照
├── data/recordings/  # 录像 mp4
├── docs/             # 方案 / 计划 / DEVLOG
├── scripts/          # 启动脚本
├── src/rs_detect/    # Python 包
└── web/public/       # Ant Design Web UI
```

## API 摘要

- `GET /api/scan` — 扫描并分路
- `POST /api/stream/start` — `{serial, role: wrist|scene}`
- `POST /api/stream/stop?serial=`
- `GET /api/frame.jpg?serial=&which=color|depth`
- `POST /api/snapshot?serial=`
- `POST /api/record/start|stop?serial=`
- `GET /api/intrinsics?serial=`
- `GET|PUT /api/extrinsics`
