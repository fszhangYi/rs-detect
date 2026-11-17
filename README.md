# rs-detect

D435（场景）+ D405（腕部）双相机感知工具：扫描分路、稳定出流、拍照/录像、SDK 内参、用户外参、单帧检测→平面吸点。

## 技术栈

- **核心**：Python 3.8+、`pyrealsense2`、OpenCV
- **桌面**：PyQt5 + **PyQt-Fluent-Widgets**
- **Web**：FastAPI + **Vite / React / Ant Design**（`web/src` → `web/dist`，对齐 file-viewer）
- **检测（可选）**：`ultralytics` YOLO（`pip install -e ".[yolo]"`）；无权重时回退 HSV / Mock ROI
- **版本**：**Git** 管控；开发过程见 `docs/DEVLOG.md`

## 当前里程碑（实现口径）

| 里程碑 | 内容 |
|--------|------|
| **M0** | 自动扫描 RealSense；D405→腕部、D435→场景；稳定出流；拍照；录像 |
| **M1** | 内参由 RealSense SDK 自动获取；外参由用户输入并保存 |
| **臂控** | Cell C CS66（对齐 sensors-arm / DCS huiling）：Three.js URDF + dry-run 下发 |
| **感知闭环** | D435 粗定位下发 → D405 精定位↔运动闭环至抓取点（进行中） |

> 算法总方案仍见 `docs/suction_cardboard_plan.md`；长期开发排期见 `docs/development_plan.md`。

## 快速开始

在项目根目录执行（脚本会自动 `source .venv`、设置 `PYTHONPATH` 等）：

```bash
cd /path/to/rs-detect

# Web UI（约 3 秒后自动打开浏览器；也可用顶栏「重启 / 安全关闭」）
./start_web.sh
./start_web.sh -p 6666          # 指定端口
# http://127.0.0.1:8787/  （或 -p 指定的端口）

# 桌面 Fluent UI（需图形界面 / DISPLAY）
./start_desktop.sh

# Cell C 机械臂 3D（主界面 Tab / 前端路由）
# 浏览器打开 http://127.0.0.1:8787/#/arm
# Connect → Arm → Home / 拖关节下发

# 感知调试（臂控通过后再接闭环）
python scripts/run_m2_pipeline.py --method auto
```

可选环境变量：

| 变量 | 默认 | 说明 |
|------|------|------|
| `RS_DETECT_WEB_HOST` | `0.0.0.0` | Web 绑定地址 |
| `RS_DETECT_WEB_PORT` | `8787` | Web 端口 |
| `RS_DETECT_OPEN_BROWSER` | `1` | 启动 3 秒后自动打开浏览器（`0` 关闭） |
| `RS_DETECT_WEB_REBUILD` | `0` | `1` 强制重建前端（`cd web && pnpm run build`） |
| `QT_QPA_PLATFORM` | `xcb`（仅桌面脚本） | Qt 平台插件 |

前端开发（热更新，API 代理到 8787）：

```bash
cd web && pnpm install && pnpm run dev   # http://127.0.0.1:5174/
```

无实物相机时自动提供 **Mock D405/D435**，画面来自 `data/mock/`（见 `config/mock.yaml`），便于联调 UI 与 M2。

### YOLO 环境

```bash
./scripts/setup_yolo_env.sh          # 安装 ultralytics/torch，下载 yolov8n 基座，自检
python scripts/check_yolo_env.py     # 仅检查
```

纸板数据放 `data/yolo/`（见该目录 README）。快速标注用配套仓库 **`../fast-label`**（导出到本目录后）再跑 `./scripts/train_yolo_cardboard.sh`（自动写入 `models/cardboard_yolo.pt`），并在 `config/yolo.yaml` 配置 `method` / `device`。

## 目录

```text
rs-detect/
├── start_web.sh / start_desktop.sh
├── config/           # cameras.yaml, extrinsics.yaml, yolo.yaml
├── models/           # YOLO 权重（gitignore）
├── data/captures/    # 拍照
├── data/recordings/  # 录像 mp4
├── docs/             # 方案 / 计划 / DEVLOG
├── scripts/          # env + run_single_frame.py
├── src/rs_detect/    # camera / calib / detect / geometry
└── web/              # Vite 前端：src/（ES modules）→ dist/（FastAPI 托管）
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
- `GET|POST /api/selection`
- `POST /api/pipeline/m2` — D435 粗 → D405 精 → 门控（主路径）
- `POST /api/pipeline/coarse` · `POST /api/pipeline/fine` · `GET /api/pipeline/last`
- `GET /api/detect/config` · `POST /api/detect/run`（单相机调试）· `GET /api/detect/viz.jpg`
