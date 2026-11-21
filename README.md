# rs-detect

**D435 场景相机 + D405 腕部相机** 的双路感知与吸盘抓取工具包：扫描分路、稳定出流、标定、YOLO 检测、Cell C 臂控与抓取闭环，一套 Web / 桌面 UI 跑通现场联调。

```text
D435（场景）粗定位 ──► 腕部到位 ──► D405 精定位 / 多视点 ──► RANSAC 吸点 ──► 门控下吸
         │                                                      ▲
         └──────────────────── 一致性门控 ───────────────────────┘
```

| 角色 | 传感器 | 职责 |
|------|--------|------|
| 场景 | RealSense **D435** | 粗定位、工作空间门控 |
| 腕部 | RealSense **D405** | 终测距与精定位（不以两路深度盲平均） |
| 执行 | Cell C **CS66** | Three.js URDF 可视化 + dry-run / 实机下发 |

无实物相机时自动启用 **Mock D405/D435**（`data/mock/` + `config/mock.yaml`），可先联调 UI 与管线。

---

## 功能一览

- **M0 出流**：按 SN 扫描分路；腕部 / 场景独立预览；拍照与录像
- **M1 标定容器**：SDK 自动内参；用户可编辑并持久化 4×4 外参
- **检测通路**：YOLO（可选）→ Mask→点云 → 单帧 RANSAC 吸点；无权重时 HSV / Mock ROI
- **臂控**：Cell C CS66 Three.js 画布、关节点动 / Home、相机 CAD 叠加与图层
- **抓取面板**：Grasp Agent 状态机、幽灵预览、可配置 S0 站立点 / J6、M0 流配置
- **双 UI**：FastAPI Web（Vite + React + Ant Design）与 PyQt Fluent 桌面端共用服务层

详细方案见 [`docs/suction_cardboard_plan.md`](docs/suction_cardboard_plan.md)，排期见 [`docs/development_plan.md`](docs/development_plan.md)，过程记录见 [`docs/DEVLOG.md`](docs/DEVLOG.md)。

---

## 技术栈

| 层 | 选型 |
|----|------|
| 核心 | Python ≥ 3.8（建议 3.10+）、`pyrealsense2`、OpenCV、NumPy、PyYAML |
| Web | FastAPI + Uvicorn · Vite / React / Ant Design（`web/src` → `web/dist`） |
| 桌面 | PyQt5 + **PyQt-Fluent-Widgets** |
| 检测 | `ultralytics` YOLO（`pip install -e ".[yolo]"`） |
| 臂 / 场景 | 对齐 **sensors-arm** Cell C；本地可软链 `sensors` |

---

## 快速开始

### Linux（原环境）

```bash
git clone git@github.com:fszhangYi/rs-detect.git
cd rs-detect

python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[desktop]"
# 有 RealSense 时再加：pip install -e ".[realsense]"
# YOLO：pip install -e ".[yolo]"

./start_web.sh
./start_web.sh -p 6666
./start_desktop.sh          # 需 DISPLAY；默认 QT_QPA_PLATFORM=xcb
```

### Windows（本仓兼容）

策略见下文「跨平台兼容」：无 RealSense 时自动 Mock；`sensors` / `sensors-arm` 与本仓同级即可被自动加入 `sys.path`。

```powershell
cd rs-detect
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
# 桌面 UI：pip install -e ".[desktop]"
# 实机相机：pip install -e ".[realsense]"   # 需已装 Intel RealSense SDK

# 前端（Node 20+；有 pnpm 更好，否则 npm）
cd web; npm install; npm run build; cd ..

.\start_web.ps1                 # 默认 http://127.0.0.1:8787/
.\start_web.ps1 -Port 6666
.\start_web.ps1 -NoBrowser
.\start_desktop.ps1             # 需先装 [desktop]
```

前端热更新（API 代理到 8787）：

```bash
cd web && pnpm install && pnpm run dev   # 或 npm install && npm run dev
# → http://127.0.0.1:5174/
```

| 环境变量 | 默认 | 说明 |
|----------|------|------|
| `RS_DETECT_WEB_HOST` | Linux `0.0.0.0` / Win 脚本 `127.0.0.1` | Web 绑定地址 |
| `RS_DETECT_WEB_PORT` | `8787` | Web 端口 |
| `RS_DETECT_OPEN_BROWSER` | `1` | 启动后自动打开浏览器（`0` 关闭） |
| `RS_DETECT_WEB_REBUILD` | `0` | `1` 强制重建前端 |

### 跨平台兼容（策略）

| 问题（Linux 仓常见） | 做法 |
|----------------------|------|
| 只有 `.sh` 启动器 | 增加 `start_web.ps1` / `start_desktop.ps1`；`.sh` 仍保留给 Linux |
| 硬编码 `/home/ubuntu/autodl-tmp/...` | Python `ensure_workspace_imports()` + 启动器按**同级目录**找 `sensors` / `sensors-arm` |
| `pyrealsense2` 装不上 / 无相机 | 改为 optional extra `[realsense]`；扫描失败自动 **Mock D405/D435** |
| PyQt / Fluent 拖垮 Web 安装 | 改为 optional `[desktop]`；Web 只用 FastAPI |
| `rollup-linux-x64-gnu` 锁死前端 | 挪到 `optionalDependencies`；Windows 用 npm/pnpm 本机 rollup |
| `QT_QPA_PLATFORM=xcb` | 仅 Linux 默认；Windows 不设，走 Qt windows 插件 |
| `os.killpg` 停训练 | Windows 用 `terminate`/`kill`；Linux 仍用进程组 |
| 无 `pnpm` | 启动器回退 `npm install` / `npm run build` |

常用入口：

| 页面 | 地址 |
|------|------|
| 相机 / 标定 / 检测 / 训练 / 抓取 | `http://127.0.0.1:8787/` |
| Cell C 臂控 3D | `http://127.0.0.1:8787/#/arm` |

感知管线调试（臂控就绪后再接闭环）：

```bash
python scripts/run_m2_pipeline.py --method auto
```

### YOLO 环境

```bash
./scripts/setup_yolo_env.sh          # ultralytics / torch、基座权重、自检
python scripts/check_yolo_env.py

# 纸板数据：data/yolo/（见该目录 README）
# 训练示例：./scripts/train_yolo_cardboard.sh → models/cardboard_yolo.pt
```

---

## 目录结构

```text
rs-detect/
├── start_web.sh / start_desktop.sh
├── config/                 # cameras / extrinsics / yolo / grasp / pipeline …
├── models/                 # YOLO 权重（gitignore；保留 .gitkeep）
├── data/
│   ├── mock/               # 无硬件时的 Mock 画面
│   ├── captures/           # 拍照
│   └── recordings/         # 录像
├── docs/                   # 方案、计划、DEVLOG
├── scripts/                # 环境、单帧、M2、YOLO 辅助
├── src/rs_detect/          # camera · calib · detect · geometry · motion · grasp · api
└── web/                    # Vite 前端 → dist（FastAPI 托管）
```

与多仓工作区的关系：核心驱动仍在 [`sensors`](https://github.com/fszhangYi/sensors)；臂可视化对齐 [`sensors-arm`](https://github.com/fszhangYi/sensors-arm)；眼在手外标定见 [`rs-calibration`](https://github.com/fszhangYi/rs-calibration)。本仓库专注 **双相机感知 + 吸点 + 抓取联调 UI**。

---

## API 摘要

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/scan` | 扫描并分路 |
| `POST` | `/api/stream/start` | `{serial, role: wrist\|scene}` |
| `POST` | `/api/stream/stop` | 停流 |
| `GET` | `/api/frame.jpg` | `color` / `depth` |
| `POST` | `/api/snapshot` · `/api/record/start\|stop` | 拍照 / 录像 |
| `GET` | `/api/intrinsics` | SDK / Mock 内参 |
| `GET`/`PUT` | `/api/extrinsics` | 外参读写 |
| `GET`/`POST` | `/api/selection` | M1 选中串号联动 |
| `POST` | `/api/pipeline/m2` | D435 粗 → D405 精 → 门控 |
| `POST` | `/api/detect/run` | 单相机检测调试 |
| `GET` | `/api/detect/viz.jpg` | 检测可视化 |

更多抓取服务约定见 [`docs/grasp-service-server-expectations.md`](docs/grasp-service-server-expectations.md)。

---

## 里程碑（实现口径）

| 里程碑 | 内容 | 状态 |
|--------|------|------|
| **M0** | 扫描、出流、拍照、录像 | 已落地 |
| **M1** | SDK 内参 + 用户外参 | 已落地 |
| **检测** | YOLO / HSV / Mock → 吸点 | 通路可用 |
| **臂控** | Cell C Three.js + 下发 | 已落地 |
| **闭环** | 粗定位 → 精定位 ↔ 运动 → 抓取 | 进行中 |

---

## 许可证

内部 / 实验室项目；与 `sensors` 栈部署策略一致。欢迎同工作区姊妹仓联调。
