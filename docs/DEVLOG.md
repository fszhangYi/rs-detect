# rs-detect 开发过程记录

> 约定：有可运行增量即 **及时 git commit**；本文件按时间追加，记录决策与验收。

---

## 2026-11-12 — 启动 M1 / M2 实现

### 背景

- 技术方案见 `suction_cardboard_plan.md`，总排期见 `development_plan.md`。
- 本次按产品实现口径推进：
  - **M1**：自动扫描 RealSense；D405→腕部待选区、D435→场景待选区；稳定出流；拍照；录像。
  - **M2**：内参 SDK 自动获取；外参用户输入。
- UI 技术栈：**PyQt + PyQtFluentWidgets + Web(Ant Design)**；Git 管控。

### 本轮完成

1. 初始化仓库骨架：`src/rs_detect`、`config/`、`web/public/`、`scripts/`、`docs/`。
2. **相机层**
   - `scanner.py`：枚举设备并按型号分路；无硬件时 Mock D405/D435。
   - `streamer.py`：单串号稳定出流线程；Align 深度到彩色；Mock 动画画面。
   - `recorder.py`：单帧拍照（color/depth/meta）+ mp4 录像。
3. **标定层（M2）**
   - 内参：从 live profile / 短开 pipeline 读 SDK intrinsics。
   - 外参：`config/extrinsics.yaml`，桌面与 Web 均可编辑 4×4。
4. **服务层** `DetectService`：桌面与 FastAPI 共用。
5. **Web**：FastAPI `:8787` + Ant Design（CDN React）双栏出流 / 外参页。
6. **桌面**：FluentWindow 双栏相机页 + 外参页。

### 环境备注

- 当前开发机 **未插 RealSense**，`rs-enumerate-devices` 为 0；联调走 Mock。
- 本机已装 `pyrealsense2`、`PyQt-Fluent-Widgets`（项目 `.venv`，system-site-packages 复用系统 PyQt5）。
- 系统 `sudo` 不可用，Web 前端采用 **Ant Design CDN 静态页**（`web/public`），避免强依赖本机 npm 构建；后续可再加 Vite 工程。

### 验收清单（本轮）

- [x] 扫描结果分为腕部 / 场景待选区
- [x] 两路可独立出流（Mock）
- [x] 拍照写入 `data/captures/`
- [x] 录像写入 `data/recordings/`
- [x] 内参接口返回 SDK/Mock 数值
- [x] 外参可保存到 yaml
- [x] 服务层 smoke：`SMOKE OK`（双路出流 + 拍照 + 录像 18 帧 + 内参 + 外参）
- [x] Web API smoke：`/api/health` `/api/scan` `/api/stream/start` `/api/frame.jpg` `/api/snapshot` `/api/intrinsics` 均 200；首页 200
- [ ] 真机 D405/D435 插上后复验（待现场）
- [ ] 桌面 GUI 在有 DISPLAY 的工控机上复验

### 后续

- 真机 SN 写入 `config/cameras.yaml` preferred。
- 外参与手眼标定流程对接 `calibration/`。
- YOLO / 多视角融合按总方案下一阶段推进。

---

## 2026-11-12 — 根目录启动脚本

- 新增 `start_web.sh` / `start_desktop.sh`（项目根目录）。
- 公共环境：`scripts/env.sh`（激活 `.venv`、`PYTHONPATH=src`、弱化 conda 干扰、Web host/port）。
- `scripts/run_*.sh` 改为转发到根目录脚本，保持兼容。

---

## 2026-11-12 — M1/M2 联动与交互修正

1. **M1→M2 序列号联动**：选定/出流时 `bind_selection` 写入服务状态与 `extrinsics.yaml` 的 serial；M2 序列号只读。
2. **预览比例**：桌面 `AspectRatioPreview` letterbox；Web `aspect-ratio: 4/3` + `object-fit: contain`。
3. **外参输入**：改为 4×4 数值格（桌面 `QDoubleSpinBox`，Web `InputNumber`），去掉纯文本矩阵。

API 新增：`GET/POST /api/selection`。

---

## Commit 对照

| Commit | 说明 |
|--------|------|
| `40701dd` | chore: bootstrap rs-detect with M1/M2 dual-camera toolkit |
| `700743e` | docs: record initial commit hash and commit tips in DEVLOG |
| `a3de169` | chore: add root start_web/start_desktop scripts with shared env |

> 注：本环境 `git commit -m` 会被注入不兼容的 `--trailer`；提交时使用 `git commit -F <msgfile>`。
