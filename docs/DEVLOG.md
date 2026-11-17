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

## 2026-11-13 — 路线纠正：先 Cell C 臂控 + Three.js，再感知闭环

用户明确主路径（对齐 sensors-arm Cell C / sensors-dcs huiling+CS66）：

1. **机械臂控制 + Three.js 3D 画布**（准确显示 FK / 可下发）
2. D435 粗定位 → **下发运动到粗观测点**
3. D405 精定位 ↔ 臂运动 **闭环**，直到抓取点收敛

### 本轮已落地（步骤 1）

- 配置：`config/robot_cell_c.yaml`（home / tcp=0.117 / cs66_kin_g26562 / URDF cell_c）
- 适配器：`motion/arm_adapter.py` dry-run Connect→Arm→Home/关节下发 + FK/IK（`sensors_arm.arm_pose`）
- 资源：`web/assets` → symlink sensors-arm `vendor` + `models/cs66`
- 页面：主 SPA Tab「机械臂 (Cell C)」+ 前端路由 `/#/arm`（Three.js URDF）
- 修复：Tab 轮询导致 useEffect 反复销毁 Three/URDF；改为仅 `active` 依赖 + App 级 URDF/STL 预加载
- 旧 `/arm.html` 仅 redirect → `/#/arm`
- API：`/api/arm/{state,connect,arm,disarm,home,command,move_pose,ik}`
- Web 启动优先 `.venv312`（Python 3.12；hik-sensors 需 ≥3.10）
- 闭环骨架占位：`pipeline/servo_loop.py`（待臂控验收后再接）

### M2 Steps UI（进行中）

- 检测 Tab 改为 Ant Design `Steps` 流程
- **Step1**：D435 拍照粗定位 → `p_coarse` → `p_safe = p_coarse+[0,0,Δz]`
- **Step2**：按钮「运动到安全点 p_safe」→ `CellCArmAdapter.move_xyz_keep_rpy`
- **Step3**：按钮「开始精定位闭环」→ D405 测 `c*` → `p_pregrasp=c*+[0,0,Δz]` → 运动 → 再测至 XY 收敛
- **Step4**：按钮「抓取（占位）」→ 吸盘 ON（占位 IO）
- **Step5**：安全点列表编辑 / 保存 /「下发依次运动」（`config/safe_points.yaml`）
- **Step6**：按钮「放开吸盘」→ 吸盘 OFF（占位）
- API：`/api/pipeline/{…,grasp,safe_points,safe_points/run,suction/release,workflow}`

### M3 多视角融合（已落地骨架）

- `motion/viewpoint.py`：N 视点环
- `pipeline/multiview_capture.py` + `multiview_fuse.py`：到位→短停→抓帧→体素/平面融合
- `geometry/voxel_fuse.py` / `plane_fuse.py`：点云融合 + 平面一致性 + 交叉验证
- Step3 主按钮「多视角精定位（M3）」；单帧闭环保留为备用
- API：`POST /api/pipeline/multiview`；配置见 `config/pipeline.yaml` `multiview_*`
- 评测模板：`docs/metrics_m3.md`

### 下一步

- M3 真机重复性填表；弱交叉验证时的补采策略加固
- M4：门控强制进抓取、真 FK、下降、吸盘 IO
- 接 live `arm_cs_write`（192.168.1.200）

---

## 2026-11-13 — 算法 M2：单帧检测通路落地

### 本轮完成

1. **检测 / 几何模块**
   - `detect/detector.py`：YOLO（可选）/ HSV / Mock ROI 统一入口
   - `geometry/mask_to_cloud.py`：Mask + Align 深度反投影
   - `geometry/plane_ransac.py`：RANSAC 平面 + 吸点（相机系，法向朝向相机）
   - `detect/pipeline.py`：单帧管线 + 可视化 JPEG
2. **按方案主路径（D435 粗 → D405 精 → 门控）**
   - `pipeline/coarse_localize.py`：`p_coarse` / `n_coarse`（base）+ 腕部观测 hint
   - `pipeline/fine_localize.py`：D405 单帧 `c*`（M3 再融多视）
   - `pipeline/gate.py`：`||c*-p_coarse||` / 法向角门控（不平均深度）
   - `pipeline/m2_pipeline.py` + `DetectService.run_m2_pipeline`
   - API：`/api/pipeline/{m2,coarse,fine,last}`；UI 主按钮「粗→精→门控」
3. **配置 / CLI**：`config/pipeline.yaml`；`scripts/run_m2_pipeline.py`；单帧 CLI 保留作调试
4. **里程碑文案**：M0 出流 · M1 外参 · M2 检测；Mock 平面纸板便于 smoke

### 验收（Mock）

- [x] `run_m2_pipeline`：先出 `p_coarse`，再 D405 `c_star`，门控通过
- [x] 粗失败可按配置跳过精定位
- [ ] 真机 + 自训 YOLO + 真实 `T_base_flange` 复验（待现场）

### 后续

- M2.1/M2.2：纸板样张与 `models/cardboard_yolo.pt`
- M3：多视角融合（替换单帧 fine）
- M4：接入臂 FK 提供真实 `T_base_flange`

---

## Commit 对照

| Commit | 说明 |
|--------|------|
| `40701dd` | chore: bootstrap rs-detect with M1/M2 dual-camera toolkit |
| `700743e` | docs: record initial commit hash and commit tips in DEVLOG |
| `a3de169` | chore: add root start_web/start_desktop scripts with shared env |
| `3a20d21` | fix(ui): link M1 selection to M2 serials, fix preview ratio, matrix grid extrinsics |

> 注：本环境 `git commit -m` 会被注入不兼容的 `--trailer`；提交时使用 `git commit -F <msgfile>`。
