# grasp-service 服务端约定（rs-detect Client Agent 侧理解）

> **用途**：Grasp Agent 对 `grasp-service` 的接口假设与字段契约，便于与真实服务端对照。  
> **客户端**：`rs-detect`（`src/rs_detect/grasp/`）  
> **修订**：已按服务端反馈纠正（hole/detect、`tcp_target` null、Final 前置、S2 mask、422、**site 进程单例**、**手眼仅 hole/S0**、**S2 纯 tool 系**、**force 需 yaml allow** 等）  
> **默认文档端口**：`http://127.0.0.1:6010`（以 `config/grasp_client.yaml` 的 `base_url` 为准）

---

## 0. 一句话边界

| 角色 | 负责 | 不负责 |
|------|------|--------|
| **grasp-service** | SAM/孔/GraspNet 规划；运动/夹爪**计划**与周期指令；**session**；site 热配置 | 臂驱、夹爪硬件、采图、定时循环；**S2 的 tool→base**；ICP 站级路径 |
| **rs-detect Agent** | 采图/Mock、FrameUpload、`session_id`、执行运动；**S2 全程 tool→控制器/base 积分** | 替换服务端算法；内部调 SAM3 |

**标定 / 相机系→base（易误读，以服务端澄清为准）**：

- 服务端**只用**手眼/TCP 在 **`hole/detect`、`S0/plan`** 里产出 **base 位姿**（如 `tcp_target`、`pregrasp_flange`）。
- **S2 不算、也不下发 base pose/速度**；只有 tool 系 `cartesian_velocity`。S2 全程由**客户端**做 tool→base（或控制器）积分。
- 不要把「服务端持有手眼」读成「S2 也帮你变 base」。

**结论**：规划与分阶段 HTTP 状态机 S0→S1→S2→Final 已可对接；真机闭环、S2 积分、ICP 站级路径在客户端侧。

---

## 1. HTTP 阶段表

| 阶段 | 路径 | 服务端职责 |
|------|------|------------|
| **S0** | `POST /v1/s0/plan` | SAM mask QC + GraspNet → flange、**创建 session**、gripper |
| **S1** | `POST /v1/s1/latch`、`/v1/s1/step` | 锁 mask；**monitor_only / 速度 0**；handoff / exit |
| **S2** | `POST /v1/s2/latch`、`/v1/s2/step` | 孔 + tool 系 `cartesian_velocity`；`exit_s2_servo` |
| **Final** | `POST /v1/final/approach` | `linear_approach_segments` + gripper 闭合意图（**无** `/v1/s3`） |

辅助：`GET /health`、`GET/POST /v1/site/config|configure`、`POST /v1/hole/detect`、`POST /v1/materials/detect`（hole/materials **不**创建 session）。

OpenAPI：`{base}/docs`。

---

## 1.1 Site 配置（流水线第 1 步）

| 方法 | 路径 | 作用 |
|------|------|------|
| `GET` | `/v1/site/config` | 字段说明表 + 当前生效配置摘要 |
| `POST` | `/v1/site/configure` | 客户端 payload → **进程级**热更新 + 写入 `server_grasp/runtime/` |

**进程级 SITE 单例（非 per-session）**：

- `POST /v1/site/configure` 更新的是 **进程内 SITE 单例**，**不是**每个 `session_id` 一份。
- 同 worker 内后续请求立即用新值；落盘后重启同配置仍可恢复。
- **多 worker / 多实例**：每个进程要各自 `configure`（或部署 `workers=1`），否则部分 worker 仍是旧 site。

**客户端应推送的字段：**

| Payload 键 | 内容 | 热更新后影响 |
|------------|------|----------------|
| `hand_eye` | `calibrated` + `transform_base_from_scene_camera_m`（4×4） | **仅** hole/detect、S0/plan 的 base 位姿 |
| `wrist_hand_eye` | `transform_base_from_wrist_camera_m`（4×4） | 存盘预留；**当前 HTTP S2 完全不用腕→base**（S2 只用上传的腕 `color_intrinsics` + 图） |
| `tcp` | 整份 tcp yaml（`transform_flange_*` + gripper） | hole TCP、S0 法兰链、Final 夹爪宽 |
| `sam.material_text` | 物料 SAM 文案 | S0/S1 mask |
| `sam.hole_sam_text` | 孔 SAM 文案 | hole/detect、S2 |
| `motion.retreat_m` | 米 | S0 pre-grasp 后退 |
| `final_approach` | approach/gripper/policy 子键 merge | Final 分段；policy 含 **`allow_force_without_s2_exit`** |
| `meta` | 任意 JSON | 仅记录（robot_id、标定时间） |

**本仓库映射（rs-detect → payload）：**

| 本地源 | Payload |
|--------|---------|
| `config/extrinsics.yaml` · d435 `T`（`T_base_cam`） | `hand_eye` |
| d405 `T`（`T_flange_cam`）± 当前 FK 合成 | `wrist_hand_eye`（另带 `transform_flange_from_wrist_camera_m`） |
| `config/robot_cell_c.yaml` · `tcp_xyz` / `gripper_rpy` | `tcp` |
| `config/grasp_client.yaml` · sam / retreat / final | `sam` / `motion` / `final_approach` |

本地代理：`GET/POST /api/grasp/site/config|configure`，预览 `GET /api/grasp/site/payload`。

---

## 2. FrameUpload

| 字段 | 约定 |
|------|------|
| `color_b64` | PNG/JPEG base64；服务端 **BGR** 解码 |
| `depth_b64` | **uint16 毫米 PNG** base64 |
| `color_intrinsics` | `width,height,fx,fy,ppx,ppy` |
| `depth_scale` | 客户端继续发 **`1000.0`**；当前服务端以 uint16 mm PNG 为主，**未必**在所有路径使用该浮点字段 |

**相机**：S0/S1/hole/materials = **场景**；S2 = **腕**。服务端不校验「是不是真腕」，契约上由客户端保证。

**多余字段** `serial` / `timestamp_ms`：Pydantic v2 默认 **忽略**（除非日后 `extra=forbid`）。

**Mock**：`data/mock/d435_*`（场景）、`data/mock/d405_*`（腕）。

---

## 3. Session

- **仅** `POST /v1/s0/plan` 创建 `session_id`
- S1/S2/Final 必须带同一 id
- TTL ~3600s；过期 **404**；**无** DELETE session
- `hole/detect` **不**创建 session；要走 S1/S2/Final 必须先 S0

---

## 4. 分接口要点（含服务端纠正）

### 4.0 Site（优先于 health）

流水线**第一步**是 `POST /v1/site/configure`（同步真机数据），不是 `GET /health`。  
`GET /health` 仅作可选连通性探测。

### 4.0b `GET /health`

- 含 `ok`、`stages`（可用于能力探测）
- **还有**其它字段：`service`、`version`、`sam_base_url`、`port_hint`、`s2_config` 等 — **勿假定只有 ok+stages**

---

### 4.1 `POST /v1/hole/detect` ⚠️ 纠正

**HoleDetectRequest 有**：

| 字段 | 含义 |
|------|------|
| FrameUpload | 场景图 |
| `sam_text` | 取 **整件物料** mask（物料 SAM），**不是**孔 SAM 文案 |
| `offset_along_normal_mm` / `normal_direction` | 沿法线偏置 |

**没有** `hole_sam_text`。传入会被 Pydantic **忽略**。  
孔侧文本在服务端 **`sam_stl_map.yaml` → `hole_sam_text`**（默认如 `large rectangular hole opening`）。  
客户端要改孔提示词：改服务端 yaml，或走 **S2** 的 `hole_sam_text`。

**本仓库**：hole/detect 只传 `sam_text`（物料，来自 `grasp_client.yaml`）；`hole_sam_text` 仅用于 **S2 latch**。

#### `tcp_target` 未标定时 ⚠️ 纠正

`calibrated: false` 时**不是假位姿**，而是空位姿结构，例如：

```json
"tcp_target": {
  "command_type": "move_tcp",
  "frame": "robot_base",
  "position_m": null,
  "quaternion_xyzw": null
}
```

**客户端规则**：不能只看 `calibrated`；必须 **`position_m` 非 null** 才可 `move_tcp`。  
标定后：`position_m` / `quaternion_xyzw` 可用。

成功时另有：`center_camera_m`、`plane_normal_camera`、`pregrasp_camera_m` 等。

---

### 4.2 `POST /v1/s0/plan`

- 创建 `session_id`
- `motion.pregrasp_flange`：`position_m` + `quaternion_xyzw`（robot_base）
- `gripper.width_m`：开口  
- `gripper.action`：如 **`set_width_at_pregrasp`**（动作语义看此字段）
- `calibrated`、`mask_qc`、`client_actions`
- **`servo_anchor_scene_m`**：仍来自 **GraspNet**，与 hole **无关**；S1 latch 会用该锚点填 telemetry `targets` 里的像素 u/v（S1 已不减伺服，但 JSON 里仍有）。**跳过 S1 无影响**；若开 S1，心里有数即可。

**缺口（仍成立）**：S0 GraspNet flange 与 hole `tcp_target` **未自动合并**；孔任务优先 hole（且 `position_m` 可用时）。

hole 与 s0 可用不同场景帧，但 session 与孔位可能语义分裂；联调建议同一次采集。

---

### 4.2b 与 rs-detect ICP / M2–M3 预抓取的关系

- grasp-service **6010 路径不含 ICP**；文档不写 ICP 进服务端是正确的。
- rs-detect **Detect / pipeline（M2–M3）** 若用 ICP / 多视角融合算 `c*` / `p_pregrasp`，那是 **客户端站级路径**，与 Grasp Tab 的 `hole/detect` **不是同一条链**。
- 组合时须在本仓库自己的流程里写清（二选一或先 ICP 再 hole）：**`/api/grasp/run/hole` 只用 hole→S0→move，不调用 Detect ICP**，避免与后端 hole 算两套 pre-grasp。

---

### 4.3 S1

#### latch

```json
"motion": { "command_type": "none", "note": "latched" }
```

telemetry 可含**嵌套** `targets`（像素目标可来自 S0 的 `servo_anchor_scene_m`）等。

#### step

| 字段 | 约定 |
|------|------|
| `command_type` | `monitor_only` |
| `velocity` | **全 0** — 客户端禁止当下发伺服 |
| `hint_handoff_wrist_cam` | mask 相对 latch 下跌 |
| `exit_s1_servo` | 可结束 S1 |
| `proceed_to_s2` | 本帧 handoff 或已 exit 时可能为 **true**（与 hint **不完全相同**） |

自动切 S2 时建议同时看 **`exit_s1_servo` 与 `proceed_to_s2`**。

**telemetry 形状**：S1 **step** 多为**扁平**（如 `mask_area_ratio_vs_latch`）；**勿**与 latch/S2 的嵌套结构混用假设。

---

### 4.4 S2

**坐标系**：服务端 S2 **完全不用** `wrist_hand_eye` / 腕→base；只用当帧上传的 **腕 `color_intrinsics` + 图（+ 可选 mask）**。  
下发速度恒为 **tool** 系；**tool→base / 控制器积分在客户端**（本仓库 `s2_step_once` → `_apply_tool_velocity`）。

#### latch

- Body：`session_id` + **腕** FrameUpload
- 可选：`mask_b64`、`object_mask_b64`、**`hole_sam_text`**（可覆盖孔文本）
- 成功后 session 上有 **`s2_targets`** — **Final 前置硬条件之一**
- telemetry：含嵌套 `targets`、`hole` 等

#### step

| 字段 | 约定 |
|------|------|
| `command_type` | `cartesian_velocity` |
| `frame` | `tool`（**不是** robot_base） |
| `velocity` | `vx/vy/vz_m_s`、`wz_rad_s` |
| `exit_s2_servo` | true → 停环，进 Final |

⚠️ **不传 `mask_b64` 时**：服务端**每帧重跑孔检测 + SAM**，延迟大，很难稳定 10–30 Hz。  
生产 loop **建议每帧带孔 `mask_b64`（跟踪）**；否则只是「能跑」不是「能控」。  
（本仓库当前 loop 尚未上传跟踪 mask，已在日志/UI 警示。）

---

### 4.5 `POST /v1/final/approach` ⚠️ 前置写全

请求无图，可带 `insert_total_m` / `step_m` / `linear_speed_m_s` / `direction_sign` / **`force_without_s2_exit`**（请求字段）。

**必须同时满足（默认）**：

1. 已成功 **`POST /v1/s2/latch`**（session 有 `s2_targets`），否则 **400：请先 POST /v1/s2/latch**
2. 通常还需 **`exit_s2_servo` 已完成**
3. 跳过 exit 时：**请求** `force_without_s2_exit=true` **还不够** — 服务端 `final_approach` 配置（如 `final_approach.example.yaml`）里须  
   `policy.allow_force_without_s2_exit: true`（**默认 false**）。  
   可通过 `POST /v1/site/configure` 的 `final_approach.policy` merge 打开；本仓库 Site 表单推送该键。

`motion`：`linear_approach_segments` + `segment_m` / `segment_count` / …  
`gripper`：`action_after_approach` / `width_m_command` / `when`

---

## 5. 错误码

| 码 | 含义 |
|----|------|
| 400 | 业务前置失败、尺寸不符等；`detail` 多为字符串 |
| 404 | session 无效/过期 |
| **422** | FastAPI/Pydantic **校验失败**；`detail` **常为数组** `{loc,msg,...}` |
| 500 | SAM/GraspNet/孔检测等 |

客户端须解析 **422 数组 detail**（本仓库 `api.js` + `GraspServiceError` 已处理）。

---

## 6. 推荐调用序

### 6.1 孔抓取精简（默认无 S1）

```text
1. GET  /v1/site/config     # 可选：看字段表 / 当前摘要
2. POST /v1/site/configure  # 同步手眼·TCP·SAM·retreat·final（热更新）
3. POST /v1/hole/detect     # 不建 session；sam_text=物料
4. POST /v1/s0/plan         # → session_id, gripper
5. [Client] 仅当 tcp_target.position_m ≠ null 时 move_tcp；set_width(gripper)
6. 腕就位
7. POST /v1/s2/latch        # 必须；可选 hole_sam_text
8. loop POST /v1/s2/step    # 生产建议带 mask_b64
9. POST /v1/final/approach  # 需 latch + 通常需 exit_s2
10. [Client] segments + close intent
```

### 6.2 带 S1

接近过程中 `s1/latch` + `s1/step`；看 `exit_s1_servo` / `proceed_to_s2` 后进 S2。

---

## 7. 环境变量（服务端，客户端不读）

文档常见：`GRASP_PORT`、`SAM_BASE_URL`、`GRASP_HAND_EYE`、`GRASP_TCP`、`GRASP_SAM_TEXT`、`GRASP_S*_CFG`。  
另有如 `GRASP_RETREAT_M`、`GRASP_CHECKPOINT` 等，对照清单可不写全。

---

## 8. 已知缺口 / 联调注意

1. **S0 与孔位未统一**（GraspNet flange vs hole tcp）— 孔任务用 hole，且检查 `position_m`。  
2. 未标定：`tcp_target` 为 **null 位姿结构**，不是随机假数。  
3. 无硬件运动 API；无 DELETE session。  
4. S2 无图 dry-run：客户端用 Mock 存量图。  
5. S2 无 `mask_b64` → 性能差。  
6. Final 必须先 **s2/latch**；force 还需 yaml **`allow_force_without_s2_exit`**。  
7. `client_actions` 非硬协议。  
8. Site 为**进程单例**；多 worker 须各配一次。  
9. Detect ICP / M3 `p_pregrasp` 与 Grasp `hole/detect` 是两条客户端路径，勿混为同一 pre-grasp。

---

## 9. 本仓库映射

| 本地 | 远端 |
|------|------|
| `/api/grasp/site/configure` | `POST /v1/site/configure`（流水线第 1 步） |
| `/api/grasp/site/config` | `GET /v1/site/config` |
| `/api/grasp/site/payload` | **本地**预览组包 |
| `/api/grasp/health` | `GET /health`（可选） |
| `/api/grasp/hole/detect` | `POST /v1/hole/detect`（`sam_text`，无 `hole_sam_text`） |
| `/api/grasp/s0/plan` | `POST /v1/s0/plan` |
| `/api/grasp/s1/*` | `/v1/s1/*` |
| `/api/grasp/s2/*` | `/v1/s2/*`（latch 传 `hole_sam_text`） |
| `/api/grasp/final/approach` | `POST /v1/final/approach` |
| `move_pregrasp` / `final/execute` / `mock` | **本地** |

---

## 10. 验收清单（服务端 anyGrasp 当前答案）

| 问题 | 答案 |
|------|------|
| health `ok` + `stages` | **是**（另有额外字段） |
| FrameUpload BGR + uint16 mm + intrinsics | **是** |
| 场景/腕按接口区分 | **是**（契约；不校验真假腕） |
| 仅 S0 创建 session | **是** |
| S1 `monitor_only` + 速度 0 | **是** |
| S2 tool + `cartesian_velocity` + `exit_s2_servo` | **是**（无 base；不用腕外参） |
| Final 路径 + 默认要 S2 exit | **是（且要先 s2/latch）** |
| force 还需 yaml `allow_force_without_s2_exit` | **是（默认 false）** |
| hole `tcp_target` 标定后可用 | **是（未标定为 null）** |
| site 进程单例 / 非 per-session | **是** |
| 未知 JSON 字段 | **忽略** |
| 错误 `detail` | **是；校验失败多为 422** |

---

## 11. 与服务端一致、可保持不动的部分

- S0–Final 阶段表  
- S1 禁止发非零速度  
- S2 tool 系由客户端积分到控制器  
- session TTL / 无 DELETE  
- S0 与 hole 未合并；`servo_anchor_scene_m` 与 hole 无关  
- `client_actions` 非硬协议  
- Mock 路径约定  
- 精简流程 6.1 / 带 S1 的 6.2（跳过 S1 无影响）  

---

*文档与 rs-detect Grasp Agent 实现同步；供与 grasp-service 对照。*
