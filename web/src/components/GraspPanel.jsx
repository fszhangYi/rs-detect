import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert, Badge, Button, Card, Col, Collapse, Descriptions, Divider, Input, InputNumber,
  Progress, Row, Select, Slider, Space, Steps, Switch, Tag, Tooltip, Typography, message,
} from 'antd';
import { API, api } from '../lib/api.js';
import { armGhostTarget, armSceneMarkers, ARM_MARKER_COLORS } from '../lib/armMarkers.js';

const { Text, Title, Paragraph } = Typography;

/** 4×4 → {xyz_m, rpy_deg} using scipy-equivalent extrinsic xyz (R = Rz@Ry@Rx). */
function mat4ToPose(T) {
  if (!Array.isArray(T) || T.length !== 4) {
    return { xyz_m: [0, 0, 0], rpy_deg: [0, 0, 0] };
  }
  const m00 = Number(T[0][0]); const m01 = Number(T[0][1]); const m02 = Number(T[0][2]);
  const m10 = Number(T[1][0]); const m11 = Number(T[1][1]); const m12 = Number(T[1][2]);
  const m20 = Number(T[2][0]); const m21 = Number(T[2][1]); const m22 = Number(T[2][2]);
  const xyz_m = [Number(T[0][3]) || 0, Number(T[1][3]) || 0, Number(T[2][3]) || 0];
  let roll; let pitch; let yaw;
  const sy = -m20;
  if (Math.abs(sy) < 0.999999) {
    pitch = Math.asin(Math.max(-1, Math.min(1, sy)));
    roll = Math.atan2(m21, m22);
    yaw = Math.atan2(m10, m00);
  } else {
    pitch = sy > 0 ? Math.PI / 2 : -Math.PI / 2;
    roll = Math.atan2(-m12, m11);
    yaw = 0;
  }
  const rpy_deg = [roll, pitch, yaw].map((v) => (Number.isFinite(v) ? (v * 180) / Math.PI : 0));
  return { xyz_m, rpy_deg };
}

function degToRad3(rpyDeg) {
  return (rpyDeg || [0, 0, 0]).map((v) => (Number(v) * Math.PI) / 180);
}

function radToDeg3(rpyRad) {
  return (rpyRad || [0, 0, 0]).map((v) => (Number(v) * 180) / Math.PI);
}

function poseFromMatOrFields(mat, xyz, rpyRad, rpyDeg) {
  if (Array.isArray(xyz) && (Array.isArray(rpyDeg) || Array.isArray(rpyRad))) {
    return {
      xyz_m: xyz.map(Number),
      rpy_deg: Array.isArray(rpyDeg) ? rpyDeg.map(Number) : radToDeg3(rpyRad),
    };
  }
  return mat4ToPose(mat);
}

function payloadToSiteForm(payload) {
  const p = payload || {};
  const he = p.hand_eye || {};
  const wh = p.wrist_hand_eye || {};
  const tcp = p.tcp || {};
  const grip = tcp.gripper || {};
  const sam = p.sam || {};
  const motion = p.motion || {};
  const fa = p.final_approach || {};
  const ap = fa.approach || {};
  const pol = fa.policy || {};
  const meta = p.meta || {};
  const scenePose = poseFromMatOrFields(
    he.transform_base_from_scene_camera_m,
    he.xyz_m,
    he.rpy_rad,
    he.rpy_deg,
  );
  const flangePose = poseFromMatOrFields(
    wh.transform_flange_from_wrist_camera_m,
    wh.flange_xyz_m || wh.xyz_m,
    wh.flange_rpy_rad || wh.rpy_rad,
    wh.flange_rpy_deg || wh.rpy_deg,
  );
  const hasBase = wh.transform_base_from_wrist_camera_m != null
    || wh.base_xyz_m != null
    || wh.base_rpy_rad != null;
  const basePose = hasBase
    ? poseFromMatOrFields(
      wh.transform_base_from_wrist_camera_m,
      wh.base_xyz_m,
      wh.base_rpy_rad,
      wh.base_rpy_deg,
    )
    : { xyz_m: [0, 0, 0], rpy_deg: [0, 0, 0] };
  const gripRad = Array.isArray(tcp.gripper_rpy_rad) ? tcp.gripper_rpy_rad.map(Number) : [0, 0, Math.PI];
  return {
    calibrated: !!he.calibrated,
    scene_xyz: scenePose.xyz_m,
    scene_rpy_deg: scenePose.rpy_deg,
    wrist_flange_xyz: flangePose.xyz_m,
    wrist_flange_rpy_deg: flangePose.rpy_deg,
    wrist_base_enabled: !!hasBase,
    wrist_base_xyz: basePose.xyz_m,
    wrist_base_rpy_deg: basePose.rpy_deg,
    wrist_source: wh.source || '',
    tcp_xyz: Array.isArray(tcp.tcp_xyz_m) ? tcp.tcp_xyz_m.map(Number) : [0, 0, 0.117],
    gripper_rpy_deg: radToDeg3(gripRad),
    width_open: Number(grip.width_m_open ?? 0.08),
    width_closed: Number(grip.width_m_closed ?? 0),
    material_text: String(sam.material_text || ''),
    hole_sam_text: String(sam.hole_sam_text || ''),
    retreat_m: Number(motion.retreat_m ?? 0.05),
    insert_total_m: Number(ap.insert_total_m ?? 0.012),
    step_m: Number(ap.step_m ?? 0.002),
    linear_speed_m_s: Number(ap.linear_speed_m_s ?? 0.004),
    direction_sign: Number(ap.direction_sign ?? 1),
    pause_ms: Number(ap.pause_ms_between_steps ?? 80),
    force_without_s2_exit: !!(pol.allow_force_without_s2_exit ?? pol.force_without_s2_exit),
    robot_id: String(meta.robot_id || 'arm-c'),
    robot_ip: String(meta.robot_ip || ''),
    source_note: '',
  };
}

/** Pull fields from remote GET /v1/site/config (shape varies). */
function remoteConfigToSiteForm(remote, prev) {
  const root = remote?.config || remote?.site || remote?.effective || remote || {};
  const he = root.hand_eye || root.handEye || {};
  const wh = root.wrist_hand_eye || root.wristHandEye || {};
  const tcp = root.tcp || {};
  const sam = root.sam || {};
  const motion = root.motion || {};
  const fa = root.final_approach || root.finalApproach || {};
  const base = prev || payloadToSiteForm({});
  const remoteForm = payloadToSiteForm({
    hand_eye: he,
    wrist_hand_eye: wh,
    tcp,
    sam,
    motion,
    final_approach: fa,
    meta: root.meta || {},
  });
  const next = { ...base, source_note: 'remote /v1/site/config' };
  if (he.calibrated != null || he.transform_base_from_scene_camera_m || he.xyz_m) {
    next.calibrated = remoteForm.calibrated;
    next.scene_xyz = remoteForm.scene_xyz;
    next.scene_rpy_deg = remoteForm.scene_rpy_deg;
  }
  if (wh.transform_flange_from_wrist_camera_m || wh.flange_xyz_m || wh.xyz_m) {
    next.wrist_flange_xyz = remoteForm.wrist_flange_xyz;
    next.wrist_flange_rpy_deg = remoteForm.wrist_flange_rpy_deg;
    next.wrist_source = remoteForm.wrist_source;
  }
  if (wh.transform_base_from_wrist_camera_m != null || wh.base_xyz_m != null) {
    next.wrist_base_enabled = remoteForm.wrist_base_enabled;
    next.wrist_base_xyz = remoteForm.wrist_base_xyz;
    next.wrist_base_rpy_deg = remoteForm.wrist_base_rpy_deg;
  }
  if (tcp.tcp_xyz_m || tcp.gripper_rpy_rad || tcp.gripper) {
    next.tcp_xyz = remoteForm.tcp_xyz;
    next.gripper_rpy_deg = remoteForm.gripper_rpy_deg;
    next.width_open = remoteForm.width_open;
    next.width_closed = remoteForm.width_closed;
  }
  if (sam.material_text) next.material_text = remoteForm.material_text;
  if (sam.hole_sam_text) next.hole_sam_text = remoteForm.hole_sam_text;
  if (motion.retreat_m != null) next.retreat_m = remoteForm.retreat_m;
  if (fa.approach || fa.policy) {
    next.insert_total_m = remoteForm.insert_total_m;
    next.step_m = remoteForm.step_m;
    next.linear_speed_m_s = remoteForm.linear_speed_m_s;
    next.force_without_s2_exit = remoteForm.force_without_s2_exit;
  }
  if (root.meta?.robot_id) next.robot_id = remoteForm.robot_id;
  if (root.meta?.robot_ip) next.robot_ip = remoteForm.robot_ip;
  return next;
}

function siteFormToPayload(form) {
  const tcp_xyz = (form.tcp_xyz || [0, 0, 0.117]).map(Number);
  const gripper_rpy = degToRad3(form.gripper_rpy_deg || [0, 0, 180]);
  const scene_xyz = (form.scene_xyz || [0, 0, 0]).map(Number);
  const scene_rpy = degToRad3(form.scene_rpy_deg || [0, 0, 0]);
  const flange_xyz = (form.wrist_flange_xyz || [0, 0, 0]).map(Number);
  const flange_rpy = degToRad3(form.wrist_flange_rpy_deg || [0, 0, 0]);
  const wrist = {
    flange_xyz_m: flange_xyz,
    flange_rpy_rad: flange_rpy,
    source: form.wrist_source || 'client_form',
  };
  if (form.wrist_base_enabled) {
    wrist.base_xyz_m = (form.wrist_base_xyz || [0, 0, 0]).map(Number);
    wrist.base_rpy_rad = degToRad3(form.wrist_base_rpy_deg || [0, 0, 0]);
  } else {
    wrist.transform_base_from_wrist_camera_m = null;
  }
  return {
    hand_eye: {
      calibrated: !!form.calibrated,
      xyz_m: scene_xyz,
      rpy_rad: scene_rpy,
    },
    wrist_hand_eye: wrist,
    tcp: {
      tcp_xyz_m: tcp_xyz,
      gripper_rpy_rad: gripper_rpy,
      gripper: {
        action_default: 'set_width_at_pregrasp',
        width_m_open: Number(form.width_open),
        width_m_closed: Number(form.width_closed),
      },
    },
    sam: {
      material_text: String(form.material_text || ''),
      hole_sam_text: String(form.hole_sam_text || ''),
    },
    motion: { retreat_m: Number(form.retreat_m) },
    final_approach: {
      approach: {
        insert_total_m: Number(form.insert_total_m),
        step_m: Number(form.step_m),
        linear_speed_m_s: Number(form.linear_speed_m_s),
        direction_sign: Number(form.direction_sign),
        pause_ms_between_steps: Number(form.pause_ms),
      },
      gripper: {
        action_after_approach: 'close',
        width_m_command: Number(form.width_closed),
      },
      policy: {
        allow_force_without_s2_exit: !!form.force_without_s2_exit,
      },
    },
    meta: {
      client: 'rs-detect',
      robot_id: String(form.robot_id || ''),
      robot_ip: form.robot_ip || null,
    },
  };
}

function PoseEditor({ title, hint, xyz, rpyDeg, onXyz, onRpy, disabled }) {
  const setXyz = (i, v) => {
    const next = [...(xyz || [0, 0, 0])];
    next[i] = typeof v === 'number' ? v : next[i];
    onXyz(next);
  };
  const setRpy = (i, v) => {
    const next = [...(rpyDeg || [0, 0, 0])];
    next[i] = typeof v === 'number' ? v : next[i];
    onRpy(next);
  };
  const labelsXyz = ['X m', 'Y m', 'Z m'];
  const labelsRpy = ['Rx °', 'Ry °', 'Rz °'];
  return (
    <div className={`grasp-pose-editor${disabled ? ' is-disabled' : ''}`}>
      <div className="grasp-pose-title">
        <b>{title}</b>
        {hint ? <span>{hint}</span> : null}
      </div>
      <div className="grasp-pose-row">
        {labelsXyz.map((lab, i) => (
          <div className="grasp-field inline" key={`xyz-${lab}`}>
            <label>{lab}</label>
            <InputNumber
              step={0.001}
              disabled={disabled}
              value={xyz?.[i]}
              onChange={(v) => setXyz(i, v)}
            />
          </div>
        ))}
      </div>
      <div className="grasp-pose-row">
        {labelsRpy.map((lab, i) => (
          <div className="grasp-field inline" key={`rpy-${lab}`}>
            <label>{lab}</label>
            <InputNumber
              step={0.1}
              disabled={disabled}
              value={rpyDeg?.[i]}
              onChange={(v) => setRpy(i, v)}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

const PIPELINE = [
  {
    key: 'site',
    title: 'Site sync',
    subtitle: '同步真机',
    hint: '先填写/从本机标定同步手眼·TCP·SAM，再推送到 POST /v1/site/configure。',
    cam: null,
  },
  {
    key: 'hole',
    title: 'Hole',
    subtitle: '孔检测',
    hint: '先检测全部物料实例，点选其一后再做孔检测（裁剪 ROI）。画布：孔心小球 + 法向。',
    cam: 'scene',
  },
  {
    key: 's0',
    title: 'S0 Plan',
    subtitle: '建 session',
    hint: 'S0 = 孔心沿法向反方向偏移 s0_along_normal_m（默认 7cm）。远端仍可建 session/夹爪；运动位姿用本地几何。',
    cam: 'scene',
  },
  {
    key: 'move',
    title: 'Pre-grasp',
    subtitle: '到位',
    hint: '优先 S0=孔心−法向×s0_along_normal_m。默认 DRY-RUN。',
    cam: null,
  },
  {
    key: 's2',
    title: 'S2 Servo',
    subtitle: '腕孔精调',
    hint: '须先 latch。单步或 Loop 直到 exit_s2。服务端只给 tool 系速度；不用腕→base 外参（只用腕内参+图）。',
    cam: 'wrist',
  },
  {
    key: 'final',
    title: 'Final',
    subtitle: '插入计划',
    hint: '需 s2/latch + 通常需 exit_s2。force 还须服务端 policy.allow_force_without_s2_exit（可用 Site 推送）。',
    cam: null,
  },
  {
    key: 'done',
    title: 'Done',
    subtitle: '完成',
    hint: '分段执行结束；可重置 session 再跑一轮。',
    cam: null,
  },
];

const STAGE_STEP = {
  idle: 0,
  site_ok: 0,
  site_bad: 0,
  health_ok: 0,
  health_bad: 0,
  materials: 1,
  hole: 1,
  s0: 2,
  moved: 3,
  s1: 3,
  s2_latch: 4,
  s2_step: 4,
  s2_loop: 4,
  s2_exit: 4,
  s2_stopped: 4,
  final: 5,
  done: 6,
  error: -1,
};

const CHECKLIST = [
  '先 site/configure（进程级 SITE；多 worker 各配一次）',
  '手眼/TCP 只影响 hole 与 S0 的 base 位姿',
  '场景 / 腕 内参分相机上传',
  '深度 uint16 mm PNG',
  '保存并复用 session_id',
  'S1 可选；servo_anchor 来自 GraspNet 非 hole',
  'S2：客户端做 tool→base；服务端不用腕外参',
  'Final：force 还需 allow_force_without_s2_exit',
  'run/hole 不用 Detect ICP，避免两套 pre-grasp',
  'calibrated=false 时勿盲信 base 位姿',
];

function shortJson(v, max = 900) {
  try {
    const s = JSON.stringify(v, null, 2);
    return s.length > max ? `${s.slice(0, max)}\n…` : s;
  } catch (_) {
    return String(v);
  }
}

function fmtNum(v, digits = 4) {
  if (v == null || Number.isNaN(Number(v))) return '—';
  return Number(v).toFixed(digits);
}

function fmtVec3(v, digits = 4) {
  if (!Array.isArray(v) || v.length < 3) return '—';
  return v.slice(0, 3).map((x) => fmtNum(x, digits)).join(', ');
}

function pickPos(obj) {
  if (!obj || typeof obj !== 'object') return null;
  if (Array.isArray(obj.position_m)) return obj.position_m;
  if (Array.isArray(obj.xyz_m)) return obj.xyz_m;
  if (Array.isArray(obj.xyzrpy)) return obj.xyzrpy.slice(0, 3);
  if (obj.x != null) return [obj.x, obj.y, obj.z];
  return null;
}

function Metric({ label, value, hint, accent }) {
  return (
    <div className={`grasp-metric${accent ? ` is-${accent}` : ''}`}>
      <div className="k">{label}</div>
      <div className="v" title={hint || (typeof value === 'string' ? value : undefined)}>{value}</div>
      {hint ? <div className="h">{hint}</div> : null}
    </div>
  );
}

function CamPane({ src, caption, emptyText }) {
  const [ok, setOk] = useState(false);
  if (!src) {
    return (
      <div className="grasp-cam-pane">
        <div className="grasp-cam-empty">{emptyText || '无图'}</div>
        <div className="cap">{caption}</div>
      </div>
    );
  }
  return (
    <div className="grasp-cam-pane">
      <img key={src} src={src} alt={caption} onLoad={() => setOk(true)} onError={() => setOk(false)} />
      {!ok && <div className="grasp-cam-empty overlay">{emptyText || '加载失败'}</div>}
      <div className="cap">{caption}</div>
    </div>
  );
}

function CamThumb({ serial, label, accent, mockMode, mockColorSrc, mockDepthSrc, pathHint }) {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (mockMode || !serial) return undefined;
    const id = setInterval(() => setTick((t) => t + 1), 400);
    return () => clearInterval(id);
  }, [serial, mockMode]);

  const colorSrc = mockMode && mockColorSrc
    ? `${API}${mockColorSrc}${mockColorSrc.includes('?') ? '&' : '?'}t=1`
    : (serial ? `${API}/api/frame.jpg?serial=${encodeURIComponent(serial)}&which=color&t=${tick}` : null);
  const depthSrc = mockMode && mockDepthSrc
    ? `${API}${mockDepthSrc}${mockDepthSrc.includes('?') ? '&' : '?'}t=1`
    : (serial ? `${API}/api/frame.jpg?serial=${encodeURIComponent(serial)}&which=depth&t=${tick}` : null);

  return (
    <div className={`grasp-cam-rgbd accent-${accent || 'slate'}${mockMode ? ' is-mock' : ''}`}>
      <div className="grasp-cam-rgbd-title">
        <span>{mockMode ? `${label} · MOCK RGB-D` : `${label} · RGB-D`}</span>
        <code>{pathHint || (mockMode ? 'mock' : (serial || '—'))}</code>
      </div>
      <div className="grasp-cam-rgbd-row">
        <CamPane
          src={colorSrc}
          caption="彩色 color"
          emptyText={mockMode ? 'Mock 彩色缺失' : (serial ? '无彩色预览' : `未选定${label}`)}
        />
        <CamPane
          src={depthSrc}
          caption="深度 depth"
          emptyText={mockMode ? 'Mock 深度缺失' : (serial ? '无深度预览' : '—')}
        />
      </div>
    </div>
  );
}

function ActionChip({ title, desc, primary, danger, disabled, loading, onClick }) {
  return (
    <button
      type="button"
      className={`grasp-action${primary ? ' is-primary' : ''}${danger ? ' is-danger' : ''}`}
      disabled={disabled || loading}
      onClick={onClick}
    >
      <span className="t">{loading ? '…' : title}</span>
      <span className="d">{desc}</span>
    </button>
  );
}

/** Hole / S0 client-side overlay jpeg from /api/grasp/viz/*.jpg */
function VizPane({ path, stamp, title, empty }) {
  const [ok, setOk] = useState(false);
  const src = path
    ? `${API}${path}${path.includes('?') ? '&' : '?'}t=${encodeURIComponent(stamp || '0')}`
    : null;
  useEffect(() => { setOk(false); }, [src]);
  return (
    <div className="grasp-viz">
      <div className="grasp-viz-title">{title}</div>
      <div className="grasp-viz-frame">
        {src ? (
          <img
            src={src}
            alt={title}
            onLoad={() => setOk(true)}
            onError={() => setOk(false)}
          />
        ) : null}
        {!src || !ok ? (
          <div className="grasp-viz-empty">{empty || '运行检测后显示投影叠加'}</div>
        ) : null}
      </div>
    </div>
  );
}

export default function GraspPanel({ active, selection }) {
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState(false);
  const [baseUrl, setBaseUrl] = useState('http://127.0.0.1:6010');
  const [samText, setSamText] = useState('white plastic part');
  const [holeText, setHoleText] = useState('hole');
  const [includeS1, setIncludeS1] = useState(false);
  const [preferHole, setPreferHole] = useState(true);
  const [s0FlipApproach, setS0FlipApproach] = useState(true);
  const [s0AlongNormalM, setS0AlongNormalM] = useState(0.07);
  const [s0J6Rad, setS0J6Rad] = useState(12.56);
  const [executeMotion, setExecuteMotion] = useState(false);
  const [forceFinal, setForceFinal] = useState(false);
  const [s2Hz, setS2Hz] = useState(10);
  const [insertTotal, setInsertTotal] = useState(0.012);
  const [stepM, setStepM] = useState(0.002);
  const [previewAll, setPreviewAll] = useState(false);
  const [useMock, setUseMock] = useState(false);
  const [focus, setFocus] = useState(0);
  const [siteForm, setSiteForm] = useState(null);
  const [siteFormSrc, setSiteFormSrc] = useState('');
  const seededRef = useRef(false);
  const siteSeededRef = useRef(false);
  const logRef = useRef(null);

  const refresh = useCallback(async () => {
    const st = await api('/api/grasp/status');
    setStatus(st);
    if (!seededRef.current && st?.config) {
      seededRef.current = true;
      const c = st.config;
      if (c.base_url) setBaseUrl(String(c.base_url));
      if (c.sam_text) setSamText(String(c.sam_text));
      if (c.hole_sam_text) setHoleText(String(c.hole_sam_text));
      if (c.include_s1 != null) setIncludeS1(!!c.include_s1);
      if (c.prefer_hole_tcp != null) setPreferHole(!!c.prefer_hole_tcp);
      if (c.s0_flip_approach != null) setS0FlipApproach(!!c.s0_flip_approach);
      if (c.s0_along_normal_m != null && Number.isFinite(Number(c.s0_along_normal_m))) {
        setS0AlongNormalM(Number(c.s0_along_normal_m));
      }
      if (c.s0_j6_rad != null && Number.isFinite(Number(c.s0_j6_rad))) {
        setS0J6Rad(Number(c.s0_j6_rad));
      }
      if (c.execute_motion != null) setExecuteMotion(!!c.execute_motion);
      if (c.force_without_s2_exit != null) setForceFinal(!!c.force_without_s2_exit);
      if (c.s2_hz != null) setS2Hz(Number(c.s2_hz));
      if (c.insert_total_m != null) setInsertTotal(Number(c.insert_total_m));
      if (c.step_m != null) setStepM(Number(c.step_m));
      if (c.use_mock_frames != null) setUseMock(!!c.use_mock_frames);
    }
    if (st?.mock?.enabled != null) setUseMock(!!st.mock.enabled);
    return st;
  }, []);

  useEffect(() => {
    if (!active) return undefined;
    let cancelled = false;
    const tick = () => refresh().catch(() => {});
    tick();
    const t = setInterval(() => { if (!cancelled) tick(); }, 1200);
    return () => { cancelled = true; clearInterval(t); };
  }, [active, refresh]);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [status?.log_tail]);

  // Auto-follow pipeline only on forward progress / reset — never yank user back
  // when they click an earlier step in the guide.
  const autoStepRef = useRef(-1);
  useEffect(() => {
    if (previewAll) return;
    const stageNow = status?.stage || 'idle';
    const idx = STAGE_STEP[stageNow];
    if (idx == null) return;
    const progress = idx < 0 ? 0 : idx;
    const prev = autoStepRef.current;
    // Only hard-reset focus on idle (显式「重置」)；site_ok 不应把用户从手点的步骤拽回。
    const hardReset = stageNow === 'idle' && progress === 0 && prev > 0;
    if (prev < 0 || progress > prev || hardReset) {
      setFocus(progress);
      autoStepRef.current = progress;
    } else if (progress >= 0) {
      autoStepRef.current = Math.max(prev, progress);
    }
  }, [status?.stage, previewAll]);

  const patchSite = (patch) => {
    setSiteForm((prev) => ({ ...(prev || payloadToSiteForm({})), ...patch }));
    // Live-preview finger opening on Arm ghost when Grasp open width changes.
    if (patch && Object.prototype.hasOwnProperty.call(patch, 'width_open')) {
      const cur = armGhostTarget.current;
      if (cur && Array.isArray(cur.joints_rad)) {
        armGhostTarget.set({
          ...cur,
          gripper_width_m: Number(patch.width_open),
        });
      }
    }
  };

  const fillSiteFromLocal = useCallback(async () => {
    setBusy(true);
    try {
      const st = await api('/api/grasp/site/payload', { method: 'GET' });
      setStatus(st);
      const form = payloadToSiteForm(st?.payload);
      form.source_note = 'local extrinsics / robot_cell_c / grasp_client';
      setSiteForm(form);
      setSiteFormSrc('local');
      message.success('已从本机标定与配置填入表单');
      return form;
    } catch (e) {
      message.error(String(e.message || e));
      throw e;
    } finally {
      setBusy(false);
    }
  }, []);

  const fillSiteFromRemote = useCallback(async () => {
    setBusy(true);
    try {
      const st = await api('/api/grasp/site/config', { method: 'GET' });
      setStatus(st);
      if (!st?.ok) throw new Error(st?.error || '拉取远端 site/config 失败');
      const form = remoteConfigToSiteForm(st.site_config, siteForm);
      setSiteForm(form);
      setSiteFormSrc('remote');
      message.success('已从远端 site/config 填入表单（可再编辑）');
      return form;
    } catch (e) {
      message.error(String(e.message || e));
      throw e;
    } finally {
      setBusy(false);
    }
  }, [siteForm]);

  const pushSiteConfigure = useCallback(async () => {
    setBusy(true);
    try {
      let form = siteForm;
      if (!form) {
        const st0 = await api('/api/grasp/site/payload', { method: 'GET' });
        form = payloadToSiteForm(st0?.payload);
        setSiteForm(form);
      }
      const payload = siteFormToPayload(form);
      const st = await api('/api/grasp/site/configure', {
        method: 'POST',
        body: JSON.stringify({ payload }),
      });
      setStatus(st);
      if (st?.config) {
        if (st.config.sam_text) setSamText(String(st.config.sam_text));
        if (st.config.hole_sam_text) setHoleText(String(st.config.hole_sam_text));
        if (st.config.insert_total_m != null) setInsertTotal(Number(st.config.insert_total_m));
        if (st.config.step_m != null) setStepM(Number(st.config.step_m));
        if (st.config.force_without_s2_exit != null) setForceFinal(!!st.config.force_without_s2_exit);
      }
      message.success('真机配置已推送到 grasp-service');
      return st;
    } catch (e) {
      message.error(String(e.message || e));
      refresh().catch(() => {});
      throw e;
    } finally {
      setBusy(false);
    }
  }, [siteForm, refresh]);

  useEffect(() => {
    if (!active || siteSeededRef.current) return undefined;
    siteSeededRef.current = true;
    fillSiteFromLocal().catch(() => { siteSeededRef.current = false; });
    return undefined;
  }, [active, fillSiteFromLocal]);

  const saveConfig = async () => {
    setBusy(true);
    try {
      const st = await api('/api/grasp/config', {
        method: 'PUT',
        body: JSON.stringify({
          base_url: baseUrl,
          sam_text: samText,
          hole_sam_text: holeText,
          include_s1: includeS1,
          prefer_hole_tcp: preferHole,
          s0_flip_approach: s0FlipApproach,
          s0_along_normal_m: s0AlongNormalM,
          s0_j6_rad: s0J6Rad,
          execute_motion: executeMotion,
          force_without_s2_exit: forceFinal,
          use_mock_frames: useMock,
          s2_hz: s2Hz,
          insert_total_m: insertTotal,
          step_m: stepM,
        }),
      });
      setStatus(st);
      message.success('配置已写入 grasp_client.yaml');
    } catch (e) {
      message.error(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const call = async (path, opts = {}, okMsg) => {
    setBusy(true);
    try {
      const method = String(opts.method || 'POST').toUpperCase();
      const { method: _m, body, ...rest } = opts;
      const fetchOpts = { method, ...rest };
      if (method !== 'GET' && method !== 'HEAD') {
        fetchOpts.body = body ?? '{}';
      }
      const st = await api(path, fetchOpts);
      setStatus(st);
      if (okMsg) message.success(okMsg);
      return st;
    } catch (e) {
      message.error(String(e.message || e));
      refresh().catch(() => {});
      throw e;
    } finally {
      setBusy(false);
    }
  };

  /** Hole detect → center ball + normal arrow (length = S0 offset, tip = S0). */
  const syncHolePoseToCanvas = async () => {
    setBusy(true);
    try {
      const r = await api('/api/grasp/hole/canvas_pose', { method: 'POST', body: '{}' });
      const pos = r?.position_m;
      const normal = r?.normal_base;
      if (!Array.isArray(pos) || pos.length < 3) {
        throw new Error('canvas_pose 未返回 position_m');
      }
      if (!Array.isArray(normal) || normal.length < 3) {
        throw new Error('canvas_pose 未返回 normal_base');
      }
      const arrowLen = Number(r?.normal_length_m) > 0 ? Number(r.normal_length_m) : 0.07;
      ['hole_grasp', 'hole_center', 'hole_pregrasp', 'hole_flange'].forEach((id) => {
        armSceneMarkers.remove(id);
      });
      const marked = armSceneMarkers.upsert({
        id: 'hole_center',
        label: 'Hole center',
        xyz: pos.slice(0, 3),
        color: ARM_MARKER_COLORS.hole_center,
        dir: normal.slice(0, 3),
        dir_length: arrowLen,
      });
      if (Array.isArray(r?.s0_pregrasp_base_m) && r.s0_pregrasp_base_m.length >= 3) {
        armSceneMarkers.upsert({
          id: 'hole_pregrasp',
          label: 'S0 (=hole+n·L)',
          xyz: r.s0_pregrasp_base_m.slice(0, 3),
          color: ARM_MARKER_COLORS.s0_pregrasp,
        });
      }
      if (!marked) throw new Error('点位无效');
      const L = Number(r?.s0_along_normal_m) || arrowLen;
      message.success(`已标出孔心 + 法向（箭头至 S0，L=${L}m）`);
      return r;
    } catch (e) {
      message.error(String(e.message || e));
      throw e;
    } finally {
      setBusy(false);
    }
  };

  /** S0 pregrasp (= TCP point) + approach dir ‖ hole normal; ghost IK. */
  const syncS0PoseToCanvas = async () => {
    setBusy(true);
    try {
      const r = await api('/api/grasp/s0/canvas_pose', { method: 'POST', body: '{}' });
      const xyzrpy = r?.xyzrpy;
      if (!Array.isArray(xyzrpy) || xyzrpy.length < 3) {
        throw new Error('canvas_pose 未返回 xyzrpy');
      }
      const tcpXyzrpy = Array.isArray(r?.tcp_xyzrpy) && r.tcp_xyzrpy.length >= 3
        ? r.tcp_xyzrpy
        : xyzrpy;
      const arrowLen = Number(r?.normal_length_m) > 0 ? Number(r.normal_length_m) : 0.1;
      const approach = Array.isArray(r?.normal_base) && r.normal_base.length >= 3
        ? r.normal_base
        : null;
      armSceneMarkers.remove('s0_pregrasp');
      const marked = armSceneMarkers.upsert({
        id: 's0_pregrasp_tcp',
        label: 'S0 TCP',
        xyz: tcpXyzrpy.slice(0, 3),
        color: ARM_MARKER_COLORS.s0_pregrasp,
        ...(approach ? { dir: approach.slice(0, 3), dir_length: arrowLen } : {}),
      });
      if (Array.isArray(r?.flange_xyzrpy) && r.flange_xyzrpy.length >= 3) {
        armSceneMarkers.upsert({
          id: 's0_pregrasp_flange',
          label: 'S0 flange',
          xyz: r.flange_xyzrpy.slice(0, 3),
          color: 0x94a3b8,
        });
      }
      const joints = r?.ik?.joints_rad;
      const ikOk = !!(r?.ik?.ok && Array.isArray(joints) && joints.length >= 6);
      const strategy = r?.ik?.strategy || '';
      if (ikOk) {
        const gripOpen = [
          (r?.gripper || {}).width_m,
          (status?.s0 || {}).gripper?.width_m,
          siteForm?.width_open,
          status?.config?.gripper_width_m_open,
        ].map(Number).find((v) => Number.isFinite(v));
        armGhostTarget.set({
          joints_rad: joints,
          xyzrpy: tcpXyzrpy,
          label: 'S0 TCP',
          source: r.source || 's0.pregrasp→tcp',
          gripper_width_m: Number.isFinite(gripOpen) ? gripOpen : undefined,
        });
      }
      if (!marked) throw new Error('点位无效');
      if (ikOk) {
        const extra = strategy ? ` · ${strategy}` : '';
        const soft = r?.ik?.preview_relaxed ? '（姿态近似）' : '';
        const j6 = Number(r?.s0_j6_rad ?? r?.ik?.s0_j6_rad);
        const j6Hint = Number.isFinite(j6) ? ` · J6=${j6.toFixed(2)}` : '';
        message.success(`已同步 S0 预抓取点 + 接近向（标记 + ghost${extra}${j6Hint}）${soft}`);
      } else {
        message.warning(
          `已标记 S0；IK 失败：${r?.ik?.error || '无 joints'}（仍可在画布看点位）`,
        );
      }
      return r;
    } catch (e) {
      message.error(String(e.message || e));
      throw e;
    } finally {
      setBusy(false);
    }
  };

  const stage = status?.stage || 'idle';
  const stepIdx = STAGE_STEP[stage] ?? 0;
  const scene = selection?.scene || selection?.d435_serial || status?.selection?.scene || '';
  const wrist = selection?.wrist || selection?.d405_serial || status?.selection?.wrist || '';
  const hole = status?.hole || {};
  const materialsInfo = status?.materials || {};
  const materialItems = Array.isArray(materialsInfo.items) ? materialsInfo.items : [];
  const selectedMaterialId = materialsInfo.selected_id;
  const s0 = status?.s0 || {};
  const viz = status?.viz || {};
  const vizHole = viz.hole || null;
  const vizMaterials = viz.materials || null;
  const vizS0 = viz.s0 || null;
  const vizHoleAt = (viz.meta && viz.meta.hole && viz.meta.hole.at) || '';
  const vizMaterialsAt = (viz.meta && viz.meta.materials && viz.meta.materials.at) || '';
  const vizS0At = (viz.meta && viz.meta.s0 && viz.meta.s0.at) || '';
  const finalPlan = status?.final || {};
  const pre = status?.pregrasp_target || {};
  const last = status?.last || {};
  const s2stats = status?.s2_stats || {};
  const motionS2 = ((last.s2_step || last.s2_latch || {}).result || last.s2_step || {}).motion
    || ((last.s2_step || {}).motion) || {};
  const teleS2 = ((last.s2_step || {}).result || last.s2_step || {}).telemetry
    || ((last.s2_latch || {}).result || last.s2_latch || {}).telemetry || {};
  const clientActions = (
    (last.s0_plan || {}).client_actions
    || (last.s2_step || {}).client_actions
    || (last.final_approach || {}).client_actions
    || (s0.client_actions)
    || (finalPlan.client_actions)
    || []
  );

  const holeTcp = hole.tcp_target || null;
  const holeCenter = hole.center_camera_m || null;
  const tcpPos = Array.isArray(holeTcp?.position_m) ? holeTcp.position_m : null;
  const tcpUsable = !!(tcpPos && tcpPos.every((v) => v != null));
  const gripperW = (s0.gripper || {}).width_m;
  const gripperAction = (s0.gripper || {}).action;
  const calibrated = s0.calibrated;
  const segments = (finalPlan.motion || {}).segment_count;
  const segM = (finalPlan.motion || {}).segment_m;
  const s2Progress = Math.min(100, Math.round(100 * (Number(s2stats.steps || 0) / Math.max(1, Number(status?.config?.s2_max_steps || 600)))));

  const siteSynced = stage === 'site_ok' || !!(status?.site?.synced) || stepIdx > 0;
  const mockOn = !!(status?.mock?.enabled ?? useMock);
  const mockReady = !!status?.mock?.ready;
  const stageTone = (stage === 'error' || stage === 'site_bad' || stage === 'health_bad')
    ? 'danger'
    : (stage === 'done' ? 'ok' : (status?.s2_running ? 'live' : 'idle'));
  const siteSummary = status?.site?.summary || {};

  const toggleMock = async (enabled) => {
    setBusy(true);
    try {
      const st = await api('/api/grasp/mock', {
        method: 'POST',
        body: JSON.stringify({ enabled: !!enabled }),
      });
      setStatus(st);
      setUseMock(!!enabled);
      message.success(enabled
        ? '已启用 Mock：从选定 capture / stock 发送 RGB-D'
        : '已关闭 Mock：恢复实时相机采图');
    } catch (e) {
      message.error(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const setMockSource = async (patch) => {
    setBusy(true);
    try {
      const st = await api('/api/grasp/mock', {
        method: 'POST',
        body: JSON.stringify(patch),
      });
      setStatus(st);
      if (st?.mock?.enabled != null) setUseMock(!!st.mock.enabled);
      const sid = patch.scene_id || patch.wrist_id || '';
      message.success(`Mock 源已切换：${sid || 'updated'}`);
    } catch (e) {
      message.error(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const showStep = (n) => previewAll || focus === n;

  const stepItems = useMemo(() => PIPELINE.map((p, i) => {
    let status = 'wait';
    if (stepIdx < 0) {
      status = i === focus ? 'error' : (i < Math.max(focus, 0) ? 'finish' : 'wait');
    } else if (i < stepIdx) {
      status = 'finish';
    } else if (i === stepIdx) {
      status = 'process';
    }
    // Viewing an earlier finished step: keep finish, Steps `current={focus}` marks it active.
    return {
      title: p.title,
      description: p.subtitle,
      status,
      disabled: false,
    };
  }), [stepIdx, focus]);

  return (
    <div className="grasp-page">
      <section className={`grasp-hero tone-${stageTone}`}>
        <div className="grasp-hero-main">
          <div className="eyebrow">grasp-service · Client Agent</div>
          <Title level={3} className="grasp-hero-title">孔抓取状态机 · S0 → S2 → Final</Title>
          <Paragraph className="grasp-hero-lead">
            服务端只输出计划与周期指令；采图、session、臂控与定时循环在本 Agent。
            推荐路径：Site sync → Hole → S0 → Move → S2 → Final。
          </Paragraph>
          <Space wrap size={[8, 8]}>
            <Tag color={stageTone === 'danger' ? 'error' : (stageTone === 'ok' ? 'success' : 'processing')}>
              stage · {stage}
            </Tag>
            <Tag color={executeMotion ? 'magenta' : 'default'}>{executeMotion ? 'LIVE 运动' : 'DRY-RUN'}</Tag>
            <Tag color={mockOn ? 'purple' : 'default'}>{mockOn ? '帧源 MOCK' : '帧源 LIVE'}</Tag>
            {status?.s2_running ? <Badge status="processing" text="S2 loop 运行中" /> : null}
            <Tag>session {status?.session_id ? `${String(status.session_id).slice(0, 8)}…` : '—'}</Tag>
            <Button size="small" onClick={() => refresh().catch((e) => message.error(String(e.message || e)))}>刷新</Button>
            <Button size="small" loading={busy} onClick={() => call('/api/grasp/reset', {}, '会话已重置').catch(() => {})}>重置</Button>
            <Button
              size="small"
              type={mockOn ? 'primary' : 'default'}
              loading={busy}
              disabled={!mockReady && !mockOn}
              onClick={() => toggleMock(!mockOn)}
            >
              {mockOn ? '关闭 Mock' : '启用 Mock'}
            </Button>
          </Space>
          {mockOn && (
            <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 10, maxWidth: 560 }}>
              <Alert
                type="warning"
                showIcon
                message="Mock：场景/腕帧可从 data/capture 列表选择（或 stock data/mock）。无需 M0 出流。"
              />
              <Space wrap align="center">
                <Text type="secondary">场景源</Text>
                <Select
                  size="small"
                  style={{ minWidth: 320 }}
                  loading={busy}
                  value={status?.mock?.scene_id || '__stock__'}
                  options={(status?.mock?.sources || [{ id: '__stock__', label: 'stock · data/mock' }]).map((s) => ({
                    value: s.id,
                    label: s.label || s.id,
                  }))}
                  showSearch
                  optionFilterProp="label"
                  onChange={(v) => setMockSource({ scene_id: v }).catch(() => {})}
                />
              </Space>
              <Space wrap align="center">
                <Text type="secondary">腕部源</Text>
                <Select
                  size="small"
                  style={{ minWidth: 320 }}
                  loading={busy}
                  value={status?.mock?.wrist_id || '__stock__'}
                  options={(status?.mock?.sources || [{ id: '__stock__', label: 'stock · data/mock' }]).map((s) => ({
                    value: s.id,
                    label: s.label || s.id,
                  }))}
                  showSearch
                  optionFilterProp="label"
                  onChange={(v) => setMockSource({ wrist_id: v }).catch(() => {})}
                />
              </Space>
            </div>
          )}
        </div>
        <div className="grasp-hero-side">
          <div className="grasp-hero-stat">
            <span>场景</span>
            <code>{mockOn ? (status?.mock?.scene?.color || 'data/mock/d435_color.png') : (scene || '未选定')}</code>
          </div>
          <div className="grasp-hero-stat">
            <span>腕部</span>
            <code>{mockOn ? (status?.mock?.wrist?.color || 'data/mock/d405_color.png') : (wrist || '未选定')}</code>
          </div>
          <div className="grasp-hero-stat">
            <span>服务</span>
            <code>{baseUrl.replace(/^https?:\/\//, '')}</code>
          </div>
        </div>
      </section>

      {status?.error ? (
        <Alert type="error" showIcon style={{ marginBottom: 16 }} message={status.error} />
      ) : null}

      <div className="grasp-metric-row">
        <Metric
          label="Site sync"
          value={siteSynced ? 'synced' : '—'}
          accent={siteSynced ? 'ok' : (stage === 'site_bad' ? 'warn' : undefined)}
          hint={siteSummary.robot_id || 'hand_eye / tcp / sam'}
        />
        <Metric label="Hole center (cam)" value={fmtVec3(holeCenter)} hint="m" />
        <Metric
          label="TCP usable"
          value={tcpUsable ? 'position_m ok' : (holeTcp ? 'null pose' : '—')}
          accent={tcpUsable ? 'ok' : (holeTcp ? 'warn' : undefined)}
          hint={tcpUsable ? fmtVec3(tcpPos) : '需 position_m≠null 才可 move'}
        />
        <Metric label="Calibrated" value={calibrated == null ? '—' : (calibrated ? 'true' : 'false')} accent={calibrated ? 'ok' : (calibrated === false ? 'warn' : undefined)} />
        <Metric
          label="Gripper"
          value={gripperW == null ? '—' : `${fmtNum(gripperW, 3)} m`}
          hint={gripperAction || ''}
        />
        <Metric label="S2 steps" value={String(s2stats.steps ?? 0)} hint={status?.s2_running ? 'looping' : (s2stats.last_exit ? 'exit' : '')} accent={status?.s2_running ? 'live' : undefined} />
        <Metric label="Final segs" value={segments == null ? '—' : `${segments}×${fmtNum(segM, 3)}`} hint="m" />
      </div>

      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col xs={24} lg={16}>
          <Card className="grasp-card" size="small" title="流水线引导">
            <div className="grasp-steps-head">
              <Steps
                size="small"
                current={Math.min(Math.max(focus, 0), 6)}
                status={stage === 'error' || stage === 'site_bad' || stage === 'health_bad' ? 'error' : (stage === 'done' ? 'finish' : 'process')}
                items={stepItems}
                onChange={(c) => {
                  const next = Math.min(Math.max(Number(c) || 0, 0), PIPELINE.length - 1);
                  setFocus(next);
                  // User chose this step — don't let the next status poll snap focus forward-only past it,
                  // but keep autoStep floor so later stage advances still follow.
                  if (autoStepRef.current < 0) autoStepRef.current = stepIdx < 0 ? 0 : stepIdx;
                }}
              />
              <Space>
                <Text type="secondary">展开全部</Text>
                <Switch checked={previewAll} onChange={setPreviewAll} checkedChildren="全" unCheckedChildren="步" />
              </Space>
            </div>

            {showStep(0) && (
              <div className={`grasp-step-block${focus === 0 ? ' is-focus' : ''}`}>
                <div className="grasp-step-head"><b>① Site sync</b><span>{PIPELINE[0].hint}</span></div>
                <Space wrap>
                  <ActionChip
                    title="从本机同步填入"
                    desc="extrinsics · robot · grasp_client"
                    primary
                    loading={busy}
                    onClick={() => fillSiteFromLocal().catch(() => {})}
                  />
                  <ActionChip
                    title="从远端填入"
                    desc="GET /v1/site/config"
                    loading={busy}
                    onClick={() => fillSiteFromRemote().catch(() => {})}
                  />
                  <ActionChip
                    title="推送到服务端"
                    desc="POST /v1/site/configure"
                    primary
                    loading={busy}
                    onClick={() => pushSiteConfigure().catch(() => {})}
                  />
                </Space>
                <Text type="secondary" style={{ display: 'block', marginTop: 8, fontSize: 12 }}>
                  表单来源：{siteFormSrc || '未加载'}{siteForm?.source_note ? ` · ${siteForm.source_note}` : ''}
                  {' · '}可改后再推送；推送会写回 grasp_client，并 merge 服务端进程级 SITE（非 per-session；多 worker 需各推一次）。
                </Text>

                {siteForm ? (
                  <div className="grasp-site-form">
                    <Row gutter={[12, 8]}>
                      <Col xs={24} md={12}>
                        <div className="grasp-field">
                          <label>sam.material_text（物料）</label>
                          <Input value={siteForm.material_text} onChange={(e) => patchSite({ material_text: e.target.value })} />
                        </div>
                        <div className="grasp-field">
                          <label>sam.hole_sam_text（孔）</label>
                          <Input value={siteForm.hole_sam_text} onChange={(e) => patchSite({ hole_sam_text: e.target.value })} />
                        </div>
                        <Space wrap size={[12, 8]} style={{ width: '100%' }}>
                          <div className="grasp-field inline">
                            <label>retreat_m</label>
                            <InputNumber min={0} max={0.5} step={0.005} value={siteForm.retreat_m} onChange={(v) => patchSite({ retreat_m: typeof v === 'number' ? v : siteForm.retreat_m })} />
                          </div>
                          <div className="grasp-field inline">
                            <label>insert_total_m</label>
                            <InputNumber min={0.001} max={0.1} step={0.001} value={siteForm.insert_total_m} onChange={(v) => patchSite({ insert_total_m: typeof v === 'number' ? v : siteForm.insert_total_m })} />
                          </div>
                          <div className="grasp-field inline">
                            <label>step_m</label>
                            <InputNumber min={0.0005} max={0.05} step={0.0005} value={siteForm.step_m} onChange={(v) => patchSite({ step_m: typeof v === 'number' ? v : siteForm.step_m })} />
                          </div>
                          <div className="grasp-field inline">
                            <label>speed m/s</label>
                            <InputNumber min={0.0005} max={0.05} step={0.0005} value={siteForm.linear_speed_m_s} onChange={(v) => patchSite({ linear_speed_m_s: typeof v === 'number' ? v : siteForm.linear_speed_m_s })} />
                          </div>
                        </Space>
                        <PoseEditor
                          title="TCP · flange→tcp"
                          hint="xyz m · rpy °（推送时转 4×4）"
                          xyz={siteForm.tcp_xyz}
                          rpyDeg={siteForm.gripper_rpy_deg}
                          onXyz={(xyz) => patchSite({ tcp_xyz: xyz })}
                          onRpy={(rpy) => patchSite({ gripper_rpy_deg: rpy })}
                        />
                        <div style={{ marginTop: 8 }}>
                          <div className="grasp-field" style={{ marginBottom: 6 }}>
                            <label>gripper 开度 open（总宽 m · 约 2×单侧 finger）</label>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 110px', gap: 8, alignItems: 'center' }}>
                              <Slider
                                min={0}
                                max={0.08}
                                step={0.001}
                                value={Math.max(0, Math.min(0.08, Number(siteForm.width_open) || 0))}
                                onChange={(v) => patchSite({ width_open: typeof v === 'number' ? v : siteForm.width_open })}
                              />
                              <InputNumber
                                min={0}
                                max={0.2}
                                step={0.001}
                                value={siteForm.width_open}
                                onChange={(v) => patchSite({ width_open: typeof v === 'number' ? v : siteForm.width_open })}
                                addonAfter="m"
                                style={{ width: 110 }}
                              />
                            </div>
                            <Text type="secondary" style={{ fontSize: 12 }}>
                              {((Number(siteForm.width_open) || 0) * 1000).toFixed(1)} mm 总开度
                              · 单侧约 {((Number(siteForm.width_open) || 0) * 500).toFixed(1)} mm
                            </Text>
                          </div>
                          <div className="grasp-field inline" style={{ marginBottom: 8 }}>
                            <label>closed</label>
                            <InputNumber min={0} max={0.2} step={0.005} value={siteForm.width_closed} onChange={(v) => patchSite({ width_closed: typeof v === 'number' ? v : siteForm.width_closed })} />
                          </div>
                        </div>
                        <Space wrap size={[12, 8]} style={{ marginTop: 8 }}>
                          <div className="grasp-field inline">
                            <label>robot_id</label>
                            <Input style={{ width: 140 }} value={siteForm.robot_id} onChange={(e) => patchSite({ robot_id: e.target.value })} />
                          </div>
                          <div className="grasp-field inline">
                            <label>robot_ip</label>
                            <Input style={{ width: 160 }} value={siteForm.robot_ip} onChange={(e) => patchSite({ robot_ip: e.target.value })} />
                          </div>
                        </Space>
                      </Col>
                      <Col xs={24} md={12}>
                        <div className="grasp-pose-stack">
                          <div className="grasp-switch" style={{ marginBottom: 8 }}>
                            <Switch checked={!!siteForm.calibrated} onChange={(v) => patchSite({ calibrated: v })} />
                            hand_eye.calibrated
                          </div>
                          <PoseEditor
                            title="场景手眼 · base←scene_cam"
                            hint="从本机 extrinsics.d435 填入 · xyz m / rpy °"
                            xyz={siteForm.scene_xyz}
                            rpyDeg={siteForm.scene_rpy_deg}
                            onXyz={(xyz) => patchSite({ scene_xyz: xyz })}
                            onRpy={(rpy) => patchSite({ scene_rpy_deg: rpy })}
                          />
                          <PoseEditor
                            title="腕手眼 · flange←wrist_cam"
                            hint="extrinsics.d405 · S2 服务端不用此 4×4，仅存盘"
                            xyz={siteForm.wrist_flange_xyz}
                            rpyDeg={siteForm.wrist_flange_rpy_deg}
                            onXyz={(xyz) => patchSite({ wrist_flange_xyz: xyz })}
                            onRpy={(rpy) => patchSite({ wrist_flange_rpy_deg: rpy })}
                          />
                          <div className="grasp-switch" style={{ margin: '8px 0' }}>
                            <Switch
                              checked={!!siteForm.wrist_base_enabled}
                              onChange={(v) => patchSite({ wrist_base_enabled: v })}
                            />
                            含腕→base（FK 合成，可关）
                          </div>
                          <PoseEditor
                            title="腕手眼 · base←wrist_cam"
                            hint="可选 · 关闭则推送 null"
                            xyz={siteForm.wrist_base_xyz}
                            rpyDeg={siteForm.wrist_base_rpy_deg}
                            disabled={!siteForm.wrist_base_enabled}
                            onXyz={(xyz) => patchSite({ wrist_base_xyz: xyz })}
                            onRpy={(rpy) => patchSite({ wrist_base_rpy_deg: rpy })}
                          />
                        </div>
                      </Col>
                    </Row>
                  </div>
                ) : (
                  <Alert type="info" showIcon style={{ marginTop: 10 }} message="点「从本机同步填入」加载可编辑真机数据" />
                )}

                <Descriptions size="small" column={2} style={{ marginTop: 10 }} bordered>
                  <Descriptions.Item label="hand_eye.calibrated">{String(siteSummary.hand_eye_calibrated ?? siteForm?.calibrated ?? '—')}</Descriptions.Item>
                  <Descriptions.Item label="retreat_m">{siteSummary.retreat_m ?? siteForm?.retreat_m ?? '—'}</Descriptions.Item>
                  <Descriptions.Item label="robot_id">{siteSummary.robot_id || siteForm?.robot_id || '—'}</Descriptions.Item>
                  <Descriptions.Item label="wrist_source">{siteSummary.wrist_source || siteForm?.wrist_source || '—'}</Descriptions.Item>
                  <Descriptions.Item label="last push" span={2}>
                    <code className="grasp-code">{shortJson((last.site_sync || {}).response || status?.site?.configure_response || '—', 360)}</code>
                  </Descriptions.Item>
                </Descriptions>
              </div>
            )}

            {showStep(1) && (
              <div className={`grasp-step-block${focus === 1 ? ' is-focus' : ''}`}>
                <div className="grasp-step-head"><b>② Hole detect</b><span>{PIPELINE[1].hint}</span></div>
                <Space wrap>
                  <ActionChip
                    title="检测全部物料"
                    desc="POST /v1/materials/detect"
                    primary
                    loading={busy}
                    onClick={() => call('/api/grasp/materials/detect', {}, '物料检测完成 — 请点选一个').catch(() => {})}
                  />
                  <ActionChip
                    title="画布显示孔心/法向"
                    desc={`孔心 − 法向→S0(${(s0AlongNormalM * 100).toFixed(0)}cm)`}
                    disabled={!(Array.isArray(hole.center_camera_m) && hole.center_camera_m.length >= 3)}
                    loading={busy}
                    onClick={() => syncHolePoseToCanvas().catch(() => {})}
                  />
                  <ActionChip
                    title="整图孔检测"
                    desc="跳过选择 · 旧路径"
                    loading={busy}
                    onClick={() => call('/api/grasp/hole/detect', {}, '整图孔检测完成').catch(() => {})}
                  />
                </Space>
                {materialItems.length > 0 && (
                  <div className="grasp-material-pick" style={{ marginTop: 10 }}>
                    <Text type="secondary" style={{ display: 'block', marginBottom: 6 }}>
                      候选 {materialItems.length} 个 — 点选后自动对该 ROI 做孔检测
                      {selectedMaterialId != null ? ` · 当前 #${selectedMaterialId}` : ''}
                    </Text>
                    <Space wrap>
                      {materialItems.map((m) => {
                        const id = Number(m.id);
                        const active = selectedMaterialId != null && Number(selectedMaterialId) === id;
                        const score = Number(m.score);
                        const area = Number(m.area_px);
                        return (
                          <Button
                            key={`mat-${id}`}
                            type={active ? 'primary' : 'default'}
                            loading={busy}
                            onClick={async () => {
                              setBusy(true);
                              try {
                                const st = await api('/api/grasp/materials/select', {
                                  method: 'POST',
                                  body: JSON.stringify({ material_id: id }),
                                });
                                setStatus(st);
                                message.success(`已选物料 #${id} 并完成孔检测`);
                                const center = st?.hole?.center_camera_m;
                                if (Array.isArray(center) && center.length >= 3) {
                                  await syncHolePoseToCanvas().catch(() => {});
                                }
                              } catch (e) {
                                message.error(String(e.message || e));
                                refresh().catch(() => {});
                              } finally {
                                setBusy(false);
                              }
                            }}
                          >
                            #{id}
                            {Number.isFinite(score) ? ` · ${score.toFixed(2)}` : ''}
                            {Number.isFinite(area) ? ` · ${Math.round(area)}px` : ''}
                            {active ? ' ✓' : ''}
                          </Button>
                        );
                      })}
                    </Space>
                  </div>
                )}
                <VizPane
                  path={vizMaterials || vizHole}
                  stamp={vizMaterialsAt || vizHoleAt}
                  title={vizMaterials ? '物料候选 · 编号框 / 选中高亮' : 'Hole 可视化 · center / pregrasp / normal / mask'}
                  empty="先「检测全部物料」，再点选候选"
                />
                {vizHole && vizMaterials && (
                  <VizPane
                    path={vizHole}
                    stamp={vizHoleAt}
                    title="选中物料的孔检测结果"
                    empty=""
                  />
                )}
                <Descriptions size="small" column={1} style={{ marginTop: 10 }} bordered>
                  <Descriptions.Item label="selected_material">
                    {selectedMaterialId != null ? `#${selectedMaterialId}` : '—'}
                    {hole.material_id != null ? ` · hole.material_id=${hole.material_id}` : ''}
                  </Descriptions.Item>
                  <Descriptions.Item label="center_camera_m">{fmtVec3(hole.center_camera_m)}</Descriptions.Item>
                  <Descriptions.Item label="plane_normal">{fmtVec3(hole.plane_normal_camera)}</Descriptions.Item>
                  <Descriptions.Item label="pregrasp_camera_m">{fmtVec3(hole.pregrasp_camera_m)}</Descriptions.Item>
                  <Descriptions.Item label="tcp_target">
                    <Tag color={tcpUsable ? 'green' : 'orange'}>{tcpUsable ? 'usable' : 'position_m null'}</Tag>
                    <code className="grasp-code">{shortJson(hole.tcp_target || '—', 360)}</code>
                  </Descriptions.Item>
                </Descriptions>
              </div>
            )}

            {showStep(2) && (
              <div className={`grasp-step-block${focus === 2 ? ' is-focus' : ''}`}>
                <div className="grasp-step-head"><b>③ S0 plan</b><span>{PIPELINE[2].hint}</span></div>
                <Space wrap>
                  <ActionChip
                    title="创建 session"
                    desc={`孔心−法向×${(s0AlongNormalM * 100).toFixed(0)}cm`}
                    primary
                    loading={busy}
                    disabled={!(Array.isArray(hole.center_camera_m) && hole.center_camera_m.length >= 3)}
                    onClick={() => call(
                      '/api/grasp/s0/plan',
                      {},
                      `S0 完成（孔心沿法向反方向 ${(s0AlongNormalM * 100).toFixed(0)}cm）`,
                    ).catch(() => {})}
                  />
                  <ActionChip
                    title="同步到画布"
                    desc={`S0=孔−${(s0AlongNormalM * 100).toFixed(0)}cm·n + ghost`}
                    disabled={!(Array.isArray(hole.center_camera_m) && hole.center_camera_m.length >= 3)}
                    loading={busy}
                    onClick={() => syncS0PoseToCanvas().catch(() => {})}
                  />
                  <span className="grasp-switch" style={{ marginLeft: 4 }}>
                    偏移
                    <InputNumber
                      style={{ width: 118, marginLeft: 6 }}
                      min={0.005}
                      max={0.5}
                      step={0.005}
                      value={s0AlongNormalM}
                      disabled={busy}
                      addonAfter="m"
                      onChange={async (v) => {
                        if (typeof v !== 'number' || !Number.isFinite(v) || v <= 0) return;
                        setS0AlongNormalM(v);
                        try {
                          await api('/api/grasp/config', {
                            method: 'PUT',
                            body: JSON.stringify({ s0_along_normal_m: v }),
                          });
                          message.info(`S0 偏移 = ${(v * 100).toFixed(1)} cm`);
                          if (Array.isArray(hole.center_camera_m) && hole.center_camera_m.length >= 3) {
                            await syncS0PoseToCanvas();
                          }
                        } catch (e) {
                          message.error(String(e.message || e));
                        }
                      }}
                    />
                  </span>
                  <span className="grasp-switch" style={{ marginLeft: 4 }}>
                    <Switch
                      checked={s0FlipApproach}
                      disabled={busy}
                      onChange={async (v) => {
                        setS0FlipApproach(v);
                        try {
                          await api('/api/grasp/config', {
                            method: 'PUT',
                            body: JSON.stringify({ s0_flip_approach: v }),
                          });
                          message.info(v ? '尖端朝下（默认）' : '不强制尖端朝下');
                          if (Array.isArray(hole.center_camera_m) && hole.center_camera_m.length >= 3) {
                            await syncS0PoseToCanvas();
                          }
                        } catch (e) {
                          message.error(String(e.message || e));
                        }
                      }}
                    />
                    {' '}尖端朝下
                  </span>
                  <span className="grasp-switch" style={{ marginLeft: 8 }}>
                    J6
                    <InputNumber
                      style={{ width: 120, marginLeft: 6 }}
                      min={-20}
                      max={20}
                      step={0.01}
                      value={s0J6Rad}
                      disabled={busy}
                      addonAfter="rad"
                      onChange={async (v) => {
                        if (typeof v !== 'number' || !Number.isFinite(v)) return;
                        setS0J6Rad(v);
                        try {
                          await api('/api/grasp/config', {
                            method: 'PUT',
                            body: JSON.stringify({ s0_j6_rad: v }),
                          });
                          message.info(`S0 J6 = ${v.toFixed(2)} rad`);
                          if (Array.isArray(hole.center_camera_m) && hole.center_camera_m.length >= 3) {
                            await syncS0PoseToCanvas();
                          }
                        } catch (e) {
                          message.error(String(e.message || e));
                        }
                      }}
                    />
                  </span>
                </Space>
                <VizPane
                  path={vizS0}
                  stamp={vizS0At}
                  title="S0 可视化 · servo_anchor / SAM mask / grasps"
                  empty="跑 S0 plan 后显示场景帧投影"
                />
                <Descriptions size="small" column={2} style={{ marginTop: 10 }} bordered>
                  <Descriptions.Item label="session_id" span={2}>
                    <Text code copyable={!!status?.session_id}>{status?.session_id || '—'}</Text>
                  </Descriptions.Item>
                  <Descriptions.Item label="material_id" span={2}>
                    {(s0.material_id != null || selectedMaterialId != null)
                      ? `#${s0.material_id ?? selectedMaterialId}`
                      : '—'}
                    {selectedMaterialId != null && s0.material_id != null
                      && Number(selectedMaterialId) !== Number(s0.material_id)
                      ? ' · 与 Hole 选择不一致，请重跑 S0'
                      : ''}
                  </Descriptions.Item>
                  <Descriptions.Item label="calibrated">{String(s0.calibrated ?? '—')}</Descriptions.Item>
                  <Descriptions.Item label="gripper.width_m">{gripperW == null ? '—' : fmtNum(gripperW, 4)}</Descriptions.Item>
                  <Descriptions.Item label="gripper.action">{gripperAction || '—'}</Descriptions.Item>
                  <Descriptions.Item label="pregrasp_flange" span={2}>
                    <code className="grasp-code">{shortJson((s0.motion || {}).pregrasp_flange || '—', 360)}</code>
                  </Descriptions.Item>
                  <Descriptions.Item label="mask_qc" span={2}>
                    <code className="grasp-code">{shortJson(s0.mask_qc || '—', 280)}</code>
                  </Descriptions.Item>
                </Descriptions>
              </div>
            )}

            {showStep(3) && (
              <div className={`grasp-step-block${focus === 3 ? ' is-focus' : ''}`}>
                <div className="grasp-step-head"><b>④ Move pre-grasp</b><span>{PIPELINE[3].hint}</span></div>
                <Space wrap>
                  <ActionChip
                    title={executeMotion ? 'LIVE 运动到位' : 'DRY-RUN 记录目标'}
                    desc="优先 hole.tcp_target"
                    primary
                    danger={executeMotion}
                    loading={busy}
                    onClick={() => call('/api/grasp/move_pregrasp', {}, 'Pre-grasp 已处理').catch(() => {})}
                  />
                  {includeS1 && (
                    <>
                      <ActionChip title="S1 latch" desc="场景 mask 锁存" loading={busy} onClick={() => call('/api/grasp/s1/latch').catch(() => {})} />
                      <ActionChip title="S1 step" desc="monitor_only" loading={busy} onClick={() => call('/api/grasp/s1/step').catch(() => {})} />
                    </>
                  )}
                </Space>
                <Descriptions size="small" column={1} style={{ marginTop: 10 }} bordered>
                  <Descriptions.Item label="source">{pre.source || '—'}</Descriptions.Item>
                  <Descriptions.Item label="xyzrpy">
                    <code className="grasp-code">{shortJson((last.move_pregrasp || {}).xyzrpy || pickPos(pre.target) || '—', 240)}</code>
                  </Descriptions.Item>
                  <Descriptions.Item label="gripper intent">{String((last.move_pregrasp || {}).gripper_width_m ?? gripperW ?? '—')}</Descriptions.Item>
                </Descriptions>
              </div>
            )}

            {showStep(4) && (
              <div className={`grasp-step-block${focus === 4 ? ' is-focus' : ''}`}>
                <div className="grasp-step-head"><b>⑤ S2 腕相机伺服</b><span>{PIPELINE[4].hint}</span></div>
                <Space wrap>
                  <ActionChip
                    title="S2 latch"
                    desc="腕 RGB-D 锁孔（前置）"
                    primary
                    loading={busy}
                    onClick={() => call('/api/grasp/s2/latch', {}, 'S2 latch 完成').catch(() => {})}
                  />
                </Space>
                <div className="grasp-s2-modes">
                  <div className="grasp-s2-mode">
                    <div className="grasp-s2-mode-title">单步调试</div>
                    <div className="grasp-s2-mode-desc">每点一次只与服务端通讯一帧，可反复点直到 exit_s2_servo。</div>
                    <ActionChip
                      title="单步 → 下一帧"
                      desc={`POST /v1/s2/step · 已 ${Number(s2stats.steps || 0)} 步`}
                      primary
                      loading={busy}
                      disabled={
                        !!status?.s2_running
                        || !status?.s2_latched
                        || !!(motionS2.exit_s2_servo || s2stats.last_exit)
                      }
                      onClick={() => call('/api/grasp/s2/step', {}, undefined).then((st) => {
                        const exited = !!(st?.exit_s2_servo
                          || ((st?.result || {}).motion || {}).exit_s2_servo
                          || st?.s2_stats?.last_exit);
                        const n = st?.steps ?? st?.s2_stats?.steps;
                        if (exited) message.success(`S2 单步结束：服务端 exit_s2_servo（共 ${n ?? '?'} 步）`);
                        else message.success(`S2 单步 #${n ?? '?'} 完成（未停止，可继续）`);
                      }).catch(() => {})}
                    />
                  </div>
                  <div className="grasp-s2-mode">
                    <div className="grasp-s2-mode-title">Loop 自动循环</div>
                    <div className="grasp-s2-mode-desc">后台按 {s2Hz} Hz 连续 step，直到服务端返回停止或手动停。</div>
                    <Space wrap>
                      <ActionChip
                        title="启动 Loop"
                        desc={`${s2Hz} Hz 直到 exit`}
                        primary
                        disabled={
                          !!status?.s2_running
                          || !status?.s2_latched
                          || !!(motionS2.exit_s2_servo || s2stats.last_exit)
                        }
                        loading={busy}
                        onClick={() => call('/api/grasp/s2/start', {}, 'S2 Loop 已启动').catch(() => {})}
                      />
                      <ActionChip
                        title="停止 Loop"
                        desc="客户端安全停"
                        danger
                        disabled={!status?.s2_running}
                        loading={busy}
                        onClick={() => call('/api/grasp/s2/stop', {}, 'S2 Loop 已停').catch(() => {})}
                      />
                    </Space>
                  </div>
                </div>
                <Alert
                  type={(motionS2.exit_s2_servo || s2stats.last_exit) ? 'success' : (status?.s2_latched ? 'info' : 'warning')}
                  showIcon
                  style={{ marginTop: 10 }}
                  message={
                    (motionS2.exit_s2_servo || s2stats.last_exit)
                      ? `服务端已 exit_s2_servo（${s2stats.mode || 'manual/loop'} · ${s2stats.steps ?? 0} 步）— 可进 Final`
                      : (status?.s2_running
                        ? 'Loop 运行中：持续 /v1/s2/step 直至 exit_s2_servo'
                        : (status?.s2_latched
                          ? '已 latch：请用「单步」调试，或「Loop」自动跑到停止'
                          : '尚未 S2 latch — Final 会 400'))
                  }
                  description="S2 服务端不下发 base 速度（仅 tool）；不用 wrist_hand_eye。未传 mask_b64 时每帧重跑孔+SAM，延迟大。"
                />
                <Progress percent={s2Progress} size="small" status={status?.s2_running ? 'active' : ((motionS2.exit_s2_servo || s2stats.last_exit) ? 'success' : 'normal')} style={{ marginTop: 12, maxWidth: 420 }} />
                <Row gutter={12} style={{ marginTop: 10 }}>
                  <Col span={12}>
                    <Descriptions size="small" column={1} bordered title="motion">
                      <Descriptions.Item label="mode">{s2stats.mode || (status?.s2_running ? 'loop' : '—')}</Descriptions.Item>
                      <Descriptions.Item label="command">{motionS2.command_type || '—'}</Descriptions.Item>
                      <Descriptions.Item label="frame">{motionS2.frame || 'tool?'}</Descriptions.Item>
                      <Descriptions.Item label="exit_s2">{String(motionS2.exit_s2_servo ?? s2stats.last_exit ?? '—')}</Descriptions.Item>
                      <Descriptions.Item label="velocity">
                        <code className="grasp-code">{shortJson(motionS2.velocity || '—', 200)}</code>
                      </Descriptions.Item>
                    </Descriptions>
                  </Col>
                  <Col span={12}>
                    <Descriptions size="small" column={1} bordered title="telemetry">
                      <Descriptions.Item label="u*,v*">
                        {teleS2?.targets
                          ? `${fmtNum(teleS2.targets.u_star_px, 1)}, ${fmtNum(teleS2.targets.v_star_px, 1)}`
                          : '—'}
                      </Descriptions.Item>
                      <Descriptions.Item label="Z*">{teleS2?.targets ? fmtNum(teleS2.targets.Z_star_m) : '—'}</Descriptions.Item>
                      <Descriptions.Item label="θ*">{teleS2?.targets ? fmtNum(teleS2.targets.theta_star_rad) : '—'}</Descriptions.Item>
                      <Descriptions.Item label="raw">
                        <code className="grasp-code">{shortJson(teleS2 || '—', 220)}</code>
                      </Descriptions.Item>
                    </Descriptions>
                  </Col>
                </Row>
              </div>
            )}

            {showStep(5) && (
              <div className={`grasp-step-block${focus === 5 ? ' is-focus' : ''}`}>
                <div className="grasp-step-head"><b>⑥ Final approach</b><span>{PIPELINE[5].hint}</span></div>
                <Space wrap>
                  <ActionChip title="拉取插入计划" desc="POST /v1/final/approach" primary loading={busy} onClick={() => call('/api/grasp/final/approach', {}, 'Final 计划就绪').catch(() => {})} />
                  <ActionChip
                    title={executeMotion ? 'LIVE 执行分段' : 'DRY-RUN 执行记录'}
                    desc="segments + close intent"
                    danger={executeMotion}
                    loading={busy}
                    onClick={() => call('/api/grasp/final/execute', {}, 'Final 执行结束').catch(() => {})}
                  />
                </Space>
                <Descriptions size="small" column={2} style={{ marginTop: 10 }} bordered>
                  <Descriptions.Item label="segment_count">{segments ?? '—'}</Descriptions.Item>
                  <Descriptions.Item label="segment_m">{segM ?? '—'}</Descriptions.Item>
                  <Descriptions.Item label="total_insert_m">{(finalPlan.motion || {}).total_insert_m ?? '—'}</Descriptions.Item>
                  <Descriptions.Item label="speed">{(finalPlan.motion || {}).linear_speed_m_s ?? '—'}</Descriptions.Item>
                  <Descriptions.Item label="axis / sign">
                    {(finalPlan.motion || {}).axis || '—'} / {(finalPlan.motion || {}).direction_sign ?? '—'}
                  </Descriptions.Item>
                  <Descriptions.Item label="pause_ms">{(finalPlan.motion || {}).pause_ms_between_steps ?? '—'}</Descriptions.Item>
                  <Descriptions.Item label="gripper" span={2}>
                    <code className="grasp-code">{shortJson(finalPlan.gripper || '—', 280)}</code>
                  </Descriptions.Item>
                </Descriptions>
              </div>
            )}

            {showStep(6) && (
              <div className={`grasp-step-block${focus === 6 ? ' is-focus' : ''}`}>
                <div className="grasp-step-head"><b>⑦ Done</b><span>{PIPELINE[6].hint}</span></div>
                <Alert
                  type={stage === 'done' ? 'success' : 'info'}
                  showIcon
                  message={stage === 'done' ? '本轮流水线已完成' : '完成 Final 执行后进入 Done'}
                  description="需要新一轮时点右上角「重置」，会清空 session 与缓存结果。"
                />
              </div>
            )}

            <Divider style={{ margin: '16px 0 12px' }} />
            <div className="grasp-oneshot">
              <div>
                <Text strong>一键孔抓取</Text>
                <Text type="secondary" style={{ display: 'block', fontSize: 12 }}>
                  site → hole → s0 → move → {includeS1 ? 's1 → ' : ''}s2 latch/loop → final → execute
                  {executeMotion ? '（LIVE）' : '（DRY-RUN）'}
                </Text>
              </div>
              <Button
                type="primary"
                size="large"
                danger={executeMotion}
                loading={busy}
                onClick={() => call('/api/grasp/run/hole', {
                  body: JSON.stringify({ include_s1: includeS1 }),
                }, '孔抓取流水线结束').catch(() => {})}
              >
                运行完整流水线
              </Button>
            </div>
          </Card>
        </Col>

        <Col xs={24} lg={8}>
          <Card
            className="grasp-card"
            size="small"
            title="相机 / Mock 预览"
            extra={mockOn ? <Tag color="purple">MOCK</Tag> : <Tag>LIVE</Tag>}
            style={{ marginBottom: 16 }}
          >
            <div className="grasp-cam-grid">
              <CamThumb
                serial={scene}
                label="场景 · S0/Hole"
                accent="blue"
                mockMode={mockOn}
                mockColorSrc={status?.mock?.preview?.scene?.color || status?.mock?.preview?.scene}
                mockDepthSrc={status?.mock?.preview?.scene?.depth}
                pathHint={status?.mock?.scene?.color || scene}
              />
              <CamThumb
                serial={wrist}
                label="腕部 · S2"
                accent="orange"
                mockMode={mockOn}
                mockColorSrc={status?.mock?.preview?.wrist?.color || status?.mock?.preview?.wrist}
                mockDepthSrc={status?.mock?.preview?.wrist?.depth}
                pathHint={status?.mock?.wrist?.color || wrist}
              />
            </div>
            <Space direction="vertical" size={6} style={{ width: '100%', marginTop: 10 }}>
              <Button
                block
                type={mockOn ? 'default' : 'primary'}
                loading={busy}
                disabled={!mockReady && !mockOn}
                onClick={() => toggleMock(!mockOn)}
              >
                {mockOn ? '切回实时相机' : '一键 Mock（发送存量 RGB-D）'}
              </Button>
              <Text type="secondary" style={{ fontSize: 12 }}>
                {mockOn
                  ? `上传 RGB-D：${status?.mock?.scene?.color || '…'} + depth（腕同配置）。请求含 color_b64 与 depth_b64。`
                  : '预览依赖 M0 出流（彩色+深度）。上传同样带 depth_b64。'}
              </Text>
            </Space>
          </Card>

          <Card className="grasp-card" size="small" title="对接检查清单" style={{ marginBottom: 16 }}>
            <ul className="grasp-checklist">
              {CHECKLIST.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </Card>

          {Array.isArray(clientActions) && clientActions.length > 0 && (
            <Card className="grasp-card" size="small" title="client_actions" style={{ marginBottom: 16 }}>
              <ol className="grasp-actions-list">
                {clientActions.map((a, i) => <li key={`${i}-${a}`}>{String(a)}</li>)}
              </ol>
            </Card>
          )}
        </Col>
      </Row>

      <Collapse
        className="grasp-collapse"
        defaultActiveKey={['cfg', 'log']}
        items={[
          {
            key: 'cfg',
            label: '连接与参数（写入 config/grasp_client.yaml）',
            children: (
              <Row gutter={[16, 12]}>
                <Col xs={24} md={12}>
                  <div className="grasp-field">
                    <label>grasp-service URL</label>
                    <Input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="http://127.0.0.1:6010" />
                  </div>
                  <div className="grasp-field">
                    <label>sam_text（物料 mask；hole/detect 与 S0 共用）</label>
                    <Input value={samText} onChange={(e) => setSamText(e.target.value)} />
                  </div>
                  <div className="grasp-field">
                    <label>hole_sam_text（仅 S2 latch；hole/detect 不接受此字段）</label>
                    <Input value={holeText} onChange={(e) => setHoleText(e.target.value)} />
                  </div>
                </Col>
                <Col xs={24} md={12}>
                  <Space wrap size={[16, 12]}>
                    <div className="grasp-field inline">
                      <label>S2 Hz</label>
                      <InputNumber min={1} max={30} value={s2Hz} onChange={(v) => setS2Hz(typeof v === 'number' ? v : s2Hz)} />
                    </div>
                    <div className="grasp-field inline">
                      <label>insert_total_m</label>
                      <InputNumber min={0.001} max={0.1} step={0.001} value={insertTotal} onChange={(v) => setInsertTotal(typeof v === 'number' ? v : insertTotal)} />
                    </div>
                    <div className="grasp-field inline">
                      <label>step_m</label>
                      <InputNumber min={0.0005} max={0.05} step={0.0005} value={stepM} onChange={(v) => setStepM(typeof v === 'number' ? v : stepM)} />
                    </div>
                  </Space>
                  <Space wrap size={[20, 12]} style={{ marginTop: 12 }}>
                    <Tooltip title="接近过程中监视场景 mask">
                      <span className="grasp-switch"><Switch checked={includeS1} onChange={setIncludeS1} /> 含 S1</span>
                    </Tooltip>
                    <Tooltip title="运动目标优先 hole.tcp_target">
                      <span className="grasp-switch"><Switch checked={preferHole} onChange={setPreferHole} /> 优先 hole.tcp</span>
                    </Tooltip>
                    <Tooltip title="开启后会真实 move_pose">
                      <span className="grasp-switch"><Switch checked={executeMotion} onChange={setExecuteMotion} checkedChildren="LIVE" unCheckedChildren="DRY" /> 执行运动</span>
                    </Tooltip>
                    <Tooltip title="请求 force_without_s2_exit；仍需服务端 policy.allow_force_without_s2_exit（Site 推送可打开）">
                      <span className="grasp-switch"><Switch checked={forceFinal} onChange={setForceFinal} /> force Final</span>
                    </Tooltip>
                    <Tooltip title="用 data/mock 存量图代替实时采图">
                      <span className="grasp-switch">
                        <Switch checked={useMock} onChange={(v) => { setUseMock(v); toggleMock(v); }} /> Mock 存量图
                      </span>
                    </Tooltip>
                  </Space>
                </Col>
                <Col span={24}>
                  <Space wrap>
                    <Button type="primary" loading={busy} onClick={saveConfig}>保存配置</Button>
                    <Button
                      loading={busy}
                      type="default"
                      onClick={() => pushSiteConfigure().catch(() => {})}
                    >
                      推送表单 → site configure
                    </Button>
                    <Button loading={busy} onClick={() => fillSiteFromLocal().catch(() => {})}>
                      从本机填入表单
                    </Button>
                    <Button loading={busy} onClick={() => call('/api/grasp/health').catch(() => {})}>测试 Health</Button>
                  </Space>
                </Col>
              </Row>
            ),
          },
          {
            key: 'log',
            label: '最近响应 / Agent 日志',
            children: (
              <Row gutter={16}>
                <Col xs={24} lg={12}>
                  <div className="grasp-panel-label">last payload</div>
                  <div className="train-log grasp-log">{shortJson(last, 3500)}</div>
                </Col>
                <Col xs={24} lg={12}>
                  <div className="grasp-panel-label">agent log</div>
                  <div className="train-log grasp-log" ref={logRef}>
                    {(status?.log_tail || []).join('\n') || '（尚无日志）'}
                  </div>
                </Col>
              </Row>
            ),
          },
        ]}
      />
    </div>
  );
}
