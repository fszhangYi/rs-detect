import { useEffect, useState, useCallback, useRef } from 'react';
import { Card, Button, Space, Tag, Alert, Input, InputNumber, Select, Slider, Divider, Descriptions, Switch, Typography, message } from 'antd';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import URDFLoader from 'urdf-loader';
import { api } from '../lib/api.js';
import { fmtCsv6, parseCsv6, copyText } from '../lib/format.js';
import { armGhostTarget, armSceneMarkers, ARM_MARKER_COLORS } from '../lib/armMarkers.js';
import { robotToThree } from '../lib/armCoords.js';
import { armPreload, preloadArmAssets } from '../lib/armPreload.js';

const { Text } = Typography;
const { TextArea } = Input;

/** Cell C EFG prismatic finger (m). Same as sensors-arm viz grip-open. */
const GRIP_OPEN_JOINT = "left_finger_joint";
const GRIP_OPEN_MIMIC = "right_finger_joint";
const GRIP_OPEN_DEFAULT_HI_M = 0.033;

function clampRound(v, min, max, step) {
  const s = Number(step);
  const stepSafe = Number.isFinite(s) && s > 0 ? s : 0.001;
  let x = Math.min(max, Math.max(min, Number(v) || 0));
  // snap to step grid relative to 0 for stable ± nudges
  x = Math.round(x / stepSafe) * stepSafe;
  x = Math.min(max, Math.max(min, x));
  const decimals = Math.min(8, Math.max(0, Math.ceil(-Math.log10(stepSafe) + 1e-12)));
  return Number(x.toFixed(decimals));
}

/** 4×4 T_base_cam → {offset_m, offset_rpy_deg} for camera CAD (Three.js XYZ / scipy xyz). */
function poseFromMat4(T) {
  if (!Array.isArray(T) || T.length < 3) {
    return { offset_m: [0, 0, 0], offset_rpy_deg: [0, 0, 0] };
  }
  const m00 = Number(T[0][0]), m01 = Number(T[0][1]), m02 = Number(T[0][2]);
  const m10 = Number(T[1][0]), m11 = Number(T[1][1]), m12 = Number(T[1][2]);
  const m20 = Number(T[2][0]), m21 = Number(T[2][1]), m22 = Number(T[2][2]);
  const sy = Math.sqrt(m00 * m00 + m10 * m10);
  let roll; let pitch; let yaw;
  if (sy > 1e-8) {
    roll = Math.atan2(m21, m22);
    pitch = Math.atan2(-m20, sy);
    yaw = Math.atan2(m10, m00);
  } else {
    roll = Math.atan2(-m12, m11);
    pitch = Math.atan2(-m20, sy);
    yaw = 0;
  }
  const offset_m = [
    Number(T[0][3]) || 0,
    Number(T[1][3]) || 0,
    Number(T[2][3]) || 0,
  ];
  const offset_rpy_deg = [roll, pitch, yaw].map((v) => (Number.isFinite(v) ? (v * 180) / Math.PI : 0));
  return { offset_m, offset_rpy_deg };
}

function NudgeSliderRow({
  label, value, min, max, step, digits, onChange, style,
}) {
  const stepSafe = Number(step) > 0 ? Number(step) : 0.001;
  const shown = Number(value) || 0;
  const setClamped = (v) => onChange(clampRound(v, min, max, stepSafe));
  return (
    <div className="arm-joint-row with-nudge" style={style}>
      <span className="lab">{label}</span>
      <Slider
        min={min}
        max={max}
        step={stepSafe}
        value={shown}
        onChange={setClamped}
      />
      <span className="arm-joint-nudge">
        <Button size="small" onClick={() => setClamped(shown - stepSafe)}>−</Button>
        <Button size="small" onClick={() => setClamped(shown + stepSafe)}>+</Button>
      </span>
      <span className="arm-joint-val">{shown.toFixed(digits)}</span>
    </div>
  );
}

function StepControls({ xyzStep, rpyStep, onXyzStep, onRpyStep, xyzMin = 0.0001, rpyMin = 0.01 }) {
  return (
    <div className="arm-step-row">
      <span className="arm-step-item">
        <span>XYZ step (m)</span>
        <InputNumber
          size="small"
          min={xyzMin}
          max={0.1}
          step={xyzMin}
          value={xyzStep}
          onChange={(v) => {
            const n = Number(v);
            if (Number.isFinite(n) && n > 0) onXyzStep(n);
          }}
          style={{ width: 92 }}
        />
      </span>
      <span className="arm-step-item">
        <span>RPY step (°)</span>
        <InputNumber
          size="small"
          min={rpyMin}
          max={30}
          step={rpyMin}
          value={rpyStep}
          onChange={(v) => {
            const n = Number(v);
            if (Number.isFinite(n) && n > 0) onRpyStep(n);
          }}
          style={{ width: 84 }}
        />
      </span>
    </div>
  );
}

function newDrawId(prefix) {
  return `${prefix}_${Date.now().toString(36)}_${Math.floor(Math.random() * 1e4).toString(36)}`;
}

function defaultDrawShape(type = "box") {
  const t = type === "sphere" || type === "cylinder" ? type : "box";
  const base = {
    id: newDrawId(t),
    type: t,
    label: t === "box" ? "长方体" : t === "sphere" ? "球体" : "圆柱",
    color: t === "box" ? "#5b8ff9" : t === "sphere" ? "#61ddaa" : "#f6bd16",
    opacity: 0.55,
    enabled: true,
    offset_m: [0.4, 0.0, 0.35],
    offset_rpy_deg: [0, 0, 0],
  };
  if (t === "sphere") base.radius_m = 0.05;
  else if (t === "cylinder") {
    base.radius_m = 0.04;
    base.height_m = 0.1;
  } else {
    base.size_m = [0.1, 0.08, 0.05];
  }
  return base;
}

function defaultDrawLayer(label) {
  const n = Math.max(1, Number(String(label || "").replace(/\D/g, "")) || 1);
  return {
    id: newDrawId("layer"),
    label: label || `绘制层 ${n}`,
    enabled: true,
    shapes: [],
  };
}

function makeShapeGeometry(shape) {
  const t = shape?.type || "box";
  if (t === "sphere") {
    const r = Math.max(1e-4, Number(shape.radius_m) || 0.05);
    return new THREE.SphereGeometry(r, 32, 24);
  }
  if (t === "cylinder") {
    const r = Math.max(1e-4, Number(shape.radius_m) || 0.04);
    const h = Math.max(1e-4, Number(shape.height_m) || 0.1);
    const g = new THREE.CylinderGeometry(r, r, h, 32);
    g.rotateX(Math.PI / 2); // height along ROS Z before Rx(-90°)
    return g;
  }
  const s = Array.isArray(shape.size_m) ? shape.size_m : [0.1, 0.08, 0.05];
  return new THREE.BoxGeometry(
    Math.max(1e-4, Number(s[0]) || 0.1),
    Math.max(1e-4, Number(s[1]) || 0.08),
    Math.max(1e-4, Number(s[2]) || 0.05),
  );
}

export default function ArmPanel({ active }) {
  const hostRef = useRef(null);
  const threeRef = useRef({});
  const modelRef = useRef(null);
  const liveJointsRef = useRef([0, 0, 0, 0, 0, 0]);
  const previewJointsRef = useRef([0, 0, 0, 0, 0, 0]);
  const [model, setModel] = useState(null);
  const [eoatId, setEoatId] = useState("cellc");
  const [eoatCatalog, setEoatCatalog] = useState([]);
  const [eoatBusy, setEoatBusy] = useState(false);
  const [gripOpenM, setGripOpenM] = useState(0);
  const gripOpenMRef = useRef(0);
  const [gripOpenCapable, setGripOpenCapable] = useState(false);
  const [gripOpenHiM, setGripOpenHiM] = useState(GRIP_OPEN_DEFAULT_HI_M);
  const [status, setStatus] = useState(null);
  const [joints, setJoints] = useState([0, 0, 0, 0, 0, 0]); // slider / preview target
  const [ghostVisible, setGhostVisible] = useState(false);
  const ghostVisibleRef = useRef(false);
  const [hint, setHint] = useState("准备加载 Cell C URDF…");
  const [preloadPct, setPreloadPct] = useState(armPreload.ready ? 100 : 0);
  const [dispatchMode, setDispatchMode] = useState("joints"); // joints | pose
  const [dispatchText, setDispatchText] = useState("");
  const [copyHint, setCopyHint] = useState("");
  const [sceneMarkers, setSceneMarkers] = useState(() => armSceneMarkers.items.slice());
  const [overlayCfg, setOverlayCfg] = useState(null);
  const [overlayVisible, setOverlayVisible] = useState(true);
  const [overlayBusy, setOverlayBusy] = useState(false);
  const [camOverlayCfg, setCamOverlayCfg] = useState(null);
  const [camOverlayVisible, setCamOverlayVisible] = useState(true);
  const camOverlayVisibleRef = useRef(true);
  const [camOverlayBusy, setCamOverlayBusy] = useState(false);
  const [activeCamId, setActiveCamId] = useState(null);
  const camGeomCacheRef = useRef({});
  const [sceneXyzStep, setSceneXyzStep] = useState(0.001);
  const [sceneRpyStep, setSceneRpyStep] = useState(0.5);
  const [camXyzStep, setCamXyzStep] = useState(0.001);
  const [camRpyStep, setCamRpyStep] = useState(0.5);
  const [layersOpen, setLayersOpen] = useState(true);
  const [robotVisible, setRobotVisible] = useState(true);
  const [markersVisible, setMarkersVisible] = useState(true);
  const [tcpVisible, setTcpVisible] = useState(true);
  const robotVisibleRef = useRef(true);
  const markersVisibleRef = useRef(true);
  const tcpVisibleRef = useRef(true);
  const [drawCfg, setDrawCfg] = useState(null);
  const [drawVisible, setDrawVisible] = useState(true);
  const drawVisibleRef = useRef(true);
  const [activeDrawLayerId, setActiveDrawLayerId] = useState(null);
  const [activeShapeId, setActiveShapeId] = useState(null);
  const [drawXyzStep, setDrawXyzStep] = useState(0.001);
  const [drawRpyStep, setDrawRpyStep] = useState(0.5);
  const [newShapeType, setNewShapeType] = useState("box");

  const jointSoftLimit = (mdl, i, side) => {
    const m = mdl || modelRef.current || model;
    const block = m && m.soft_limits_rad;
    const arr = block && block[side];
    const v = arr && arr[i];
    if (Number.isFinite(Number(v))) return Number(v);
    // Per-joint fallback (±π); never share one global min/max across axes.
    return side === "min" ? -Math.PI : Math.PI;
  };

  const gripperOpenJoint = (robotRoot) => {
    const jointsMap = robotRoot && robotRoot.joints;
    if (!jointsMap) return null;
    const j = jointsMap[GRIP_OPEN_JOINT];
    if (!j || typeof j.setJointValue !== "function") return null;
    const t = String(j.jointType || j.type || "").toLowerCase();
    if (t && t !== "prismatic") return null;
    return j;
  };

  const applyGripperOpeningTo = (robotRoot, meters) => {
    const j = gripperOpenJoint(robotRoot);
    if (!j) return false;
    const lo = (j.limit && Number.isFinite(Number(j.limit.lower))) ? Number(j.limit.lower) : 0;
    const hi = (j.limit && Number.isFinite(Number(j.limit.upper)))
      ? Number(j.limit.upper)
      : GRIP_OPEN_DEFAULT_HI_M;
    let v = Number(meters);
    if (!Number.isFinite(v)) v = lo;
    v = Math.max(lo, Math.min(hi, v));
    j.setJointValue(v);
    const jr = robotRoot.joints && robotRoot.joints[GRIP_OPEN_MIMIC];
    if (jr && typeof jr.setJointValue === "function") jr.setJointValue(v);
    if (robotRoot.updateMatrixWorld) robotRoot.updateMatrixWorld(true);
    return true;
  };

  const applyGripperOpening = () => {
    const v = gripOpenMRef.current;
    const t = threeRef.current || {};
    if (t.robot) applyGripperOpeningTo(t.robot, v);
    if (t.ghost) applyGripperOpeningTo(t.ghost, v);
  };

  const syncGripperOpenFromRobot = (robotRoot) => {
    const j = gripperOpenJoint(robotRoot);
    if (!j) {
      setGripOpenCapable(false);
      return false;
    }
    const lo = (j.limit && Number.isFinite(Number(j.limit.lower))) ? Number(j.limit.lower) : 0;
    const hi = (j.limit && Number.isFinite(Number(j.limit.upper)))
      ? Number(j.limit.upper)
      : GRIP_OPEN_DEFAULT_HI_M;
    setGripOpenCapable(true);
    setGripOpenHiM(hi);
    let cur = gripOpenMRef.current;
    if (!Number.isFinite(cur)) cur = lo;
    cur = Math.max(lo, Math.min(hi, cur));
    gripOpenMRef.current = cur;
    setGripOpenM(cur);
    return true;
  };

  /** Grasp total jaw width (m) → single-side prismatic finger (≈ total / 2). */
  const setGripOpenFromTotalWidth = (totalWidthM) => {
    const total = Number(totalWidthM);
    if (!Number.isFinite(total)) return;
    const hi = Number.isFinite(gripOpenHiM) ? gripOpenHiM : GRIP_OPEN_DEFAULT_HI_M;
    const side = Math.max(0, Math.min(hi, total * 0.5));
    gripOpenMRef.current = side;
    setGripOpenM(side);
    applyGripperOpening();
  };

  const onGripOpenMm = (mm) => {
    const hi = Number.isFinite(gripOpenHiM) ? gripOpenHiM : GRIP_OPEN_DEFAULT_HI_M;
    const v = Math.max(0, Math.min(hi, Number(mm) / 1000));
    if (!Number.isFinite(v)) return;
    gripOpenMRef.current = v;
    setGripOpenM(v);
    applyGripperOpening();
  };

  const paintJointsOn = (robot, j, mdl) => {
    const m = mdl || modelRef.current;
    if (!robot || !robot.joints || !m || !Array.isArray(j)) return;
    const names = m.joint_names || [];
    const offs = m.joint_offset_rad || [];
    const signs = m.joint_signs || [];
    for (let i = 0; i < Math.min(6, j.length, names.length); i++) {
      const joint = robot.joints[names[i]];
      if (!joint || typeof joint.setJointValue !== "function") continue;
      if (joint.limit && Number(joint.limit.lower) === 0 && Number(joint.limit.upper) === 0) {
        joint.limit.lower = jointSoftLimit(m, i, "min");
        joint.limit.upper = jointSoftLimit(m, i, "max");
      }
      const q = (signs[i] == null ? 1 : Number(signs[i])) * (Number(j[i]) - Number(offs[i] || 0));
      joint.setJointValue(q);
    }
    robot.updateMatrixWorld(true);
  };

  const paintLive = (j, mdl) => {
    paintJointsOn(threeRef.current.robot, j, mdl);
    applyGripperOpening();
  };

  const paintGhost = (j, mdl) => {
    const g = threeRef.current.ghost;
    if (!g) return;
    paintJointsOn(g, j, mdl);
    applyGripperOpening();
  };

  /** Apply cross-tab ghost target (e.g. Grasp S0 pose → IK joints). */
  const applyGhostTarget = (payload) => {
    if (!payload || !Array.isArray(payload.joints_rad) || payload.joints_rad.length < 6) return false;
    const q = payload.joints_rad.slice(0, 6).map(Number);
    previewJointsRef.current = q;
    setJoints(q);
    setGhostShown(true);
    if (payload.gripper_width_m != null) {
      setGripOpenFromTotalWidth(payload.gripper_width_m);
    }
    paintGhost(q);
    if (Array.isArray(payload.xyzrpy) && payload.xyzrpy.length >= 3) {
      setTcpMarker(payload.xyzrpy);
    }
    return true;
  };

  // Shared unlit ghost material (one GPU state for whole preview arm).
  // MeshBasicMaterial + mild opacity is far cheaper than per-mesh Standard + dispose/rebuild.
  const GHOST_COLOR_HEX = 0x22d3ee;
  const GHOST_OPACITY = 0.35;
  const ghostMatRef = useRef(null);

  const getGhostMaterial = () => {
    if (typeof THREE === "undefined") return null;
    if (!ghostMatRef.current) {
      ghostMatRef.current = new THREE.MeshBasicMaterial({
        color: GHOST_COLOR_HEX,
        transparent: true,
        opacity: GHOST_OPACITY,
        depthWrite: false,
        depthTest: true,
        toneMapped: false,
      });
    }
    return ghostMatRef.current;
  };

  const styleGhostMaterials = (root) => {
    if (!root || typeof THREE === "undefined") return;
    const ghostMat = getGhostMaterial();
    if (!ghostMat) return;
    root.traverse((obj) => {
      if (!obj.isMesh) return;
      // Already bound to shared ghost mat — skip (hot path safe).
      if (obj.material === ghostMat) {
        obj.renderOrder = 2;
        return;
      }
      obj.material = ghostMat;
      obj.renderOrder = 2;
      obj.userData.rsGhost = true;
    });
  };

  const setGhostShown = (show) => {
    const on = !!show;
    ghostVisibleRef.current = on;
    setGhostVisible(on);
    const g = threeRef.current.ghost;
    if (g) g.visible = on;
  };

  const setRobotShown = (show) => {
    const on = !!show;
    robotVisibleRef.current = on;
    setRobotVisible(on);
    const r = threeRef.current.robot;
    if (r) r.visible = on;
  };

  const setMarkersShown = (show) => {
    const on = !!show;
    markersVisibleRef.current = on;
    setMarkersVisible(on);
    const g = threeRef.current.markersGroup;
    if (g) g.visible = on;
  };

  const setTcpShown = (show) => {
    const on = !!show;
    tcpVisibleRef.current = on;
    setTcpVisible(on);
    const m = threeRef.current.tcpMarker;
    if (m) m.visible = on;
  };

  const setSceneLayerShown = async (show) => {
    const on = !!show;
    setOverlayVisible(on);
    if (!overlayCfg) return;
    if (on) {
      const next = { ...overlayCfg, enabled: true };
      if (!overlayCfg.enabled) setOverlayCfg(next);
      if (!threeRef.current?.sceneOverlay) {
        await loadSceneOverlay(next);
      } else {
        threeRef.current.sceneOverlay.visible = true;
      }
    } else if (threeRef.current?.sceneOverlay) {
      threeRef.current.sceneOverlay.visible = false;
    }
  };

  const setCamGroupShown = async (show) => {
    const on = !!show;
    camOverlayVisibleRef.current = on;
    setCamOverlayVisible(on);
    if (!camOverlayCfg) return;
    if (on) {
      const next = { ...camOverlayCfg, enabled: true };
      if (!camOverlayCfg.enabled) setCamOverlayCfg(next);
      if (!threeRef.current?.cameraOverlays) {
        await loadCameraOverlays(next, { selectedId: activeCamId, visible: true });
      } else {
        applyCamMatrices(next, activeCamId, { visible: true });
      }
    } else if (threeRef.current?.cameraOverlays) {
      threeRef.current.cameraOverlays.visible = false;
    }
  };

  const setCamLayerShown = (camId, show) => {
    if (!camOverlayCfg) return;
    const cameras = (camOverlayCfg.cameras || []).map((c) =>
      c.id === camId ? { ...c, enabled: !!show } : c,
    );
    const next = { ...camOverlayCfg, cameras };
    setCamOverlayCfg(next);
    applyCamMatrices(next, activeCamId);
  };

  const setDrawGroupShown = async (show) => {
    const on = !!show;
    drawVisibleRef.current = on;
    setDrawVisible(on);
    if (!drawCfg) return;
    if (on) {
      const next = { ...drawCfg, enabled: true };
      if (!drawCfg.enabled) setDrawCfg(next);
      if (!threeRef.current?.drawShapes) {
        await loadDrawShapes(next, { visible: true });
      } else {
        applyDrawShapes(next, undefined, { visible: true });
      }
    } else if (threeRef.current?.drawShapes) {
      threeRef.current.drawShapes.visible = false;
    }
  };

  const setDrawLayerShown = (layerId, show) => {
    if (!drawCfg) return;
    const layers = (drawCfg.layers || []).map((L) =>
      L.id === layerId ? { ...L, enabled: !!show } : L,
    );
    const next = { ...drawCfg, layers };
    setDrawCfg(next);
    applyDrawShapes(next);
  };

  const setDrawShapeShown = (layerId, shapeId, show) => {
    if (!drawCfg) return;
    const layers = (drawCfg.layers || []).map((L) => {
      if (L.id !== layerId) return L;
      return {
        ...L,
        shapes: (L.shapes || []).map((s) =>
          s.id === shapeId ? { ...s, enabled: !!show } : s,
        ),
      };
    });
    const next = { ...drawCfg, layers };
    setDrawCfg(next);
    applyDrawShapes(next);
  };

  /** Toggle visibility of a canvas model layer (does not persist config). */
  const setLayerVisible = (layerId, show) => {
    if (layerId === "robot") setRobotShown(show);
    else if (layerId === "ghost") setGhostShown(show);
    else if (layerId === "scene") setSceneLayerShown(show);
    else if (layerId === "cam_group") setCamGroupShown(show);
    else if (layerId === "draw_group") setDrawGroupShown(show);
    else if (layerId === "markers") setMarkersShown(show);
    else if (layerId === "tcp") setTcpShown(show);
    else if (layerId.startsWith("cam:")) setCamLayerShown(layerId.slice(4), show);
    else if (layerId.startsWith("draw_layer:")) setDrawLayerShown(layerId.slice(11), show);
    else if (layerId.startsWith("draw_shape:")) {
      const rest = layerId.slice(11);
      const idx = rest.indexOf(":");
      if (idx > 0) setDrawShapeShown(rest.slice(0, idx), rest.slice(idx + 1), show);
    }
  };

  const layerRows = (() => {
    const urdfPath = String(model?.urdf || status?.model?.urdf || "cs66_cell_c.urdf").split("?")[0];
    const urdfName = urdfPath.split("/").pop() || urdfPath;
    const sceneUrl = overlayCfg?.url || "/assets/scene/g26562-01-00_asm.stl";
    const sceneName = sceneUrl.split("/").pop() || sceneUrl;
    const rows = [
      {
        id: "robot",
        label: "机械臂 URDF",
        file: urdfName,
        color: "#b0b8c4",
        visible: robotVisible,
        loaded: !!threeRef.current?.robot || !!model,
      },
      {
        id: "ghost",
        label: "幽灵预览",
        file: urdfName,
        color: "#22d3ee",
        visible: ghostVisible,
        loaded: true,
      },
      {
        id: "scene",
        label: overlayCfg?.name || "场景 CAD",
        file: sceneName,
        color: overlayCfg?.color || "#7a93ad",
        visible: !!overlayCfg?.enabled && overlayVisible,
        loaded: !!overlayCfg,
      },
      {
        id: "cam_group",
        label: "相机组",
        file: "realsense/*.stl",
        color: "#e8e8e8",
        visible: !!camOverlayCfg?.enabled && camOverlayVisible,
        loaded: !!camOverlayCfg,
      },
    ];
    (camOverlayCfg?.cameras || []).forEach((c) => {
      const file = String(c.url || "").split("/").pop() || c.model || "mesh";
      rows.push({
        id: `cam:${c.id}`,
        label: c.label || c.id,
        file,
        color: c.color || "#e85d4c",
        visible: !!c.enabled && !!camOverlayCfg?.enabled && camOverlayVisible,
        loaded: true,
        indent: true,
      });
    });
    rows.push({
      id: "draw_group",
      label: "绘制组",
      file: "cuboid/sphere/cylinder",
      color: "#5b8ff9",
      visible: !!drawCfg?.enabled && drawVisible,
      loaded: !!drawCfg,
    });
    (drawCfg?.layers || []).forEach((L) => {
      rows.push({
        id: `draw_layer:${L.id}`,
        label: L.label || L.id,
        file: `${(L.shapes || []).length} 个图元`,
        color: "#9ec5ff",
        visible: !!L.enabled && !!drawCfg?.enabled && drawVisible,
        loaded: true,
        indent: true,
      });
      (L.shapes || []).forEach((s) => {
        rows.push({
          id: `draw_shape:${L.id}:${s.id}`,
          label: s.label || s.id,
          file: s.type || "shape",
          color: s.color || "#5b8ff9",
          visible: !!s.enabled && !!L.enabled && !!drawCfg?.enabled && drawVisible,
          loaded: true,
          indent: true,
        });
      });
    });
    rows.push(
      {
        id: "tcp",
        label: "TCP 标记",
        file: "—",
        color: "#f0b429",
        visible: tcpVisible,
        loaded: true,
      },
      {
        id: "markers",
        label: "M2 点位",
        file: "—",
        color: "#3dd6c6",
        visible: markersVisible,
        loaded: true,
      },
    );
    return rows;
  })();

  const setTcpMarker = (xyzrpy) => {
    const { tcpMarker } = threeRef.current;
    if (!tcpMarker || !xyzrpy || xyzrpy.length < 3) return;
    const p = robotToThree(xyzrpy[0], xyzrpy[1], xyzrpy[2]);
    tcpMarker.position.set(p.x, p.y, p.z);
  };

  const syncSceneMarkers = (items) => {
    const list = Array.isArray(items) ? items : [];
    setSceneMarkers(list.slice());
    const { markersGroup, scene } = threeRef.current;
    if (!markersGroup || !scene || typeof THREE === "undefined") return;
    while (markersGroup.children.length) {
      const ch = markersGroup.children[0];
      markersGroup.remove(ch);
      if (ch.isArrowHelper) {
        if (ch.line) {
          ch.line.geometry?.dispose?.();
          ch.line.material?.dispose?.();
        }
        if (ch.cone) {
          ch.cone.geometry?.dispose?.();
          ch.cone.material?.dispose?.();
        }
      } else {
        if (ch.geometry) ch.geometry.dispose();
        if (ch.material) {
          if (Array.isArray(ch.material)) ch.material.forEach((m) => m.dispose && m.dispose());
          else if (ch.material.dispose) ch.material.dispose();
        }
      }
    }
    list.forEach((item) => {
      const color = item.color != null ? item.color : ARM_MARKER_COLORS.default;
      const ball = new THREE.Mesh(
        new THREE.SphereGeometry(0.016, 18, 14),
        new THREE.MeshStandardMaterial({
          color,
          emissive: color,
          emissiveIntensity: 0.35,
          metalness: 0.1,
          roughness: 0.35,
        })
      );
      const p = robotToThree(item.xyz[0], item.xyz[1], item.xyz[2]);
      ball.position.set(p.x, p.y, p.z);
      ball.name = item.id || item.label || "marker";
      // vertical stem for readability
      const stemH = Math.max(0.04, p.y);
      const stem = new THREE.Mesh(
        new THREE.CylinderGeometry(0.0025, 0.0025, stemH, 8),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.55 })
      );
      stem.position.set(p.x, stemH / 2, p.z);
      markersGroup.add(stem);
      markersGroup.add(ball);

      // Optional direction arrow (robot/base → Three: x,y,z → x,z,-y)
      const dir = item.dir;
      if (Array.isArray(dir) && dir.length >= 3) {
        const dThree = new THREE.Vector3(dir[0], dir[2], -dir[1]);
        const lenN = dThree.length();
        if (lenN > 1e-8) {
          dThree.multiplyScalar(1 / lenN);
          const arrowLen = Number(item.dir_length) > 0 ? Number(item.dir_length) : 0.1;
          const origin = new THREE.Vector3(p.x, p.y, p.z);
          const arrow = new THREE.ArrowHelper(
            dThree,
            origin,
            arrowLen,
            color,
            Math.min(0.035, arrowLen * 0.35),
            Math.min(0.02, arrowLen * 0.2),
          );
          arrow.name = `${item.id || item.label || "marker"}_dir`;
          markersGroup.add(arrow);
        }
      }
    });
  };

  const applyStatus = (st, { syncSliders = true } = {}) => {
    setStatus(st);
    if (st.model) {
      modelRef.current = st.model;
      setModel(st.model);
      if (st.model.eoat_id) setEoatId(String(st.model.eoat_id));
    }
    if (st.joints_rad) {
      liveJointsRef.current = st.joints_rad.slice();
      // Live mesh always follows measured / backend joints — never slider preview.
      paintLive(st.joints_rad, st.model);
      if (syncSliders) {
        previewJointsRef.current = st.joints_rad.slice();
        setJoints(st.joints_rad.slice());
        // Keep ghost pose in sync when preview is hidden.
        if (!ghostVisibleRef.current) paintGhost(st.joints_rad, st.model);
      }
    }
    setTcpMarker(st.cartesian_xyzrpy);
  };

  const refreshEoatCatalog = async () => {
    const data = await api("/api/arm/eoat");
    const items = Array.isArray(data?.catalog) ? data.catalog : [];
    setEoatCatalog(items);
    if (data?.eoat_id) setEoatId(String(data.eoat_id));
    if (data?.model) {
      modelRef.current = data.model;
      setModel(data.model);
    }
    return data;
  };

  const switchEoat = async (nextId) => {
    if (!nextId || nextId === eoatId) return;
    setEoatBusy(true);
    try {
      const data = await api("/api/arm/eoat", {
        method: "PUT",
        body: JSON.stringify({ eoat_id: nextId, persist: true }),
      });
      const mdl = data?.model
        || (data?.eoat
          ? {
              ...(modelRef.current || {}),
              urdf: data.eoat.urdf,
              eoat_id: nextId,
              tcp_xyz: data.eoat.tcp_xyz,
            }
          : null);
      if (data?.eoat_id) setEoatId(String(data.eoat_id));
      if (Array.isArray(data?.catalog)) setEoatCatalog(data.catalog);
      if (mdl) {
        modelRef.current = mdl;
        setModel(mdl);
        await preloadArmAssets({ force: true, model: mdl });
        loadUrdf(mdl);
        message.success(`末端工装已切换为 ${nextId}`);
      }
    } catch (e) {
      message.error(String(e.message || e));
    } finally {
      setEoatBusy(false);
    }
  };

  const mountUrdfInstance = (arm, r, asGhost) => {
    r.rotation.x = -Math.PI / 2;
    if (arm.mirror_y) r.scale.y = -1;
    const rpy = arm.gripper_rpy;
    if (rpy && r.links) {
      const link = r.links.gripper_base_link
        || r.links.gripper_centor_link
        || r.links.gripper_center_link
        || r.links.ee_link
        || null;
      if (link) {
        const extra = new THREE.Euler(rpy[0] || 0, rpy[1] || 0, rpy[2] || 0, "XYZ");
        link.quaternion.multiply(new THREE.Quaternion().setFromEuler(extra));
        link.updateMatrix();
      }
    }
    if (asGhost) {
      styleGhostMaterials(r);
      r.visible = !!ghostVisibleRef.current;
      r.name = "arm_ghost_preview";
    } else {
      r.name = "arm_live";
    }
    return r;
  };

  const buildOverlayMatrix = (cfg) => {
    const scale = Number(cfg.scale ?? 0.001);
    const baseMm = cfg.align?.base_in_scene_mm || [0, 0, 0];
    const yaw = (Number(cfg.align?.yaw_deg || 0) * Math.PI) / 180;
    const off = cfg.offset_m || [0, 0, 0];
    const orpy = (cfg.offset_rpy_deg || [0, 0, 0]).map((d) => (Number(d) * Math.PI) / 180);
    const S = new THREE.Matrix4().makeScale(scale, scale, scale);
    const Tbase = new THREE.Matrix4().makeTranslation(
      -Number(baseMm[0]) * scale,
      -Number(baseMm[1]) * scale,
      -Number(baseMm[2]) * scale,
    );
    const Ryaw = new THREE.Matrix4().makeRotationZ(-yaw);
    const Roff = new THREE.Matrix4().makeRotationFromEuler(
      new THREE.Euler(orpy[0] || 0, orpy[1] || 0, orpy[2] || 0, "XYZ"),
    );
    const Toff = new THREE.Matrix4().makeTranslation(
      Number(off[0]) || 0,
      Number(off[1]) || 0,
      Number(off[2]) || 0,
    );
    // Same Rx(-90°) as URDF robot group (ROS Z-up → Three Y-up)
    const Rx = new THREE.Matrix4().makeRotationX(-Math.PI / 2);
    // p_three = Rx * Toff * Roff * Ryaw * Tbase * S * p_cad
    const m = new THREE.Matrix4();
    m.identity();
    m.multiply(Rx);
    m.multiply(Toff);
    m.multiply(Roff);
    m.multiply(Ryaw);
    m.multiply(Tbase);
    m.multiply(S);
    return m;
  };

  const applyOverlayMatrix = (cfg) => {
    const t = threeRef.current;
    if (!t?.sceneOverlay || !cfg) return;
    t.sceneOverlay.matrixAutoUpdate = false;
    t.sceneOverlay.matrix.copy(buildOverlayMatrix(cfg));
    t.sceneOverlay.updateMatrixWorld(true);
  };

  const loadSceneOverlay = async (cfg) => {
    const t = threeRef.current;
    if (!t?.scene) return;
    if (t.sceneOverlay) {
      t.scene.remove(t.sceneOverlay);
      t.sceneOverlay.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
          else obj.material.dispose();
        }
      });
      t.sceneOverlay = null;
    }
    if (!cfg?.enabled || !cfg.url) return;
    setOverlayBusy(true);
    try {
      const loader = new STLLoader();
      const geom = await new Promise((resolve, reject) => {
        loader.load(cfg.url, resolve, undefined, reject);
      });
      geom.computeVertexNormals();
      const color = new THREE.Color(cfg.color || "#7a93ad");
      const mat = new THREE.MeshStandardMaterial({
        color,
        transparent: true,
        opacity: Number(cfg.opacity ?? 0.32),
        metalness: 0.08,
        roughness: 0.88,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.name = "scene_overlay_mesh";
      const group = new THREE.Group();
      group.name = "scene_overlay";
      group.add(mesh);
      group.visible = overlayVisible;
      group.matrixAutoUpdate = false;
      group.matrix.copy(buildOverlayMatrix(cfg));
      group.updateMatrixWorld(true);
      t.sceneOverlay = group;
      t.scene.add(group);
    } catch (e) {
      console.warn("[scene_overlay]", e);
      message.warning("场景 STL 加载失败: " + (e.message || e));
    } finally {
      setOverlayBusy(false);
    }
  };

  const saveOverlayCfg = async (next) => {
    try {
      const saved = await api("/api/scene_overlay", {
        method: "PUT",
        body: JSON.stringify(next),
      });
      setOverlayCfg(saved);
      applyOverlayMatrix(saved);
      if (threeRef.current?.sceneOverlay) {
        threeRef.current.sceneOverlay.visible = !!saved.enabled && overlayVisible;
        const mesh = threeRef.current.sceneOverlay.children[0];
        if (mesh?.material) {
          mesh.material.opacity = Number(saved.opacity ?? 0.32);
          if (saved.color) mesh.material.color.set(saved.color);
          mesh.material.needsUpdate = true;
        }
      }
      message.success("场景叠加已保存（未改 URDF）");
      return saved;
    } catch (e) {
      message.error(String(e.message || e));
      return null;
    }
  };

  const buildCamMatrix = (cam) => {
    const scale = Number(cam.scale ?? (cam.units_of_file === "mm" ? 0.001 : 1));
    const off = cam.offset_m || [0, 0, 0];
    const orpy = (cam.offset_rpy_deg || [0, 0, 0]).map((d) => (Number(d) * Math.PI) / 180);
    const S = new THREE.Matrix4().makeScale(scale, scale, scale);
    const Roff = new THREE.Matrix4().makeRotationFromEuler(
      new THREE.Euler(orpy[0] || 0, orpy[1] || 0, orpy[2] || 0, "XYZ"),
    );
    const Toff = new THREE.Matrix4().makeTranslation(
      Number(off[0]) || 0,
      Number(off[1]) || 0,
      Number(off[2]) || 0,
    );
    // Same Rx(-90°) as URDF / scene overlay (ROS Z-up → Three Y-up)
    const Rx = new THREE.Matrix4().makeRotationX(-Math.PI / 2);
    const m = new THREE.Matrix4();
    m.identity();
    m.multiply(Rx);
    m.multiply(Toff);
    m.multiply(Roff);
    m.multiply(S);
    return m;
  };

  const disposeObject3D = (root) => {
    if (!root) return;
    root.traverse((obj) => {
      if (obj.geometry && !obj.geometry.__rsShared) obj.geometry.dispose();
      if (obj.material) {
        if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
        else obj.material.dispose();
      }
    });
  };

  const styleCamMaterials = (group, cfg, selectedId) => {
    if (!group || !cfg) return;
    const cams = cfg.cameras || [];
    group.children.forEach((child) => {
      const cam = cams.find((c) => c.id === child.name);
      if (!cam || !child.isMesh || !child.material) return;
      const selected = selectedId && cam.id === selectedId;
      const baseOp = Number(cam.opacity ?? cfg.opacity ?? 0.72);
      child.material.opacity = selected ? Math.min(0.95, baseOp + 0.15) : baseOp;
      if (cam.color) child.material.color.set(cam.color);
      if (child.material.emissive) {
        child.material.emissive.set(selected ? cam.color || "#ffffff" : "#000000");
        child.material.emissiveIntensity = selected ? 0.22 : 0;
      }
      child.material.needsUpdate = true;
      child.visible = !!cam.enabled;
    });
  };

  const applyCamMatrices = (cfg, selectedId, opts = {}) => {
    const t = threeRef.current;
    if (!t?.cameraOverlays || !cfg) return;
    const cams = cfg.cameras || [];
    const show = opts.visible !== undefined ? !!opts.visible : camOverlayVisibleRef.current;
    t.cameraOverlays.visible = !!cfg.enabled && show;
    t.cameraOverlays.children.forEach((child) => {
      const cam = cams.find((c) => c.id === child.name);
      if (!cam) return;
      child.matrixAutoUpdate = false;
      child.matrix.copy(buildCamMatrix(cam));
      child.updateMatrixWorld(true);
    });
    styleCamMaterials(t.cameraOverlays, cfg, selectedId ?? activeCamId);
  };

  const loadCameraOverlays = async (cfg, opts = {}) => {
    const t = threeRef.current;
    if (!t?.scene) return;
    if (t.cameraOverlays) {
      t.scene.remove(t.cameraOverlays);
      disposeObject3D(t.cameraOverlays);
      t.cameraOverlays = null;
    }
    if (!cfg?.enabled) return;
    const cams = Array.isArray(cfg.cameras) ? cfg.cameras : [];
    if (!cams.length) return;
    const selectedId = opts.selectedId !== undefined ? opts.selectedId : activeCamId;
    const show = opts.visible !== undefined ? !!opts.visible : camOverlayVisibleRef.current;
    setCamOverlayBusy(true);
    try {
      const loader = new STLLoader();
      const group = new THREE.Group();
      group.name = "camera_overlays";
      for (const cam of cams) {
        const url = cam.url;
        if (!url) continue;
        let geom = camGeomCacheRef.current[url];
        if (!geom) {
          geom = await new Promise((resolve, reject) => {
            loader.load(
              url,
              resolve,
              undefined,
              (err) => {
                const msg = String(err?.message || err || "STL load failed");
                reject(new Error(
                  `${url}: ${msg}`
                  + (msg.includes("DataView")
                    ? "（常见原因：web/assets/realsense/*.stl 仍是 Linux symlink 占位文本，请跑 scripts/ensure_realsense_assets.py）"
                    : ""),
                ));
              },
            );
          });
          geom.computeVertexNormals();
          geom.__rsShared = true;
          camGeomCacheRef.current[url] = geom;
        }
        const mat = new THREE.MeshStandardMaterial({
          color: new THREE.Color(cam.color || "#e85d4c"),
          transparent: true,
          opacity: Number(cam.opacity ?? 0.72),
          metalness: 0.12,
          roughness: 0.55,
          side: THREE.DoubleSide,
          emissive: new THREE.Color(0x000000),
          emissiveIntensity: 0,
        });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.name = cam.id;
        mesh.matrixAutoUpdate = false;
        mesh.matrix.copy(buildCamMatrix(cam));
        mesh.visible = !!cam.enabled;
        group.add(mesh);
      }
      group.visible = !!show;
      group.updateMatrixWorld(true);
      t.cameraOverlays = group;
      t.scene.add(group);
      styleCamMaterials(group, cfg, selectedId);
    } catch (e) {
      console.warn("[camera_overlays]", e);
      message.warning("相机 STL 加载失败: " + (e.message || e));
    } finally {
      setCamOverlayBusy(false);
    }
  };

  const patchActiveCam = (patch) => {
    if (!camOverlayCfg || !activeCamId) return null;
    const cameras = (camOverlayCfg.cameras || []).map((c) =>
      c.id === activeCamId ? { ...c, ...patch } : c,
    );
    const next = { ...camOverlayCfg, cameras };
    setCamOverlayCfg(next);
    applyCamMatrices(next, activeCamId);
    return next;
  };

  const saveCamOverlayCfg = async (next) => {
    try {
      const saved = await api("/api/camera_overlays", {
        method: "PUT",
        body: JSON.stringify(next),
      });
      setCamOverlayCfg(saved);
      applyCamMatrices(saved, activeCamId);
      message.success("相机叠加已保存（未改 URDF）");
      return saved;
    } catch (e) {
      message.error(String(e.message || e));
      return null;
    }
  };

  /** Apply extrinsics.d435.T (T_base_cam) onto the active / first D435 CAD overlay. */
  const syncD435FromExtrinsics = async ({ persist = true } = {}) => {
    try {
      const ext = await api("/api/extrinsics");
      const T = ext?.d435?.T;
      if (!Array.isArray(T) || T.length < 4) {
        message.warning("extrinsics.d435.T 无效");
        return null;
      }
      const pose = poseFromMat4(T);
      const overlayId = ext?.d435?.source?.camera_overlay_id || "d435_a";
      const cfg = camOverlayCfg || { enabled: true, cameras: [] };
      let cameras = Array.isArray(cfg.cameras) ? [...cfg.cameras] : [];
      let targetId = overlayId;
      if (!cameras.some((c) => c.id === targetId)) {
        const hit = cameras.find((c) => c.model === "d435" || String(c.id || "").startsWith("d435"));
        targetId = hit?.id || activeCamId || cameras[0]?.id;
      }
      if (!targetId) {
        message.warning("无 D435 相机叠加项");
        return null;
      }
      cameras = cameras.map((c) => {
        if (c.id !== targetId) return c;
        return {
          ...c,
          enabled: true,
          offset_m: pose.offset_m.map((v) => Number(v.toFixed(8))),
          offset_rpy_deg: pose.offset_rpy_deg.map((v) => Number(v.toFixed(8))),
          align_method: "from_extrinsics_d435_T",
        };
      });
      const next = { ...cfg, enabled: true, cameras };
      setCamOverlayCfg(next);
      setActiveCamId(targetId);
      camOverlayVisibleRef.current = true;
      setCamOverlayVisible(true);
      if (!threeRef.current?.cameraOverlays) {
        await loadCameraOverlays(next, { selectedId: targetId, visible: true });
      } else {
        applyCamMatrices(next, targetId, { visible: true });
      }
      if (persist) {
        await saveCamOverlayCfg(next);
        message.success(`已按 extrinsics.d435 放置 ${targetId}`);
      } else {
        message.success(`已预览 ${targetId}（未保存）`);
      }
      return next;
    } catch (e) {
      message.error(String(e.message || e));
      return null;
    }
  };

  const activeCam = (camOverlayCfg?.cameras || []).find((c) => c.id === activeCamId) || null;

  const buildShapeMatrix = (shape) => {
    const off = shape.offset_m || [0, 0, 0];
    const orpy = (shape.offset_rpy_deg || [0, 0, 0]).map((d) => (Number(d) * Math.PI) / 180);
    const Roff = new THREE.Matrix4().makeRotationFromEuler(
      new THREE.Euler(orpy[0] || 0, orpy[1] || 0, orpy[2] || 0, "XYZ"),
    );
    const Toff = new THREE.Matrix4().makeTranslation(
      Number(off[0]) || 0,
      Number(off[1]) || 0,
      Number(off[2]) || 0,
    );
    const Rx = new THREE.Matrix4().makeRotationX(-Math.PI / 2);
    const m = new THREE.Matrix4();
    m.identity();
    m.multiply(Rx);
    m.multiply(Toff);
    m.multiply(Roff);
    return m;
  };

  const applyDrawShapes = (cfg, selectedId, opts = {}) => {
    const t = threeRef.current;
    if (!t?.drawShapes || !cfg) return;
    const root = t.drawShapes;
    const show = opts.visible !== undefined ? !!opts.visible : drawVisibleRef.current;
    root.visible = !!cfg.enabled && show;
    const sel = selectedId !== undefined ? selectedId : activeShapeId;
    const layers = cfg.layers || [];
    root.children.forEach((layerG) => {
      const L = layers.find((x) => x.id === layerG.name);
      if (!L) {
        layerG.visible = false;
        return;
      }
      layerG.visible = !!L.enabled;
      const shapes = L.shapes || [];
      layerG.children.forEach((mesh) => {
        const s = shapes.find((x) => x.id === mesh.name);
        if (!s) {
          mesh.visible = false;
          return;
        }
        mesh.visible = !!s.enabled;
        mesh.matrixAutoUpdate = false;
        mesh.matrix.copy(buildShapeMatrix(s));
        mesh.updateMatrixWorld(true);
        if (mesh.material) {
          const selected = sel && s.id === sel;
          const baseOp = Number(s.opacity ?? 0.55);
          mesh.material.opacity = selected ? Math.min(0.95, baseOp + 0.18) : baseOp;
          if (s.color) mesh.material.color.set(s.color);
          if (mesh.material.emissive) {
            mesh.material.emissive.set(selected ? s.color || "#ffffff" : "#000000");
            mesh.material.emissiveIntensity = selected ? 0.2 : 0;
          }
          mesh.material.needsUpdate = true;
        }
      });
    });
  };

  const loadDrawShapes = async (cfg, opts = {}) => {
    const t = threeRef.current;
    if (!t?.scene) return;
    if (t.drawShapes) {
      t.scene.remove(t.drawShapes);
      disposeObject3D(t.drawShapes);
      t.drawShapes = null;
    }
    if (!cfg?.enabled && opts.visible !== true) return;
    const layers = Array.isArray(cfg?.layers) ? cfg.layers : [];
    const show = opts.visible !== undefined ? opts.visible : drawVisible;
    const root = new THREE.Group();
    root.name = "draw_shapes";
    for (const L of layers) {
      const layerG = new THREE.Group();
      layerG.name = L.id;
      layerG.visible = !!L.enabled;
      for (const s of L.shapes || []) {
        const geom = makeShapeGeometry(s);
        const mat = new THREE.MeshStandardMaterial({
          color: new THREE.Color(s.color || "#5b8ff9"),
          transparent: true,
          opacity: Number(s.opacity ?? 0.55),
          metalness: 0.08,
          roughness: 0.65,
          side: THREE.DoubleSide,
          depthWrite: false,
          emissive: new THREE.Color(0x000000),
          emissiveIntensity: 0,
        });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.name = s.id;
        mesh.matrixAutoUpdate = false;
        mesh.matrix.copy(buildShapeMatrix(s));
        mesh.visible = !!s.enabled;
        mesh.renderOrder = 3;
        layerG.add(mesh);
      }
      root.add(layerG);
    }
    root.visible = !!show && !!cfg?.enabled;
    root.updateMatrixWorld(true);
    t.drawShapes = root;
    t.scene.add(root);
    applyDrawShapes({ ...cfg, enabled: true });
  };

  const activeDrawLayer =
    (drawCfg?.layers || []).find((L) => L.id === activeDrawLayerId)
    || (drawCfg?.layers || [])[0]
    || null;
  const activeShape =
    (activeDrawLayer?.shapes || []).find((s) => s.id === activeShapeId) || null;

  const patchDrawCfg = (mutator) => {
    if (!drawCfg) return null;
    const next = mutator({ ...drawCfg, layers: (drawCfg.layers || []).map((L) => ({
      ...L,
      shapes: (L.shapes || []).map((s) => ({ ...s })),
    })) });
    setDrawCfg(next);
    // Rebuild if shape type/size changed (geometry), else just matrices
    const needRebuild = next._rebuild;
    if (needRebuild) {
      delete next._rebuild;
      loadDrawShapes(next, { visible: drawVisible });
    } else {
      applyDrawShapes(next);
    }
    return next;
  };

  const patchActiveShape = (patch, opts = {}) => {
    if (!drawCfg || !activeDrawLayerId || !activeShapeId) return null;
    return patchDrawCfg((cfg) => {
      cfg.layers = (cfg.layers || []).map((L) => {
        if (L.id !== activeDrawLayerId) return L;
        return {
          ...L,
          shapes: (L.shapes || []).map((s) =>
            s.id === activeShapeId ? { ...s, ...patch } : s,
          ),
        };
      });
      if (opts.rebuild) cfg._rebuild = true;
      return cfg;
    });
  };

  const saveDrawCfg = async (next) => {
    try {
      const body = { ...(next || drawCfg) };
      delete body._rebuild;
      delete body.config_path;
      const saved = await api("/api/draw_shapes", {
        method: "PUT",
        body: JSON.stringify(body),
      });
      setDrawCfg(saved);
      if (saved.active_layer_id) setActiveDrawLayerId(saved.active_layer_id);
      await loadDrawShapes(saved, { visible: drawVisible });
      message.success("绘制图层已保存");
      return saved;
    } catch (e) {
      message.error(String(e.message || e));
      return null;
    }
  };

  const addDrawLayer = () => {
    const n = (drawCfg?.layers || []).length + 1;
    const layer = defaultDrawLayer(`绘制层 ${n}`);
    const next = {
      ...(drawCfg || { enabled: true, layers: [] }),
      enabled: true,
      layers: [...(drawCfg?.layers || []), layer],
      active_layer_id: layer.id,
    };
    setDrawCfg(next);
    setActiveDrawLayerId(layer.id);
    setActiveShapeId(null);
    loadDrawShapes(next, { visible: true });
    setDrawVisible(true);
  };

  const renameActiveDrawLayer = (label) => {
    if (!activeDrawLayerId) return;
    patchDrawCfg((cfg) => {
      cfg.layers = (cfg.layers || []).map((L) =>
        L.id === activeDrawLayerId ? { ...L, label: String(label || L.label) } : L,
      );
      return cfg;
    });
  };

  const addDrawShape = (type) => {
    const layerId = activeDrawLayerId || drawCfg?.layers?.[0]?.id;
    if (!layerId) {
      message.warning("请先新建绘制层");
      return;
    }
    const shape = defaultDrawShape(type || newShapeType);
    const next = patchDrawCfg((cfg) => {
      cfg.enabled = true;
      cfg.active_layer_id = layerId;
      cfg.layers = (cfg.layers || []).map((L) => {
        if (L.id !== layerId) return L;
        return { ...L, enabled: true, shapes: [...(L.shapes || []), shape] };
      });
      cfg._rebuild = true;
      return cfg;
    });
    setActiveDrawLayerId(layerId);
    setActiveShapeId(shape.id);
    setDrawVisible(true);
    return next;
  };

  const deleteActiveShape = () => {
    if (!activeDrawLayerId || !activeShapeId) return;
    const next = patchDrawCfg((cfg) => {
      cfg.layers = (cfg.layers || []).map((L) => {
        if (L.id !== activeDrawLayerId) return L;
        return {
          ...L,
          shapes: (L.shapes || []).filter((s) => s.id !== activeShapeId),
        };
      });
      cfg._rebuild = true;
      return cfg;
    });
    setActiveShapeId(null);
    return next;
  };

  const deleteActiveDrawLayer = () => {
    if (!activeDrawLayerId || !drawCfg) return null;
    const layers = (drawCfg.layers || []).filter((L) => L.id !== activeDrawLayerId);
    if (!layers.length) {
      message.warning("至少保留一个绘制层");
      return null;
    }
    const nextId = layers[0].id;
    const next = {
      ...drawCfg,
      layers: layers.map((L) => ({
        ...L,
        shapes: (L.shapes || []).map((s) => ({ ...s })),
      })),
      active_layer_id: nextId,
    };
    setDrawCfg(next);
    setActiveDrawLayerId(nextId);
    setActiveShapeId((layers[0].shapes || [])[0]?.id || null);
    loadDrawShapes(next, { visible: drawVisible });
    return next;
  };


  const loadUrdf = (arm) => {
    const t = threeRef.current;
    if (!t.scene || !arm || !arm.urdf) {
      setHint("场景或型号不可用");
      return;
    }
    if (t.robot) {
      t.scene.remove(t.robot);
      t.robot = null;
    }
    if (t.ghost) {
      t.scene.remove(t.ghost);
      t.ghost = null;
    }
    const urdfUrl = String(arm.urdf);
    const pathOnly = urdfUrl.split("?")[0];
    const workingPath = pathOnly.replace(/[^/]+$/, "");
    setHint((armPreload.ready ? "使用预缓存加载 " : "加载 ") + pathOnly + " …");

    // Shared across live+ghost loads: parse STL once, reuse BufferGeometry.
    // Bust geometry cache when URDF/EOAT version changes (mesh files keep same names).
    const meshVer = (() => {
      const q = urdfUrl.split("?")[1] || "";
      const m = /(?:^|&)v=([^&]+)/.exec(q);
      return (m && m[1]) || "1";
    })();
    if (!t.meshGeomCache || t.meshGeomCacheVer !== meshVer) {
      t.meshGeomCache = Object.create(null);
      t.meshGeomWaiters = Object.create(null);
      t.meshGeomCacheVer = meshVer;
    }
    if (!t.meshGeomWaiters) t.meshGeomWaiters = Object.create(null);
    const geomCache = t.meshGeomCache;
    const geomWaiters = t.meshGeomWaiters;

    const resolveMeshUrl = (url) => {
      try {
        const u = String(url);
        const rel = u.includes("/meshes/") ? ("meshes/" + u.split("/meshes/").pop()) : null;
        // Also try basename for case variants ( .STL vs listed path ).
        const baseName = u.split("/").pop();
        const relLower = rel ? `meshes/${baseName}` : null;
        if (rel && armPreload.meshBlobs[rel]) return armPreload.meshBlobs[rel];
        if (relLower && armPreload.meshBlobs[relLower]) return armPreload.meshBlobs[relLower];
        if (armPreload.meshBlobs[u]) return armPreload.meshBlobs[u];
        // Case-insensitive match against preload keys (URDF often uses .STL).
        if (baseName) {
          const hit = Object.keys(armPreload.meshBlobs).find((k) => k.split("/").pop()?.toLowerCase() === baseName.toLowerCase());
          if (hit) return armPreload.meshBlobs[hit];
        }
      } catch (_) {}
      return url;
    };

    const loadSharedGeometry = (path, mgr) => new Promise((resolve, reject) => {
      const key = String(path);
      if (geomCache[key]) {
        resolve(geomCache[key]);
        return;
      }
      if (!geomWaiters[key]) {
        geomWaiters[key] = [];
        const stl = new STLLoader(mgr);
        stl.load(
          path,
          (geom) => {
            // Mark shared so overlay-style dispose helpers skip it.
            geom.__rsShared = true;
            geomCache[key] = geom;
            const wait = geomWaiters[key] || [];
            delete geomWaiters[key];
            wait.forEach((fn) => fn(geom));
          },
          undefined,
          (err) => {
            const wait = geomWaiters[key] || [];
            delete geomWaiters[key];
            wait.forEach((fn) => {
              try { fn(null); } catch (_) { /* ignore */ }
            });
            reject(err);
          },
        );
      }
      geomWaiters[key].push((geom) => {
        if (geom) resolve(geom);
        else reject(new Error(`STL load failed: ${key}`));
      });
    });

    // One manager for both instances so URL blob cache + STL parse are shared.
    const manager = new THREE.LoadingManager();
    if (typeof manager.setURLModifier === "function") {
      manager.setURLModifier(resolveMeshUrl);
    }

    const liveMat = new THREE.MeshStandardMaterial({
      color: 0xb0b8c4,
      metalness: 0.25,
      roughness: 0.55,
    });
    const ghostMat = getGhostMaterial();

    const makeLoader = (asGhost) => {
      const loader = new URDFLoader(manager);
      loader.workingPath = workingPath;
      // npm urdf-loader 0.12+: (path, manager, done). Older vendor used 4 args with material.
      loader.loadMeshCb = (path, mgr, done) => {
        if (typeof done !== "function") {
          console.error("[arm] loadMeshCb: done is not a function", { path, done });
          return;
        }
        loadSharedGeometry(path, mgr)
          .then((geom) => {
            const material = asGhost ? ghostMat : liveMat;
            const mesh = new THREE.Mesh(geom, material);
            if (asGhost) {
              mesh.renderOrder = 2;
              mesh.userData.rsGhost = true;
            }
            // Geometry is shared — do not dispose on mesh teardown of one instance.
            mesh.userData.rsSharedGeom = true;
            done(mesh);
          })
          .catch((err) => done(null, err));
      };
      return loader;
    };

    const token = ++t.loadToken;
    const finishRobot = (r, asGhost) => {
      if (token !== t.loadToken) return;
      mountUrdfInstance(arm, r, asGhost);
      syncGripperOpenFromRobot(r);
      if (asGhost) {
        t.ghost = r;
        t.scene.add(r);
        styleGhostMaterials(r);
        const pending = armGhostTarget.current;
        if (pending && Array.isArray(pending.joints_rad) && pending.joints_rad.length >= 6) {
          applyGhostTarget(pending);
        } else {
          paintGhost(previewJointsRef.current.length ? previewJointsRef.current : liveJointsRef.current, arm);
          r.visible = !!ghostVisibleRef.current;
        }
      } else {
        t.robot = r;
        r.visible = !!robotVisibleRef.current;
        t.scene.add(r);
        paintLive(liveJointsRef.current, arm);
        setHint((arm.label || arm.id) + " · " + arm.model + " · 拖拽旋转 · Ctrl+↑↓ 以鼠标为中心缩放");
      }
      applyGripperOpening();
    };

    // After all STLs for both arms settle, ensure ghost mats stuck (async attach race).
    manager.onLoad = () => {
      if (token !== t.loadToken) return;
      if (t.ghost) styleGhostMaterials(t.ghost);
    };

    const runLoad = (asGhost) => {
      const loader = makeLoader(asGhost);
      const onOk = (r) => finishRobot(r, asGhost);
      const onErr = (err) => {
        if (token !== t.loadToken) return;
        if (!asGhost) {
          setHint("URDF 失败: " + (err && err.message ? err.message : String(err)));
        }
      };
      // Prefer prefetched URDF text (avoids query-string / cache quirks)
      const cached =
        armPreload.urdfText
        && armPreload.urdfUrl
        && String(armPreload.urdfUrl).split("?")[0] === pathOnly;
      try {
        if (cached) {
          onOk(loader.parse(armPreload.urdfText, workingPath));
        } else {
          loader.load(urdfUrl, onOk, undefined, onErr);
        }
      } catch (e) {
        onErr(e);
      }
    };

    runLoad(false);
    // Second URDF instance = joint preview; reuses STL geometry via geomCache.
    runLoad(true);
  };

  // Init Three once when tab active — deps ONLY [active] to avoid teardown loops
  useEffect(() => {
    if (!active) return undefined;
    let cancelled = false;
    let alive = false;
    let ro = null;
    let poll = null;
    let renderer = null;
    let host = null;

    const boot = async () => {
      // wait one frame so ref is attached (panel just mounted)
      await new Promise((r) => requestAnimationFrame(() => r()));
      if (cancelled) return;
      host = hostRef.current;
      if (!host) {
        setHint("画布节点未挂载");
        return;
      }
      setHint("初始化 3D 场景…");
      try {
        setPreloadPct(armPreload.ready ? 100 : 30);
        // Force re-fetch STLs — EOAT mesh names are stable; blob/HTTP cache otherwise sticks.
        await preloadArmAssets({ force: true });
        setPreloadPct(100);
      } catch (e) {
        setHint("预加载失败: " + (e.message || e) + "（仍尝试直接加载）");
      }
      if (cancelled) return;

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x070c14);
      const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 20);
      camera.position.set(1.2, 0.9, 1.2);
      renderer = new THREE.WebGLRenderer({ antialias: true });
      host.innerHTML = "";
      host.appendChild(renderer.domElement);
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.target.set(0.3, 0.3, 0);
      controls.enableZoom = true;
      controls.zoomToCursor = true; // wheel + our Ctrl+↑↓ use cursor as zoom center
      controls.minDistance = 0.05;
      controls.maxDistance = 40;
      scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 0.85));
      const dir = new THREE.DirectionalLight(0xffffff, 0.7);
      dir.position.set(2, 3, 1);
      scene.add(dir);
      scene.add(new THREE.GridHelper(2, 20, 0x3dd6c6, 0x1a2738));
      const tcpMarker = new THREE.Mesh(
        new THREE.SphereGeometry(0.018, 24, 16),
        new THREE.MeshStandardMaterial({ color: 0xf0b429, emissive: 0x553300 })
      );
      scene.add(tcpMarker);
      const markersGroup = new THREE.Group();
      markersGroup.name = "m2_waypoints";
      markersGroup.visible = !!markersVisibleRef.current;
      tcpMarker.visible = !!tcpVisibleRef.current;
      scene.add(markersGroup);
      threeRef.current = {
        scene, camera, renderer, controls, tcpMarker, markersGroup,
        robot: null, ghost: null, sceneOverlay: null, cameraOverlays: null,
        drawShapes: null, loadToken: 0, raf: 0,
      };
      syncSceneMarkers(armSceneMarkers.items);

      // Ctrl+↑/↓ → synthesize wheel at pointer (uses OrbitControls zoomToCursor; avoids freeze)
      const pointer = { x: 0, y: 0, inside: false };
      let zoomRaf = 0;
      let zoomDeltaY = 0;
      const onPointerMove = (e) => {
        pointer.x = e.clientX;
        pointer.y = e.clientY;
        pointer.inside = true;
      };
      const onPointerLeave = () => {
        pointer.inside = false;
      };
      const flushKeyZoom = () => {
        zoomRaf = 0;
        if (!alive || !zoomDeltaY) return;
        const el = renderer.domElement;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        if (rect.width < 1 || rect.height < 1) {
          zoomDeltaY = 0;
          return;
        }
        let cx = pointer.inside ? pointer.x : rect.left + rect.width * 0.5;
        let cy = pointer.inside ? pointer.y : rect.top + rect.height * 0.5;
        cx = Math.min(rect.right, Math.max(rect.left, cx));
        cy = Math.min(rect.bottom, Math.max(rect.top, cy));
        const dy = zoomDeltaY;
        zoomDeltaY = 0;
        el.dispatchEvent(new WheelEvent("wheel", {
          bubbles: true,
          cancelable: true,
          clientX: cx,
          clientY: cy,
          deltaY: dy,
          deltaMode: 0,
        }));
      };
      const onKeyDownZoom = (e) => {
        if (!(e.ctrlKey || e.metaKey)) return;
        if (e.code !== "ArrowUp" && e.code !== "ArrowDown") return;
        const tag = (e.target && e.target.tagName) || "";
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        if (e.target && e.target.isContentEditable) return;
        e.preventDefault();
        // ArrowUp → negative deltaY → zoom in (same as wheel)
        zoomDeltaY += e.code === "ArrowUp" ? -80 : 80;
        // coalesce key-repeat to one wheel event per frame
        if (!zoomRaf) zoomRaf = requestAnimationFrame(flushKeyZoom);
      };
      renderer.domElement.addEventListener("pointermove", onPointerMove);
      renderer.domElement.addEventListener("pointerleave", onPointerLeave);
      window.addEventListener("keydown", onKeyDownZoom);
      threeRef.current._zoomCleanup = () => {
        if (zoomRaf) cancelAnimationFrame(zoomRaf);
        zoomRaf = 0;
        renderer.domElement.removeEventListener("pointermove", onPointerMove);
        renderer.domElement.removeEventListener("pointerleave", onPointerLeave);
        window.removeEventListener("keydown", onKeyDownZoom);
      };

      const resize = () => {
        if (!host) return;
        const w = host.clientWidth, h = host.clientHeight;
        camera.aspect = w / Math.max(h, 1);
        camera.updateProjectionMatrix();
        renderer.setSize(w, h, false);
      };
      resize();
      ro = new ResizeObserver(resize);
      ro.observe(host);
      alive = true;
      (function loop() {
        if (!alive) return;
        threeRef.current.raf = requestAnimationFrame(loop);
        controls.update();
        renderer.render(scene, camera);
      })();

      try {
        const st = armPreload.model
          ? { ...await api("/api/arm/state"), model: armPreload.model }
          : await api("/api/arm/state");
        // refresh full state
        const live = await api("/api/arm/state");
        if (cancelled) return;
        applyStatus(live);
        try {
          await refreshEoatCatalog();
        } catch (eoatErr) {
          console.warn("[eoat] catalog", eoatErr);
        }
        loadUrdf(live.model || armPreload.model);
        try {
          const ov = await api("/api/scene_overlay");
          if (cancelled) return;
          setOverlayCfg(ov);
          setOverlayVisible(!!ov.enabled);
          await loadSceneOverlay(ov);
        } catch (ovErr) {
          console.warn("[scene_overlay] config", ovErr);
        }
        try {
          const cov = await api("/api/camera_overlays");
          if (cancelled) return;
          setCamOverlayCfg(cov);
          camOverlayVisibleRef.current = !!cov.enabled;
          setCamOverlayVisible(!!cov.enabled);
          const firstId = (cov.cameras || []).find((c) => c.enabled)?.id
            || (cov.cameras || [])[0]?.id
            || null;
          setActiveCamId(firstId);
          await loadCameraOverlays(cov, { selectedId: firstId, visible: !!cov.enabled });
        } catch (camErr) {
          console.warn("[camera_overlays] config", camErr);
        }
        try {
          const dcfg = await api("/api/draw_shapes");
          if (cancelled) return;
          setDrawCfg(dcfg);
          drawVisibleRef.current = !!dcfg.enabled;
          setDrawVisible(!!dcfg.enabled);
          const lid = dcfg.active_layer_id
            || (dcfg.layers || []).find((L) => L.enabled)?.id
            || (dcfg.layers || [])[0]?.id
            || null;
          setActiveDrawLayerId(lid);
          const layer = (dcfg.layers || []).find((L) => L.id === lid);
          const sid = (layer?.shapes || []).find((s) => s.enabled)?.id
            || (layer?.shapes || [])[0]?.id
            || null;
          setActiveShapeId(sid);
          await loadDrawShapes(dcfg, { visible: !!dcfg.enabled });
        } catch (drawErr) {
          console.warn("[draw_shapes] config", drawErr);
        }
      } catch (e) {
        if (!cancelled) setHint("启动失败: " + (e.message || e));
      }

      poll = setInterval(() => {
        api("/api/arm/state").then((st) => {
          if (!cancelled) applyStatus(st, { syncSliders: false });
        }).catch(() => {});
      }, 250);
    };

    boot();
    const unsubMarkers = armSceneMarkers.subscribe((items) => {
      if (!cancelled) syncSceneMarkers(items);
    });
    const unsubGhost = armGhostTarget.subscribe((payload) => {
      if (!cancelled && payload) applyGhostTarget(payload);
    });
    if (armGhostTarget.current) applyGhostTarget(armGhostTarget.current);

    return () => {
      cancelled = true;
      alive = false;
      if (typeof unsubMarkers === "function") unsubMarkers();
      if (typeof unsubGhost === "function") unsubGhost();
      if (poll) clearInterval(poll);
      if (ro) ro.disconnect();
      cancelAnimationFrame((threeRef.current && threeRef.current.raf) || 0);
      if (typeof threeRef.current?._zoomCleanup === "function") {
        try { threeRef.current._zoomCleanup(); } catch (_) { /* ignore */ }
      }
      if (renderer) renderer.dispose();
      if (host) host.innerHTML = "";
      threeRef.current = {};
    };
  }, [active]);

  const run = async (path, body) => {
    try {
      const st = await api(path, {
        method: "POST",
        body: body === undefined ? "{}" : JSON.stringify(body),
      });
      applyStatus(st);
    } catch (e) { message.error(String(e.message || e)); }
  };

  const onJointChange = (i, v) => {
    const mdl = modelRef.current || model || status?.model;
    const lo = jointSoftLimit(mdl, i, "min");
    const hi = jointSoftLimit(mdl, i, "max");
    const next = joints.slice();
    next[i] = Math.min(hi, Math.max(lo, Number(v)));
    previewJointsRef.current = next;
    setJoints(next);
    // Slider only drives transparent ghost — live mesh stays on measured pose.
    if (!ghostVisibleRef.current) setGhostShown(true);
    paintGhost(next);
  };

  const syncSlidersFromLive = () => {
    const live = liveJointsRef.current || status?.joints_rad;
    if (!live || live.length < 6) {
      message.warning("无当前实测关节");
      return;
    }
    previewJointsRef.current = live.slice();
    setJoints(live.slice());
    paintGhost(live);
    message.success("滑条已同步到当前臂姿");
  };

  const pose = status?.cartesian_xyzrpy;
  const poseLabels = ["x", "y", "z", "rx", "ry", "rz"];

  const doCopyJoints = async () => {
    try {
      const csv = fmtCsv6(status?.joints_rad || joints);
      if (!csv) throw new Error("无当前 joints（请先 Connect）");
      await copyText(csv);
      setDispatchMode("joints");
      setDispatchText(csv);
      setCopyHint("已复制 joints");
      message.success("已复制 joints");
    } catch (e) { message.error(String(e.message || e)); }
  };
  const doCopyPose = async () => {
    try {
      const csv = fmtCsv6(status?.cartesian_xyzrpy);
      if (!csv) throw new Error("无当前 pose（请先 Connect，需 FK）");
      await copyText(csv);
      setDispatchMode("pose");
      setDispatchText(csv);
      setCopyHint("已复制 pose");
      message.success("已复制 pose");
    } catch (e) { message.error(String(e.message || e)); }
  };
  const doDispatch = async () => {
    const parsed = parseCsv6(dispatchText);
    if (!parsed.ok) { message.warning(parsed.error); return; }
    if (dispatchMode === "pose") {
      await run("/api/arm/move_pose", { xyzrpy: parsed.values });
    } else {
      await run("/api/arm/command", { joints_rad: parsed.values });
    }
    message.success(dispatchMode === "pose" ? "已下发 pose" : "已下发 joints");
  };
  const doSetHome = async () => {
    try {
      const st = await api("/api/arm/home/set", { method: "POST", body: "{}" });
      applyStatus(st);
      message.success(st.message || "已将当前关节设为 HOME（已写入配置）");
    } catch (e) { message.error(String(e.message || e)); }
  };
  const doSetDryRun = async (wantDry) => {
    try {
      if (status?.connected) {
        await run("/api/arm/disconnect");
      }
      const st = await api("/api/arm/dry_run", {
        method: "PUT",
        body: JSON.stringify({ dry_run: !!wantDry, persist: true }),
      });
      applyStatus(st);
      message.success(wantDry
        ? "已切回 dry-run（仿真）"
        : "已切到 LIVE（arm_cs_write）。示教器需 Remote + ExternalControl，再 Connect");
    } catch (e) { message.error(String(e.message || e)); }
  };

  // Keep DOM mounted while active so hostRef exists before effect
  return (
    <div className="arm-layout" style={{ display: active ? "grid" : "none" }}>
      <div className="arm-side">
        <Card
          size="small"
          title={
            <Space size={8} wrap>
              <span>Cell C · CS66</span>
              <Tag color={status?.connected ? "green" : "default"}>
                {status?.dry_run ? "DRY-RUN" : "LIVE"}
              </Tag>
              <Tag color={status?.armed ? "blue" : "default"}>
                {status?.armed ? "ARMED" : "DISARMED"}
              </Tag>
            </Space>
          }
          extra={
            <Text type="secondary" style={{ fontSize: 12 }}>
              {preloadPct >= 100 ? "模型就绪" : `预加载 ${preloadPct}%`}
            </Text>
          }
        >
          <div className="arm-section-title">模式</div>
          <div className="arm-toolbar" style={{ marginBottom: 12, alignItems: "center" }}>
            <Switch
              checked={!status?.dry_run}
              checkedChildren="LIVE"
              unCheckedChildren="DRY"
              onChange={(live) => doSetDryRun(!live)}
            />
            <Text type="secondary" style={{ fontSize: 12 }}>
              {status?.dry_run
                ? "仿真（sensors dry_run）"
                : `LIVE · sensors arm_cs_write · ${model?.robot_ip || status?.model?.robot_ip || "192.168.1.200"}`}
            </Text>
          </div>
          <div className="arm-section-title">末端工装 EOAT（URDF）</div>
          <div className="arm-toolbar" style={{ marginBottom: 12, alignItems: "center" }}>
            <Select
              style={{ minWidth: 240 }}
              loading={eoatBusy}
              value={eoatId}
              options={(eoatCatalog.length
                ? eoatCatalog
                : [
                    { id: "cellc", label: "cellc · EFG-C65 + D405" },
                    { id: "cellc_m", label: "cellc_m · digital-model/cellc_m" },
                  ]
              ).map((e) => ({
                value: e.id,
                label: e.label || e.id,
              }))}
              onChange={(v) => switchEoat(v)}
            />
            <Text type="secondary" style={{ fontSize: 12 }}>
              {model?.urdf ? String(model.urdf).split("/").pop() : "—"}
            </Text>
          </div>
          <div className="arm-grip-open-block" style={{ marginBottom: 12 }}>
            <div className="arm-section-title">夹爪开度（可视化）</div>
            <div className="arm-joint-row arm-grip-open-row">
              <span className="lab">闭</span>
              <Slider
                style={{ margin: 0 }}
                disabled={!gripOpenCapable}
                min={0}
                max={Math.max(0.1, (Number(gripOpenHiM) || GRIP_OPEN_DEFAULT_HI_M) * 1000)}
                step={0.5}
                value={Math.max(0, Math.min(
                  (Number(gripOpenHiM) || GRIP_OPEN_DEFAULT_HI_M) * 1000,
                  (Number(gripOpenM) || 0) * 1000,
                ))}
                onChange={onGripOpenMm}
              />
              <span className="lab">开</span>
              <span className="arm-joint-val">
                {((Number(gripOpenM) || 0) * 1000).toFixed(1)} mm
              </span>
            </div>
            <Text type="secondary" style={{ display: "block", fontSize: 12, marginTop: 4 }}>
              {gripOpenCapable
                ? `单侧 0…${((Number(gripOpenHiM) || GRIP_OPEN_DEFAULT_HI_M) * 1000).toFixed(0)} mm（总开度约 ×2）；仅可视化。`
                : "等待 URDF 加载 left_finger_joint（cellc / cellc_m）…"}
            </Text>
          </div>
          <div className="arm-section-title">场景 STL（独立于 URDF）</div>
          <div className="arm-toolbar" style={{ marginBottom: 8, alignItems: "center" }}>
            <Switch
              checked={!!overlayCfg?.enabled && overlayVisible}
              loading={overlayBusy}
              checkedChildren="显示"
              unCheckedChildren="隐藏"
              onChange={async (on) => {
                setOverlayVisible(on);
                const next = { ...(overlayCfg || {}), enabled: on };
                setOverlayCfg(next);
                if (on && !threeRef.current?.sceneOverlay) {
                  await loadSceneOverlay(next);
                } else if (threeRef.current?.sceneOverlay) {
                  threeRef.current.sceneOverlay.visible = on;
                }
                await saveOverlayCfg(next);
              }}
            />
            <Text type="secondary" style={{ fontSize: 12 }}>
              {overlayCfg?.align?.confidence
                ? `底座对齐置信度 ${overlayCfg.align.confidence}`
                : "g26562-01-00 场景叠加"}
            </Text>
          </div>
          {!!overlayCfg && (
            <div style={{ marginBottom: 12 }}>
              <Text type="secondary" style={{ fontSize: 12 }}>
                微调 XYZ (m) + Roll/Pitch/Yaw (°) · ± 按 step 步进 · 保存写入 config/scene_overlay.yaml
              </Text>
              <StepControls
                xyzStep={sceneXyzStep}
                rpyStep={sceneRpyStep}
                onXyzStep={setSceneXyzStep}
                onRpyStep={setSceneRpyStep}
              />
              {[
                { lab: "X", min: -0.5, max: 0.5, kind: "xyz", idx: 0 },
                { lab: "Y", min: -0.5, max: 0.5, kind: "xyz", idx: 1 },
                { lab: "Z", min: -1.5, max: 1.5, kind: "xyz", idx: 2 },
                { lab: "Roll°", min: -180, max: 180, kind: "rpy", idx: 0 },
                { lab: "Pitch°", min: -180, max: 180, kind: "rpy", idx: 1 },
                { lab: "Yaw°", min: -180, max: 180, kind: "yaw", idx: 0 },
              ].map((row, ri) => {
                const isXyz = row.kind === "xyz";
                const step = isXyz ? sceneXyzStep : sceneRpyStep;
                const digits = isXyz ? Math.min(6, Math.max(3, Math.ceil(-Math.log10(sceneXyzStep || 0.001)))) : Math.min(3, Math.max(1, Math.ceil(-Math.log10(sceneRpyStep || 0.5))));
                let value = 0;
                if (isXyz) value = Number(overlayCfg.offset_m?.[row.idx] || 0);
                else if (row.kind === "rpy") value = Number(overlayCfg.offset_rpy_deg?.[row.idx] || 0);
                else value = Number(overlayCfg.align?.yaw_deg || 0);
                return (
                  <NudgeSliderRow
                    key={row.lab}
                    label={row.lab}
                    min={row.min}
                    max={row.max}
                    step={step}
                    digits={digits}
                    value={value}
                    style={ri === 0 ? { marginTop: 2 } : undefined}
                    onChange={(v) => {
                      if (isXyz) {
                        const offset_m = [...(overlayCfg.offset_m || [0, 0, 0])];
                        offset_m[row.idx] = v;
                        const next = { ...overlayCfg, offset_m };
                        setOverlayCfg(next);
                        applyOverlayMatrix(next);
                      } else if (row.kind === "rpy") {
                        const offset_rpy_deg = [...(overlayCfg.offset_rpy_deg || [0, 0, 0])];
                        offset_rpy_deg[row.idx] = v;
                        const next = { ...overlayCfg, offset_rpy_deg };
                        setOverlayCfg(next);
                        applyOverlayMatrix(next);
                      } else {
                        const align = { ...(overlayCfg.align || {}), yaw_deg: v };
                        const next = { ...overlayCfg, align };
                        setOverlayCfg(next);
                        applyOverlayMatrix(next);
                      }
                    }}
                  />
                );
              })}
              <Button
                size="small"
                type="primary"
                style={{ marginTop: 6 }}
                onClick={() => saveOverlayCfg(overlayCfg)}
              >
                保存场景对齐
              </Button>
            </div>
          )}
          <Divider style={{ margin: "10px 0" }} />
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
            <Switch
              size="small"
              checked={!!camOverlayCfg?.enabled && camOverlayVisible}
              loading={camOverlayBusy}
              onChange={async (on) => {
                camOverlayVisibleRef.current = on;
                setCamOverlayVisible(on);
                const next = { ...(camOverlayCfg || {}), enabled: on };
                setCamOverlayCfg(next);
                if (on && !threeRef.current?.cameraOverlays) {
                  await loadCameraOverlays(next, { selectedId: activeCamId, visible: true });
                } else if (threeRef.current?.cameraOverlays) {
                  applyCamMatrices(next, activeCamId, { visible: on });
                }
                await saveCamOverlayCfg(next);
              }}
            />
            <Text type="secondary" style={{ fontSize: 12 }}>
              相机 CAD 叠加（颜色区分 · 逐个微调）
            </Text>
          </div>
          {!!camOverlayCfg && (
            <div style={{ marginBottom: 12 }}>
              <Select
                size="small"
                style={{ width: "100%", marginBottom: 6 }}
                value={activeCamId}
                options={(camOverlayCfg.cameras || []).map((c) => ({
                  value: c.id,
                  label: (
                    <span>
                      <span
                        style={{
                          display: "inline-block",
                          width: 10,
                          height: 10,
                          borderRadius: 2,
                          background: c.color || "#888",
                          marginRight: 8,
                          verticalAlign: "middle",
                        }}
                      />
                      {c.label || c.id}
                      {!c.enabled ? "（关）" : ""}
                    </span>
                  ),
                }))}
                onChange={(id) => {
                  setActiveCamId(id);
                  applyCamMatrices(camOverlayCfg, id);
                }}
              />
              {!!activeCam && (
                <>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                    <Switch
                      size="small"
                      checked={!!activeCam.enabled}
                      onChange={(on) => patchActiveCam({ enabled: on })}
                    />
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      显示此相机 · seed med {activeCam.seed_med_mm != null ? `${activeCam.seed_med_mm} mm` : "—"}
                    </Text>
                  </div>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    微调 XYZ (m) + RPY (°) · ± 按 step 步进 · base 系 · 保存写入 config/camera_overlays.yaml
                  </Text>
                  <StepControls
                    xyzStep={camXyzStep}
                    rpyStep={camRpyStep}
                    onXyzStep={setCamXyzStep}
                    onRpyStep={setCamRpyStep}
                  />
                  {[
                    { lab: "X", min: -2, max: 2, kind: "xyz", idx: 0 },
                    { lab: "Y", min: -2, max: 2, kind: "xyz", idx: 1 },
                    { lab: "Z", min: -2, max: 2, kind: "xyz", idx: 2 },
                    { lab: "Roll°", min: -180, max: 180, kind: "rpy", idx: 0 },
                    { lab: "Pitch°", min: -180, max: 180, kind: "rpy", idx: 1 },
                    { lab: "Yaw°", min: -180, max: 180, kind: "rpy", idx: 2 },
                  ].map((row, ri) => {
                    const isXyz = row.kind === "xyz";
                    const step = isXyz ? camXyzStep : camRpyStep;
                    const digits = isXyz
                      ? Math.min(6, Math.max(3, Math.ceil(-Math.log10(camXyzStep || 0.001))))
                      : Math.min(3, Math.max(1, Math.ceil(-Math.log10(camRpyStep || 0.5))));
                    const value = isXyz
                      ? Number(activeCam.offset_m?.[row.idx] || 0)
                      : Number(activeCam.offset_rpy_deg?.[row.idx] || 0);
                    return (
                      <NudgeSliderRow
                        key={row.lab}
                        label={row.lab}
                        min={row.min}
                        max={row.max}
                        step={step}
                        digits={digits}
                        value={value}
                        style={ri === 0 ? { marginTop: 2 } : undefined}
                        onChange={(v) => {
                          if (isXyz) {
                            const offset_m = [...(activeCam.offset_m || [0, 0, 0])];
                            offset_m[row.idx] = v;
                            patchActiveCam({ offset_m });
                          } else {
                            const offset_rpy_deg = [...(activeCam.offset_rpy_deg || [0, 0, 0])];
                            offset_rpy_deg[row.idx] = v;
                            patchActiveCam({ offset_rpy_deg });
                          }
                        }}
                      />
                    );
                  })}
                  <Space size={8} style={{ marginTop: 6 }} wrap>
                    <Button
                      size="small"
                      type="primary"
                      onClick={() => saveCamOverlayCfg(camOverlayCfg)}
                    >
                      保存相机对齐
                    </Button>
                    {(activeCam?.model === "d435" || String(activeCam?.id || "").startsWith("d435")) && (
                      <Button
                        size="small"
                        loading={camOverlayBusy}
                        onClick={() => syncD435FromExtrinsics({ persist: true })}
                      >
                        从 D435 外参同步
                      </Button>
                    )}
                  </Space>
                </>
              )}
            </div>
          )}
          <Divider style={{ margin: "10px 0" }} />
          <div className="arm-section-title">绘制（长方体 / 球 / 圆柱）</div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
            <Switch
              size="small"
              checked={!!drawCfg?.enabled && drawVisible}
              onChange={(on) => setDrawGroupShown(on)}
            />
            <Text type="secondary" style={{ fontSize: 12 }}>
              独立绘制图层 · 可新建/编辑 · XYZ/RPY 微调
            </Text>
          </div>
          {!!drawCfg && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ display: "flex", gap: 6, marginBottom: 6, flexWrap: "wrap" }}>
                <Select
                  size="small"
                  style={{ flex: 1, minWidth: 120 }}
                  value={activeDrawLayer?.id}
                  options={(drawCfg.layers || []).map((L) => ({
                    value: L.id,
                    label: `${L.label || L.id}${(L.shapes || []).length ? ` (${L.shapes.length})` : ""}`,
                  }))}
                  onChange={(id) => {
                    setActiveDrawLayerId(id);
                    const L = (drawCfg.layers || []).find((x) => x.id === id);
                    const sid = (L?.shapes || [])[0]?.id || null;
                    setActiveShapeId(sid);
                    const next = { ...drawCfg, active_layer_id: id };
                    setDrawCfg(next);
                    applyDrawShapes(next);
                  }}
                />
                <Button size="small" onClick={addDrawLayer}>新建层</Button>
                <Button
                  size="small"
                  danger
                  disabled={!activeDrawLayer || (drawCfg.layers || []).length <= 1}
                  onClick={() => deleteActiveDrawLayer()}
                >
                  删除层
                </Button>
              </div>
              {!!activeDrawLayer && (
                <Input
                  size="small"
                  style={{ marginBottom: 8 }}
                  value={activeDrawLayer.label || ""}
                  placeholder="图层名称"
                  onChange={(e) => renameActiveDrawLayer(e.target.value)}
                />
              )}
              <div style={{ display: "flex", gap: 6, marginBottom: 8, flexWrap: "wrap", alignItems: "center" }}>
                <Select
                  size="small"
                  style={{ width: 100 }}
                  value={newShapeType}
                  onChange={setNewShapeType}
                  options={[
                    { value: "box", label: "长方体" },
                    { value: "sphere", label: "球体" },
                    { value: "cylinder", label: "圆柱" },
                  ]}
                />
                <Button size="small" type="primary" onClick={() => addDrawShape(newShapeType)}>
                  新建图元
                </Button>
                <Button
                  size="small"
                  danger
                  disabled={!activeShape}
                  onClick={() => deleteActiveShape()}
                >
                  删除图元
                </Button>
                <Button
                  size="small"
                  type="primary"
                  onClick={() => saveDrawCfg(drawCfg)}
                >
                  保存绘制
                </Button>
              </div>
              <Select
                size="small"
                style={{ width: "100%", marginBottom: 6 }}
                placeholder="选择图元"
                value={activeShapeId}
                options={(activeDrawLayer?.shapes || []).map((s) => ({
                  value: s.id,
                  label: (
                    <span>
                      <span
                        style={{
                          display: "inline-block",
                          width: 10,
                          height: 10,
                          borderRadius: 2,
                          background: s.color || "#888",
                          marginRight: 8,
                          verticalAlign: "middle",
                        }}
                      />
                      {s.label || s.id} · {s.type}
                    </span>
                  ),
                }))}
                onChange={(id) => {
                  setActiveShapeId(id);
                  applyDrawShapes(drawCfg, id);
                }}
              />
              {!!activeShape && (
                <>
                  <div style={{ display: "flex", gap: 8, marginBottom: 6, flexWrap: "wrap" }}>
                    <Switch
                      size="small"
                      checked={!!activeShape.enabled}
                      onChange={(on) => patchActiveShape({ enabled: on })}
                    />
                    <Text type="secondary" style={{ fontSize: 12 }}>显示</Text>
                    <Input
                      size="small"
                      style={{ width: 110 }}
                      value={activeShape.label || ""}
                      onChange={(e) => patchActiveShape({ label: e.target.value })}
                    />
                    <Input
                      size="small"
                      style={{ width: 92 }}
                      value={activeShape.color || "#5b8ff9"}
                      onChange={(e) => patchActiveShape({ color: e.target.value })}
                      addonBefore="色"
                    />
                  </div>
                  {activeShape.type === "box" && (
                    <div className="arm-step-row">
                      {["X", "Y", "Z"].map((lab, i) => (
                        <span className="arm-step-item" key={lab}>
                          <span>边{lab}</span>
                          <InputNumber
                            size="small"
                            min={0.001}
                            max={5}
                            step={0.001}
                            value={Number(activeShape.size_m?.[i] || 0.05)}
                            onChange={(v) => {
                              const size_m = [...(activeShape.size_m || [0.1, 0.08, 0.05])];
                              size_m[i] = Number(v) || 0.05;
                              patchActiveShape({ size_m }, { rebuild: true });
                            }}
                            style={{ width: 72 }}
                          />
                        </span>
                      ))}
                    </div>
                  )}
                  {activeShape.type === "sphere" && (
                    <div className="arm-step-row">
                      <span className="arm-step-item">
                        <span>半径 (m)</span>
                        <InputNumber
                          size="small"
                          min={0.001}
                          max={2}
                          step={0.001}
                          value={Number(activeShape.radius_m || 0.05)}
                          onChange={(v) => patchActiveShape({ radius_m: Number(v) || 0.05 }, { rebuild: true })}
                          style={{ width: 84 }}
                        />
                      </span>
                    </div>
                  )}
                  {activeShape.type === "cylinder" && (
                    <div className="arm-step-row">
                      <span className="arm-step-item">
                        <span>半径</span>
                        <InputNumber
                          size="small"
                          min={0.001}
                          max={2}
                          step={0.001}
                          value={Number(activeShape.radius_m || 0.04)}
                          onChange={(v) => patchActiveShape({ radius_m: Number(v) || 0.04 }, { rebuild: true })}
                          style={{ width: 72 }}
                        />
                      </span>
                      <span className="arm-step-item">
                        <span>高</span>
                        <InputNumber
                          size="small"
                          min={0.001}
                          max={5}
                          step={0.001}
                          value={Number(activeShape.height_m || 0.1)}
                          onChange={(v) => patchActiveShape({ height_m: Number(v) || 0.1 }, { rebuild: true })}
                          style={{ width: 72 }}
                        />
                      </span>
                    </div>
                  )}
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    微调 XYZ (m) + RPY (°) · base 系 · 点「保存绘制」写入 config/draw_shapes.yaml
                  </Text>
                  <StepControls
                    xyzStep={drawXyzStep}
                    rpyStep={drawRpyStep}
                    onXyzStep={setDrawXyzStep}
                    onRpyStep={setDrawRpyStep}
                  />
                  {[
                    { lab: "X", min: -2, max: 2, kind: "xyz", idx: 0 },
                    { lab: "Y", min: -2, max: 2, kind: "xyz", idx: 1 },
                    { lab: "Z", min: -2, max: 2, kind: "xyz", idx: 2 },
                    { lab: "Roll°", min: -180, max: 180, kind: "rpy", idx: 0 },
                    { lab: "Pitch°", min: -180, max: 180, kind: "rpy", idx: 1 },
                    { lab: "Yaw°", min: -180, max: 180, kind: "rpy", idx: 2 },
                  ].map((row, ri) => {
                    const isXyz = row.kind === "xyz";
                    const step = isXyz ? drawXyzStep : drawRpyStep;
                    const digits = isXyz
                      ? Math.min(6, Math.max(3, Math.ceil(-Math.log10(drawXyzStep || 0.001))))
                      : Math.min(3, Math.max(1, Math.ceil(-Math.log10(drawRpyStep || 0.5))));
                    const value = isXyz
                      ? Number(activeShape.offset_m?.[row.idx] || 0)
                      : Number(activeShape.offset_rpy_deg?.[row.idx] || 0);
                    return (
                      <NudgeSliderRow
                        key={row.lab}
                        label={row.lab}
                        min={row.min}
                        max={row.max}
                        step={step}
                        digits={digits}
                        value={value}
                        style={ri === 0 ? { marginTop: 2 } : undefined}
                        onChange={(v) => {
                          if (isXyz) {
                            const offset_m = [...(activeShape.offset_m || [0, 0, 0])];
                            offset_m[row.idx] = v;
                            patchActiveShape({ offset_m });
                          } else {
                            const offset_rpy_deg = [...(activeShape.offset_rpy_deg || [0, 0, 0])];
                            offset_rpy_deg[row.idx] = v;
                            patchActiveShape({ offset_rpy_deg });
                          }
                        }}
                      />
                    );
                  })}
                </>
              )}
            </div>
          )}
          <div className="arm-section-title">会话</div>
          <div className="arm-toolbar" style={{ marginBottom: 12 }}>
            <Button type="primary" onClick={() => run("/api/arm/connect")}>Connect</Button>
            <Button onClick={() => run("/api/arm/disconnect")}>Disconnect</Button>
            <Button
              onClick={() => run("/api/arm/arm")}
              disabled={!status?.connected || (!status?.dry_run && !status?.can_arm)}
            >
              Arm
            </Button>
            <Button onClick={() => run("/api/arm/disarm")} disabled={!status?.connected}>Disarm</Button>
            <Button danger onClick={() => run("/api/arm/stop")}>Stop</Button>
          </div>
          <div className="arm-toolbar">
            <Button onClick={() => run("/api/arm/home")} disabled={!status?.armed}>Home</Button>
            <Button onClick={doSetHome} disabled={!status?.connected}>设为 HOME</Button>
            <Button onClick={() => api("/api/arm/state").then((st) => applyStatus(st)).catch((e) => message.error(String(e.message || e)))}>
              刷新
            </Button>
          </div>
          <Space wrap size={6} style={{ marginTop: 10 }}>
            <Tag color={status?.joints_source === "rtsi" ? "green" : status?.dry_run ? "default" : "orange"}>
              source={status?.joints_source || "—"}
            </Tag>
            <Tag color={status?.measured ? "green" : "orange"}>
              {status?.measured ? "measured" : "unmeasured"}
            </Tag>
            <Tag>{status?.live_backend || "—"}</Tag>
          </Space>
          <Text type="secondary" style={{ display: "block", marginTop: 8, fontSize: 12 }}>
            {model
              ? `TCP ${JSON.stringify(model.tcp_xyz)} · FK ${model.has_fk ? "ok" : "missing"} · sensors cs_write ${model.has_cs_write ? "ok" : "missing"}`
              : "加载型号中…"}
          </Text>
          <Text type="secondary" style={{ display: "block", marginTop: 4, fontSize: 12 }}>
            顺序：Connect → 等 source=rtsi 与示教器一致 → Arm → 下发 → Disarm
          </Text>
          {!status?.dry_run && status?.connected && !status?.can_arm && (
            <Alert
              type="warning"
              showIcon
              style={{ marginTop: 10 }}
              message="等待 RTSI measured 关节（joints_source=rtsi）后再 Arm"
            />
          )}
          {!!status?.error && (
            <Alert
              type="error"
              showIcon
              style={{ marginTop: 10 }}
              message={String(status.error)}
            />
          )}
        </Card>

        <Card size="small" title="当前位姿">
          <div className="arm-readout">
            <div className="arm-readout-row">
              <div className="arm-readout-label">Joints</div>
              <div className="arm-readout-value">{fmtCsv6(status?.joints_rad) || "—"}</div>
              <Button size="small" type="link" onClick={doCopyJoints} style={{ padding: 0, height: "auto" }}>复制</Button>
            </div>
            <div className="arm-readout-row">
              <div className="arm-readout-label">Pose</div>
              <div className="arm-readout-value">{fmtCsv6(pose) || "—"}</div>
              <Button size="small" type="link" onClick={doCopyPose} style={{ padding: 0, height: "auto" }}>复制</Button>
            </div>
            <div className="arm-pose-grid">
              {poseLabels.map((k, i) => (
                <div className="arm-pose-cell" key={k}>
                  <span className="k">{k}</span>
                  <span className="v">
                    {pose && Number.isFinite(Number(pose[i])) ? Number(pose[i]).toFixed(4) : "—"}
                  </span>
                </div>
              ))}
            </div>
            <div className="arm-readout-row" style={{ marginTop: 4 }}>
              <div className="arm-readout-label">Home</div>
              <div className="arm-readout-value" style={{ color: "#595959" }}>
                {fmtCsv6(status?.home_joints_rad || model?.home_joints_rad) || "—"}
              </div>
              <span />
            </div>
          </div>
          {copyHint && (
            <Text type="success" style={{ display: "block", marginTop: 8, fontSize: 12 }}>{copyHint}</Text>
          )}
        </Card>

        <Card size="small" title="绝对下发">
          <div className="arm-dispatch">
            <Space wrap style={{ marginBottom: 8 }} align="center">
              <Text type="secondary">类型</Text>
              <Select
                style={{ width: 110 }}
                value={dispatchMode}
                onChange={setDispatchMode}
                options={[
                  { value: "joints", label: "joints" },
                  { value: "pose", label: "pose" },
                ]}
              />
              <Text type="secondary" style={{ fontSize: 12 }}>
                CSV · 与复制格式一致
              </Text>
            </Space>
            <Input
              value={dispatchText}
              onChange={(e) => setDispatchText(e.target.value)}
              placeholder={
                dispatchMode === "pose"
                  ? "0.4000,0.0500,0.4500,3.1416,0.0000,0.0000"
                  : "2.8695,-1.9539,1.8019,-1.6618,4.7475,-0.0275"
              }
              onPressEnter={doDispatch}
              style={{ marginBottom: 8 }}
            />
            <div className="arm-toolbar">
              <Button type="primary" onClick={doDispatch}>下发</Button>
              <Button onClick={() => {
                const csv = dispatchMode === "pose"
                  ? fmtCsv6(status?.cartesian_xyzrpy)
                  : fmtCsv6(status?.joints_rad || joints);
                if (csv) setDispatchText(csv);
                else message.warning("无当前值可填入");
              }}>填入当前</Button>
              <Button onClick={dispatchMode === "pose" ? doCopyPose : doCopyJoints}>
                复制{dispatchMode === "pose" ? " pose" : " joints"}
              </Button>
            </div>
          </div>
        </Card>

        <Card
          size="small"
          title="关节点动"
          extra={
            <Space size={8}>
              <span style={{ fontSize: 12, color: "#8c8c8c" }}>幽灵预览</span>
              <Switch
                size="small"
                checked={ghostVisible}
                checkedChildren="显"
                unCheckedChildren="隐"
                onChange={(on) => {
                  setGhostShown(on);
                  if (on) paintGhost(previewJointsRef.current || joints);
                }}
              />
            </Space>
          }
        >
          <div className="arm-toolbar" style={{ marginBottom: 10 }}>
            <Button size="small" type="primary" onClick={() => run("/api/arm/command", { joints_rad: joints })}>
              下发滑条
            </Button>
            <Button size="small" onClick={syncSlidersFromLive}>同步到实测</Button>
          </div>
          <Text type="secondary" style={{ display: "block", marginBottom: 8, fontSize: 12 }}>
            拨动滑条只更新半透明幽灵臂；实体臂跟随 Connect/RTSI，互不抢姿。
          </Text>
          {[0, 1, 2, 3, 4, 5].map((i) => {
            const lo = jointSoftLimit(model || status?.model, i, "min");
            const hi = jointSoftLimit(model || status?.model, i, "max");
            const val = Number(joints[i] || 0);
            // Clamp display value into this joint's own range so Ant Slider stays valid.
            const shown = Math.min(hi, Math.max(lo, val));
            return (
              <div className="arm-joint-row" key={i}>
                <span className="lab">J{i + 1}</span>
                <Slider
                  style={{ margin: 0 }}
                  min={lo}
                  max={hi}
                  step={0.001}
                  value={shown}
                  onChange={(v) => onJointChange(i, v)}
                />
                <span className="arm-joint-val" title={`${(lo * 180 / Math.PI).toFixed(1)}° … ${(hi * 180 / Math.PI).toFixed(1)}°`}>
                  {val.toFixed(3)}
                </span>
              </div>
            );
          })}
          <Divider style={{ margin: "10px 0 8px" }} />
          <div className="arm-section-title" style={{ marginBottom: 6 }}>夹爪开度</div>
          <div className="arm-joint-row arm-grip-open-row">
            <span className="lab">闭</span>
            <Slider
              style={{ margin: 0 }}
              disabled={!gripOpenCapable}
              min={0}
              max={Math.max(0.1, (Number(gripOpenHiM) || GRIP_OPEN_DEFAULT_HI_M) * 1000)}
              step={0.5}
              value={Math.max(0, Math.min(
                (Number(gripOpenHiM) || GRIP_OPEN_DEFAULT_HI_M) * 1000,
                (Number(gripOpenM) || 0) * 1000,
              ))}
              onChange={onGripOpenMm}
            />
            <span className="lab">开</span>
            <span className="arm-joint-val">
              {((Number(gripOpenM) || 0) * 1000).toFixed(1)} mm
            </span>
          </div>
        </Card>
      </div>

      <div className="arm-canvas-wrap">
        <div className="arm-canvas-top">
          <span className="pill">{model?.label || "Cell C · CS66 G26562"}</span>
          <span className="pill">
            {status?.connected ? (status?.armed ? "Connected · Armed" : "Connected") : "Disconnected"}
            {status?.dry_run ? " · dry-run" : ""}
            {ghostVisible ? " · ghost" : ""}
          </span>
        </div>
        <div className="arm-canvas-side-stack">
          <div className={`arm-layers-panel${layersOpen ? "" : " collapsed"}`}>
            <div className="arm-layers-head">
              <span className="arm-layers-title">模型图层</span>
              <Button
                size="small"
                type="text"
                className="arm-layers-toggle"
                onClick={() => setLayersOpen((v) => !v)}
              >
                {layersOpen ? "收起" : "展开"}
              </Button>
            </div>
            {layersOpen && (
              <div className="arm-layers-body">
                <div className="arm-layers-hint">按来源文件分层 · 开关仅影响显示</div>
                {layerRows.map((row) => (
                  <div
                    key={row.id}
                    className={`arm-layers-row${row.indent ? " indent" : ""}${row.visible ? "" : " dim"}`}
                  >
                    <span
                      className="arm-layers-swatch"
                      style={{ background: row.color || "#888" }}
                      title={row.color}
                    />
                    <div className="arm-layers-meta">
                      <div className="arm-layers-label">{row.label}</div>
                      <div className="arm-layers-file" title={row.file}>{row.file}</div>
                    </div>
                    <Switch
                      size="small"
                      checked={!!row.visible}
                      disabled={!row.loaded && row.id !== "ghost"}
                      onChange={(on) => setLayerVisible(row.id, on)}
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
        <div className="arm-canvas-host" ref={hostRef} />
        <div className="arm-hint">{hint}</div>
        {!!sceneMarkers.length && (
          <div className="arm-marker-legend">
            <div className="title">M2 点位 · {sceneMarkers.length}</div>
            {sceneMarkers.map((m) => (
              <div className="row" key={m.id}>
                <span className="dot" style={{ background: `#${Number(m.color).toString(16).padStart(6, "0")}` }} />
                <span>{m.label}</span>
              </div>
            ))}
            <Button
              size="small"
              style={{ marginTop: 6, pointerEvents: "auto" }}
              onClick={() => { armSceneMarkers.clear(); message.info("已清除标记"); }}
            >
              清除
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}

