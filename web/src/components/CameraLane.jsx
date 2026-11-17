import { useEffect, useState } from 'react';
import { Card, Button, Select, Space, Tag, InputNumber, Typography, message } from 'antd';
import { api } from '../lib/api.js';
import { pickCandidateSerial } from '../lib/cameras.js';

const { Text } = Typography;

const RES_PRESETS = [
  { value: '640x480', label: '640 × 480', width: 640, height: 480 },
  { value: '848x480', label: '848 × 480', width: 848, height: 480 },
  { value: '1280x720', label: '1280 × 720', width: 1280, height: 720 },
  { value: '1280x800', label: '1280 × 800', width: 1280, height: 800 },
  { value: '1920x1080', label: '1920 × 1080', width: 1920, height: 1080 },
];

const FPS_PRESETS = [6, 15, 30, 60];

function storageKey(role) {
  return `rs-detect.m0.streamProfile.${role || 'scene'}`;
}

function loadProfile(role) {
  try {
    const raw = localStorage.getItem(storageKey(role));
    if (!raw) return null;
    const p = JSON.parse(raw);
    const w = Number(p.width);
    const h = Number(p.height);
    const f = Number(p.fps);
    if ([w, h, f].every((x) => Number.isFinite(x) && x > 0)) {
      return { width: w, height: h, fps: f };
    }
  } catch (_) {}
  return null;
}

function saveProfile(role, profile) {
  try {
    localStorage.setItem(storageKey(role), JSON.stringify(profile));
  } catch (_) {}
}

function resKey(width, height) {
  return `${width}x${height}`;
}

export default function CameraLane({ title, role, candidates, color, serial, onSerialChange }) {
  const [running, setRunning] = useState(false);
  const [recording, setRecording] = useState(false);
  const [tick, setTick] = useState(0);
  const [intr, setIntr] = useState(null);
  const saved = loadProfile(role) || { width: 640, height: 480, fps: 30 };
  const [width, setWidth] = useState(saved.width);
  const [height, setHeight] = useState(saved.height);
  const [fps, setFps] = useState(saved.fps);
  const [resPreset, setResPreset] = useState(() => {
    const k = resKey(saved.width, saved.height);
    return RES_PRESETS.some((p) => p.value === k) ? k : 'custom';
  });

  useEffect(() => {
    if (!candidates?.length) return;
    const next = pickCandidateSerial(candidates, serial);
    if (next && next !== serial) onSerialChange(next);
  }, [candidates]);

  useEffect(() => {
    if (!running || !serial) return;
    const id = setInterval(() => setTick(t => t + 1), 200);
    return () => clearInterval(id);
  }, [running, serial]);

  const applyResPreset = (value) => {
    setResPreset(value);
    if (value === 'custom') return;
    const p = RES_PRESETS.find((x) => x.value === value);
    if (!p) return;
    setWidth(p.width);
    setHeight(p.height);
  };

  const bind = async (sn) => {
    onSerialChange(sn);
    if (!sn) return;
    try {
      await api("/api/selection", {
        method: "POST",
        body: JSON.stringify({ role, serial: sn, persist: true }),
      });
    } catch (e) { message.error(String(e.message || e)); }
  };

  const start = async () => {
    const w = Math.max(16, Math.round(Number(width) || 640));
    const h = Math.max(16, Math.round(Number(height) || 480));
    const f = Math.max(1, Math.round(Number(fps) || 30));
    const profile = { width: w, height: h, fps: f };
    try {
      await api("/api/stream/start", {
        method: "POST",
        body: JSON.stringify({ serial, role, ...profile }),
      });
      setWidth(w);
      setHeight(h);
      setFps(f);
      saveProfile(role, profile);
      setRunning(true);
      message.success(`${role} 出流已启动 ${w}×${h}@${f}；M1 序列号已联动`);
    } catch (e) { message.error(String(e.message || e)); }
  };
  const stop = async () => {
    try {
      if (recording) await api(`/api/record/stop?serial=${encodeURIComponent(serial)}`, { method: "POST" });
      await api(`/api/stream/stop?serial=${encodeURIComponent(serial)}`, { method: "POST" });
      setRecording(false);
      setRunning(false);
      message.info("已停止");
    } catch (e) { message.error(String(e.message || e)); }
  };
  const snap = async () => {
    try {
      const r = await api(`/api/snapshot?serial=${encodeURIComponent(serial)}`, { method: "POST" });
      const cloudHint = r.point_cloud
        ? ` · 点云 ${r.point_count || "?"} pts → ${r.point_cloud}`
        : " · 无点云（缺深度/内参）";
      message.success(`已拍照: ${r.color}${cloudHint}`);
    } catch (e) { message.error(String(e.message || e)); }
  };
  const toggleRec = async () => {
    try {
      if (!recording) {
        const r = await api(`/api/record/start?serial=${encodeURIComponent(serial)}`, { method: "POST" });
        setRecording(true);
        message.success("录像中: " + r.path);
      } else {
        const r = await api(`/api/record/stop?serial=${encodeURIComponent(serial)}`, { method: "POST" });
        setRecording(false);
        message.success(`已保存 ${r.path} (${r.frames} frames)`);
      }
    } catch (e) { message.error(String(e.message || e)); }
  };
  const loadIntr = async () => {
    try {
      const r = await api(`/api/intrinsics?serial=${encodeURIComponent(serial)}&from_live=${running}`);
      setIntr(r);
      message.success("内参已从 SDK 读取");
    } catch (e) { message.error(String(e.message || e)); }
  };

  const src = running && serial
    ? `/api/frame.jpg?serial=${encodeURIComponent(serial)}&which=color&t=${tick}`
    : null;

  return (
    <Card title={<span className="lane-title">{title} <Tag color={color}>{role}</Tag></span>}
          extra={running ? <Tag color="green">LIVE {width}×{height}@{fps}</Tag> : <Tag>IDLE</Tag>}>
      <Space direction="vertical" style={{ width: "100%" }} size="middle">
        <Select
          style={{ width: "100%" }}
          placeholder="选择相机"
          value={serial}
          onChange={bind}
          options={(candidates || []).map(d => ({
            value: d.serial,
            label: `${d.name} [${d.serial}]${d.is_mock ? " (mock)" : ""}`,
          }))}
        />
        <div className="m0-stream-profile">
          <Text type="secondary" style={{ display: 'block', marginBottom: 6 }}>出流参数（启动时生效）</Text>
          <Space wrap align="center">
            <span>分辨率</span>
            <Select
              style={{ width: 150 }}
              value={resPreset}
              disabled={running}
              onChange={applyResPreset}
              options={[
                ...RES_PRESETS.map((p) => ({ value: p.value, label: p.label })),
                { value: 'custom', label: '自定义' },
              ]}
            />
            <InputNumber
              min={16}
              max={4096}
              step={16}
              value={width}
              disabled={running}
              addonBefore="W"
              style={{ width: 120 }}
              onChange={(v) => {
                const w = typeof v === 'number' ? v : width;
                setWidth(w);
                const k = resKey(w, height);
                setResPreset(RES_PRESETS.some((p) => p.value === k) ? k : 'custom');
              }}
            />
            <InputNumber
              min={16}
              max={4096}
              step={16}
              value={height}
              disabled={running}
              addonBefore="H"
              style={{ width: 120 }}
              onChange={(v) => {
                const h = typeof v === 'number' ? v : height;
                setHeight(h);
                const k = resKey(width, h);
                setResPreset(RES_PRESETS.some((p) => p.value === k) ? k : 'custom');
              }}
            />
            <span>频率</span>
            <Select
              style={{ width: 100 }}
              value={FPS_PRESETS.includes(Number(fps)) ? Number(fps) : 'custom'}
              disabled={running}
              onChange={(v) => {
                if (v === 'custom') return;
                setFps(Number(v));
              }}
              options={[
                ...FPS_PRESETS.map((f) => ({ value: f, label: `${f} fps` })),
                { value: 'custom', label: '自定义' },
              ]}
            />
            <InputNumber
              min={1}
              max={90}
              step={1}
              value={fps}
              disabled={running}
              addonAfter="fps"
              style={{ width: 110 }}
              onChange={(v) => setFps(typeof v === 'number' ? v : fps)}
            />
          </Space>
        </div>
        <Space wrap>
          <Button type="primary" onClick={start} disabled={!serial || running}>开始出流</Button>
          <Button onClick={stop} disabled={!running}>停止</Button>
          <Button onClick={snap} disabled={!running}>拍照</Button>
          <Button danger={recording} onClick={toggleRec} disabled={!running}>
            {recording ? "停止录像" : "开始录像"}
          </Button>
          <Button onClick={loadIntr} disabled={!serial}>读取内参</Button>
        </Space>
        <div className="preview-box">
          {src
            ? <img src={src} alt="preview" />
            : <span style={{ color: "#888" }}>无画面</span>}
        </div>
        {intr && (
          <pre className="mono" style={{ maxHeight: 160, overflow: "auto", background: "#fafafa", padding: 8 }}>
            {JSON.stringify(intr, null, 2)}
          </pre>
        )}
      </Space>
    </Card>
  );
}
