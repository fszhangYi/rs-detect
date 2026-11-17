import { useEffect, useState, useCallback, useMemo, useRef, lazy, Suspense } from 'react';
import { Layout, Typography, Button, Space, Tabs, Tag, Modal, Row, Col, Tooltip, message } from 'antd';
import { api, API } from './lib/api.js';
import { tabFromHash, setHashTab } from './lib/hashTabs.js';
import { armPreload, preloadArmAssets } from './lib/armPreload.js';
import CameraLane from './components/CameraLane.jsx';
import PanelFallback from './components/PanelFallback.jsx';
import ErrorBoundary from './components/ErrorBoundary.jsx';

/** Heavy panels — arm/three loaded last via loadArmBundle(). */
const TrainPanel = lazy(() => import('./components/TrainPanel.jsx'));
const ExtrinsicsPanel = lazy(() => import('./components/ExtrinsicsPanel.jsx'));
const DetectPanel = lazy(() => import('./components/DetectPanel.jsx'));
const GraspPanel = lazy(() => import('./components/GraspPanel.jsx'));
const SettingsModal = lazy(() => import('./components/SettingsModal.jsx'));

let armPanelModulePromise = null;
function loadArmPanelModule() {
  if (!armPanelModulePromise) {
    armPanelModulePromise = import('./components/ArmPanel.jsx');
  }
  return armPanelModulePromise;
}
const ArmPanel = lazy(() => loadArmPanelModule());

const { Header, Content } = Layout;
const { Title, Text } = Typography;

function idleSchedule(fn, timeout = 2500) {
  if (typeof window !== 'undefined' && typeof window.requestIdleCallback === 'function') {
    return window.requestIdleCallback(fn, { timeout });
  }
  return window.setTimeout(fn, Math.min(timeout, 1200));
}

function idleCancel(id) {
  if (typeof window !== 'undefined' && typeof window.cancelIdleCallback === 'function') {
    window.cancelIdleCallback(id);
  } else {
    clearTimeout(id);
  }
}

function LazyPanel({ label, children }) {
  return (
    <ErrorBoundary label={label}>
      <Suspense fallback={<PanelFallback label={label} />}>{children}</Suspense>
    </ErrorBoundary>
  );
}

function initialSafeTab() {
  const t = tabFromHash();
  // Arm disabled until resources ready — never boot onto it
  return t === 'arm' ? 'cams' : t;
}

export default function App() {
  const wantedArmRef = useRef(tabFromHash() === 'arm');
  const [tab, setTab] = useState(initialSafeTab);
  const [mountedTabs, setMountedTabs] = useState(() => new Set([initialSafeTab()]));
  const [scan, setScan] = useState({ wrist_candidates: [], scene_candidates: [] });
  const [selection, setSelection] = useState({ wrist: "", scene: "", d405_serial: "", d435_serial: "" });
  const [wristSerial, setWristSerial] = useState(undefined);
  const [sceneSerial, setSceneSerial] = useState(undefined);

  /** Arm tab clickable only after three.js chunk + URDF/STL finish (loaded last). */
  const [armReady, setArmReady] = useState(false);
  const [armLoadError, setArmLoadError] = useState(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [sysBusy, setSysBusy] = useState(false);

  const refreshSelection = useCallback(async () => {
    const sel = await api("/api/selection");
    setSelection(sel);
    if (sel.wrist) setWristSerial(sel.wrist);
    if (sel.scene) setSceneSerial(sel.scene);
    return sel;
  }, []);

  const refresh = async () => {
    try {
      const data = await api("/api/scan");
      setScan(data);
      if (data.selection) {
        setSelection(data.selection);
        if (data.selection.wrist) setWristSerial(data.selection.wrist);
        if (data.selection.scene) setSceneSerial(data.selection.scene);
      }
      message.success(`扫描到 ${data.devices?.length || 0} 台设备`);
    } catch (e) { message.error(String(e.message || e)); }
  };

  useEffect(() => { refresh(); }, []);

  useEffect(() => {
    const onHash = () => {
      const next = tabFromHash();
      if (next === 'arm' && !armReady) {
        wantedArmRef.current = true;
        message.info('机械臂资源仍在后台加载，就绪后可进入');
        setHashTab(tab === 'arm' ? 'cams' : tab);
        return;
      }
      setTab(next);
      setMountedTabs((prev) => {
        if (prev.has(next)) return prev;
        const n = new Set(prev);
        n.add(next);
        return n;
      });
    };
    window.addEventListener('hashchange', onHash);
    if (!location.hash || tabFromHash() === 'arm') setHashTab(tab);
    return () => window.removeEventListener('hashchange', onHash);
  }, [armReady, tab]);

  const onTabChange = (key) => {
    if (key === 'arm' && !armReady) {
      message.info('机械臂资源加载中，请稍候');
      return;
    }
    setTab(key);
    setHashTab(key);
    setMountedTabs((prev) => {
      if (prev.has(key)) return prev;
      const n = new Set(prev);
      n.add(key);
      return n;
    });
    if (key === 'calib' || key === 'detect') refreshSelection().catch(() => {});
  };

  const onWristSerial = async (sn) => {
    setWristSerial(sn);
    setSelection((s) => ({ ...s, wrist: sn, d405_serial: sn }));
  };
  const onSceneSerial = async (sn) => {
    setSceneSerial(sn);
    setSelection((s) => ({ ...s, scene: sn, d435_serial: sn }));
  };

  // Load order: (1) first paint (2) warm mid panels (3) LAST — three.js + URDF/STL
  useEffect(() => {
    let cancelled = false;
    let idle1 = 0;
    let idle2 = 0;

    const openArmIfWanted = () => {
      if (!wantedArmRef.current) return;
      wantedArmRef.current = false;
      setTab('arm');
      setHashTab('arm');
      setMountedTabs((prev) => {
        const n = new Set(prev);
        n.add('arm');
        return n;
      });
    };

    const loadArmBundleLast = async () => {
      if (cancelled) return;
      try {
        // 1) Unlock tab once three.js / ArmPanel chunk is ready (do not wait for ~30MB STL).
        await loadArmPanelModule();
        if (cancelled) return;
        setArmReady(true);
        setArmLoadError(null);
        openArmIfWanted();
        // 2) Prefetch URDF + STL in background (LoadingManager blob cache).
        preloadArmAssets().catch((e) => {
          console.warn('[arm] mesh preload', e);
          if (!cancelled) setArmLoadError(String(e.message || e));
        });
      } catch (e) {
        if (cancelled) return;
        setArmLoadError(String(e.message || e));
        setArmReady(false);
      }
    };

    idle1 = idleSchedule(() => {
      if (cancelled) return;
      // Mid-priority panels first (no three.js).
      Promise.all([
        import('./components/DetectPanel.jsx'),
        import('./components/ExtrinsicsPanel.jsx'),
        import('./components/TrainPanel.jsx'),
        import('./components/GraspPanel.jsx'),
      ]).finally(() => {
        if (cancelled) return;
        // After mid panels, load arm JS last.
        idle2 = idleSchedule(() => { loadArmBundleLast(); }, 2000);
      });
    }, 1500);

    return () => {
      cancelled = true;
      idleCancel(idle1);
      idleCancel(idle2);
    };
  }, []);

  const items = useMemo(() => {
    const armLabel = armReady
      ? '机械臂 (Cell C)'
      : (
        <Tooltip title={armLoadError ? `加载失败：${armLoadError}` : '三维与网格资源最后加载，就绪后可点击'}>
          <span className="tab-arm-pending">机械臂 (加载中…)</span>
        </Tooltip>
      );
    return [
      {
        key: 'cams',
        label: '相机出流 (M0)',
        children: (
          <>
            <Space style={{ marginBottom: 16 }}>
              <Button type="primary" onClick={refresh}>自动扫描 RealSense</Button>
              <Text type="secondary">选定相机后自动写入 M1 序列号。预览按 4:3 等比 letterbox。</Text>
            </Space>
            <Row gutter={16}>
              <Col span={12}>
                <CameraLane
                  title="腕部相机待选区（D405）"
                  role="wrist"
                  candidates={scan.wrist_candidates}
                  color="orange"
                  serial={wristSerial}
                  onSerialChange={onWristSerial}
                />
              </Col>
              <Col span={12}>
                <CameraLane
                  title="场景相机待选区（D435）"
                  role="scene"
                  candidates={scan.scene_candidates}
                  color="blue"
                  serial={sceneSerial}
                  onSerialChange={onSceneSerial}
                />
              </Col>
            </Row>
          </>
        ),
      },
      {
        key: 'train',
        label: 'YOLO 训练',
        children: mountedTabs.has('train') ? (
          <LazyPanel label="YOLO 训练">
            <TrainPanel active={tab === 'train'} />
          </LazyPanel>
        ) : null,
      },
      {
        key: 'calib',
        label: '标定 (M1)',
        children: mountedTabs.has('calib') ? (
          <LazyPanel label="标定">
            <ExtrinsicsPanel selection={selection} />
          </LazyPanel>
        ) : null,
      },
      {
        key: 'detect',
        label: '检测通路 (M2)',
        children: mountedTabs.has('detect') ? (
          <LazyPanel label="检测通路">
            <DetectPanel selection={selection} />
          </LazyPanel>
        ) : null,
      },
      {
        key: 'grasp',
        label: 'Grasp Agent',
        children: mountedTabs.has('grasp') ? (
          <LazyPanel label="Grasp Agent">
            <GraspPanel active={tab === 'grasp'} selection={selection} />
          </LazyPanel>
        ) : null,
      },
      {
        key: 'arm',
        label: armLabel,
        disabled: !armReady,
        children: armReady && mountedTabs.has('arm') ? (
          <LazyPanel label="机械臂三维">
            <ArmPanel active={tab === 'arm'} />
          </LazyPanel>
        ) : null,
      },
    ];
  }, [tab, mountedTabs, scan, selection, wristSerial, sceneSerial, armReady, armLoadError]);

  const waitForHealthThenReload = () => {
    const started = Date.now();
    const tick = async () => {
      if (Date.now() - started > 45000) {
        message.destroy();
        message.error('后端重启超时，请手动刷新');
        setSysBusy(false);
        return;
      }
      try {
        const res = await fetch(API + '/api/health', { cache: 'no-store' });
        if (res.ok) {
          message.destroy();
          message.success('后端已就绪，刷新页面');
          location.reload();
          return;
        }
      } catch (_) { /* still down */ }
      setTimeout(tick, 500);
    };
    setTimeout(tick, 900);
  };

  const onRestart = () => {
    Modal.confirm({
      title: '重启后端？',
      content: '将停止相机/臂控并重启服务，成功后自动刷新本页。',
      okText: '重启',
      cancelText: '取消',
      onOk: async () => {
        setSysBusy(true);
        message.loading({ content: '正在重启后端…', key: 'sys', duration: 0 });
        try {
          await api('/api/system/restart', { method: 'POST', body: '{}' });
        } catch (_) {
          // connection drop expected as process exits
        }
        waitForHealthThenReload();
      },
    });
  };

  const onSafeShutdown = () => {
    Modal.confirm({
      title: '安全关闭？',
      content: '将停止相机/录像/臂控并关闭后端，然后关闭本页。',
      okText: '安全关闭',
      okButtonProps: { danger: true },
      cancelText: '取消',
      onOk: async () => {
        setSysBusy(true);
        message.loading({ content: '正在安全关闭…', key: 'sys', duration: 0 });
        try {
          await api('/api/system/shutdown', { method: 'POST', body: '{}' });
        } catch (_) { /* expected */ }
        message.destroy();
        try { window.close(); } catch (_) { /* ignore */ }
        document.body.innerHTML =
          '<div style="padding:48px;font-family:sans-serif;max-width:520px">' +
          '<h2>后端已安全关闭</h2>' +
          '<p>可关闭此浏览器标签页。再次使用请运行 <code>./start_web.sh</code>。</p>' +
          '</div>';
      },
    });
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#001529', display: 'flex', alignItems: 'center', gap: 12 }}>
        <Title level={3} style={{ color: '#fff', margin: 0 }}>rs-detect</Title>
        <Text style={{ color: '#aaa' }}>相机 · 标定 · 检测 · Cell C 臂控</Text>
        <div className="header-actions" style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
          {!armReady
            ? <Tag>臂三维包加载中…</Tag>
            : (armPreload.ready
              ? <Tag color="cyan">臂模型已预加载</Tag>
              : <Tag color="blue">臂 Tab 可用 · 网格后台加载</Tag>)}
          <Button type="primary" onClick={() => setSettingsOpen(true)} disabled={sysBusy}>设置</Button>
          <Button onClick={onRestart} loading={sysBusy}>重启</Button>
          <Button danger type="primary" onClick={onSafeShutdown} disabled={sysBusy}>安全关闭</Button>
        </div>
      </Header>
      <Content style={{ padding: 24 }}>
        <Tabs
          activeKey={tab}
          items={items}
          destroyInactiveTabPane={false}
          onChange={onTabChange}
        />
      </Content>
      {settingsOpen ? (
        <Suspense fallback={null}>
          <SettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} />
        </Suspense>
      ) : null}
    </Layout>
  );
}
