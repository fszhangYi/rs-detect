import { useEffect, useState, useCallback, useRef } from 'react';
import { Card, Row, Col, Button, Space, Tag, Alert, Input, InputNumber, Select, Slider, Steps, Divider, Descriptions, Switch, Typography, message } from 'antd';
import { api } from '../lib/api.js';
import { fmtXYZ } from '../lib/format.js';
import { armSceneMarkers, ARM_MARKER_COLORS } from '../lib/armMarkers.js';
import ShowOnArmBtn from './ShowOnArmBtn.jsx';

const { Text } = Typography;

export default function DetectPanel({ selection }) {
  const [current, setCurrent] = useState(0);
  const [method, setMethod] = useState("auto");
  const [zLift, setZLift] = useState(0.08);
  const [pregraspLift, setPregraspLift] = useState(0.05);
  const [mvN, setMvN] = useState(5);
  const [mvRadius, setMvRadius] = useState(0.03);
  const [busy, setBusy] = useState(false);
  const [coarse, setCoarse] = useState(null);
  const [moveSafe, setMoveSafe] = useState(null);
  const [fineServo, setFineServo] = useState(null);
  const [grasp, setGrasp] = useState(null);
  const [safePoints, setSafePoints] = useState([]);
  const [safeRun, setSafeRun] = useState(null);
  const [suctionRelease, setSuctionRelease] = useState(null);
  const [suctionOn, setSuctionOn] = useState(false);
  const [previewMode, setPreviewMode] = useState(false);
  const [vizTick, setVizTick] = useState(0);
  const [cfg, setCfg] = useState(null);

  useEffect(() => {
    api("/api/detect/config").then((c) => {
      setCfg(c);
      if (c?.pipeline?.safe_z_lift_m != null) setZLift(Number(c.pipeline.safe_z_lift_m));
      if (c?.pipeline?.pregrasp_z_lift_m != null) setPregraspLift(Number(c.pipeline.pregrasp_z_lift_m));
      if (c?.pipeline?.multiview_n != null) setMvN(Number(c.pipeline.multiview_n));
      if (c?.pipeline?.multiview_radius_m != null) setMvRadius(Number(c.pipeline.multiview_radius_m));
      if (c?.method) setMethod(c.method);
    }).catch(() => {});
    api("/api/pipeline/safe_points").then((sp) => {
      setSafePoints(Array.isArray(sp?.points) ? sp.points : []);
    }).catch(() => {});
    api("/api/pipeline/suction").then((s) => setSuctionOn(!!s?.on)).catch(() => {});
    api("/api/pipeline/workflow").then((wf) => {
      if (wf?.coarse) {
        setCoarse(wf.coarse);
        if (wf.z_lift_m != null) setZLift(Number(wf.z_lift_m));
      }
      if (wf?.move_safe) setMoveSafe(wf.move_safe);
      if (wf?.fine_servo) setFineServo(wf.fine_servo);
      if (wf?.grasp) setGrasp(wf.grasp);
      if (wf?.safe_run) setSafeRun(wf.safe_run);
      if (wf?.suction_release) setSuctionRelease(wf.suction_release);
      if (wf?.pregrasp_z_lift_m != null) setPregraspLift(Number(wf.pregrasp_z_lift_m));
      // Restore results only — do not auto-advance the Steps UI.
    }).catch(() => {});
  }, []);

  const sceneSerial = selection?.scene || selection?.d435_serial || coarse?.serial;
  const wristSerial = selection?.wrist || selection?.d405_serial;
  const coarseViz = coarse?.has_viz && sceneSerial
    ? `/api/detect/viz.jpg?serial=${encodeURIComponent(sceneSerial)}&t=${vizTick}`
    : null;
  const fineViz = fineServo && wristSerial
    ? `/api/detect/viz.jpg?serial=${encodeURIComponent(wristSerial)}&t=${vizTick}`
    : null;

  const runStep1Coarse = async () => {
    if (!selection?.scene && !selection?.d435_serial) {
      message.warning("请先在 M0 选定并出流场景相机 D435");
      return;
    }
    setBusy(true);
    try {
      const payload = await api("/api/pipeline/coarse", {
        method: "POST",
        body: JSON.stringify({ method, z_lift_m: zLift, take_snapshot: true }),
      });
      setCoarse(payload);
      setVizTick((t) => t + 1);
      if (payload.ok) {
        message.success("粗定位完成，已生成安全点 p_safe");
      } else {
        message.warning(payload.message || "粗定位失败");
      }
    } catch (e) {
      message.error(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const onZLiftChange = async (v) => {
    const lift = typeof v === "number" ? v : zLift;
    setZLift(lift);
    if (!coarse?.ok || !coarse?.p_coarse) return;
    try {
      const wf = await api("/api/pipeline/safe_z", {
        method: "POST",
        body: JSON.stringify({ z_lift_m: lift }),
      });
      setCoarse((prev) => ({ ...(prev || {}), z_lift_m: wf.z_lift_m, p_safe: wf.p_safe }));
    } catch (_) {
      const p = coarse.p_coarse;
      setCoarse((prev) => ({ ...(prev || {}), z_lift_m: lift, p_safe: [p[0], p[1], p[2] + lift] }));
    }
  };

  const runStep2MoveSafe = async () => {
    if (!coarse?.p_safe) {
      message.warning("没有 p_safe，请先完成粗定位");
      return;
    }
    setBusy(true);
    try {
      const payload = await api("/api/pipeline/move_safe", { method: "POST", body: "{}" });
      setMoveSafe(payload);
      if (payload.ok) {
        message.success("已运动到安全点 p_safe");
      } else {
        message.warning(payload.message || "运动失败");
      }
    } catch (e) {
      message.error(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const runStep3Multiview = async () => {
    if (!selection?.wrist && !selection?.d405_serial) {
      message.warning("请先在 M0 选定并出流腕部相机 D405");
      return;
    }
    setBusy(true);
    try {
      const payload = await api("/api/pipeline/multiview", {
        method: "POST",
        body: JSON.stringify({
          method,
          pregrasp_z_lift_m: pregraspLift,
          n_views: mvN,
          radius_m: mvRadius,
          move_pregrasp: true,
        }),
      });
      setFineServo(payload);
      setGrasp(null);
      setVizTick((t) => t + 1);
      if (payload.ok) {
        message.success(payload.message || "M3 多视角融合完成");
      } else {
        message.warning(payload.message || "多视角融合失败");
      }
    } catch (e) {
      message.error(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const runStep3FineServo = async () => {
    if (!selection?.wrist && !selection?.d405_serial) {
      message.warning("请先在 M0 选定并出流腕部相机 D405");
      return;
    }
    setBusy(true);
    try {
      const payload = await api("/api/pipeline/fine_servo", {
        method: "POST",
        body: JSON.stringify({
          method,
          pregrasp_z_lift_m: pregraspLift,
        }),
      });
      setFineServo(payload);
      setGrasp(null);
      setVizTick((t) => t + 1);
      if (payload.ok) {
        message.success(payload.message || "精定位闭环收敛");
      } else {
        message.warning(payload.message || "精定位未完全收敛（可查看 steps）");
      }
    } catch (e) {
      message.error(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const runStep4Grasp = async () => {
    if (!fineServo?.final_c_star && !fineServo?.final_p_pregrasp) {
      message.warning("请先完成精定位闭环");
      return;
    }
    setBusy(true);
    try {
      const payload = await api("/api/pipeline/grasp", { method: "POST", body: "{}" });
      setGrasp(payload);
      setSuctionOn(!!payload?.suction?.on);
      setSafeRun(null);
      setSuctionRelease(null);
      if (payload.ok) {
        message.success(payload.message || "抓取占位已记录");
      } else {
        message.warning(payload.message || "抓取占位失败");
      }
    } catch (e) {
      message.error(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const updateSafePoint = (idx, patch) => {
    setSafePoints((prev) => prev.map((p, i) => (i === idx ? { ...p, ...patch } : p)));
  };

  const saveSafePoints = async () => {
    setBusy(true);
    try {
      const payload = await api("/api/pipeline/safe_points", {
        method: "PUT",
        body: JSON.stringify({ points: safePoints }),
      });
      setSafePoints(Array.isArray(payload?.points) ? payload.points : safePoints);
      message.success("安全点列表已保存");
    } catch (e) {
      message.error(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const captureSafePoint = async () => {
    setBusy(true);
    try {
      const payload = await api("/api/pipeline/safe_points/capture", {
        method: "POST",
        body: JSON.stringify({ name: `tcp_${safePoints.length + 1}` }),
      });
      setSafePoints(Array.isArray(payload?.points) ? payload.points : []);
      message.success("已追加当前 TCP");
    } catch (e) {
      message.error(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const runStep5SafeSequence = async () => {
    if (!safePoints.length) {
      message.warning("安全点列表为空");
      return;
    }
    setBusy(true);
    try {
      const payload = await api("/api/pipeline/safe_points/run", {
        method: "POST",
        body: JSON.stringify({ points: safePoints }),
      });
      setSafeRun(payload);
      setSuctionOn(!!payload?.suction?.on);
      if (payload.ok) {
        message.success(payload.message || "安全点下发完成");
      } else {
        message.warning(payload.message || "安全点下发未完成");
      }
    } catch (e) {
      message.error(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const runStep6Release = async () => {
    setBusy(true);
    try {
      const payload = await api("/api/pipeline/suction/release", { method: "POST", body: "{}" });
      setSuctionRelease(payload);
      setSuctionOn(!!payload?.suction?.on);
      message.success(payload.message || "已放开吸盘");
    } catch (e) {
      message.error(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const stepItems = [
    { title: "D435 粗定位", description: "拍照 → p_coarse → Z 抬高安全点" },
    { title: "运动到安全点", description: "臂 TCP → p_safe" },
    { title: "D405 多视角精定位", description: "M3 视点采集 → 融合 → 预抓取" },
    { title: "抓取", description: "吸盘开启（占位）" },
    { title: "安全点列表", description: "编辑并依次下发运动" },
    { title: "放开吸盘", description: "释放真空（占位）" },
  ];

  const showStep = (n) => previewMode || current === n;
  const stepSection = (n, body) => {
    if (!showStep(n)) return null;
    return (
      <div
        key={`m2-step-${n}`}
        style={{
          marginBottom: previewMode ? 28 : 0,
          padding: previewMode ? 16 : 0,
          border: previewMode ? "1px solid #f0f0f0" : "none",
          borderRadius: previewMode ? 8 : 0,
          background: previewMode && current === n ? "#f6ffed" : (previewMode ? "#fff" : "transparent"),
        }}
      >
        {previewMode && (
          <Divider orientation="left" style={{ marginTop: 0 }}>
            <Space>
              <Tag color={current === n ? "green" : "default"}>Step {n + 1}</Tag>
              <Text strong>{stepItems[n].title}</Text>
              <Text type="secondary">{stepItems[n].description}</Text>
              {current !== n && (
                <Button size="small" type="link" onClick={() => setCurrent(n)}>设为当前</Button>
              )}
            </Space>
          </Divider>
        )}
        {body}
      </div>
    );
  };

  const showAllArmPoints = () => {
    const batch = [];
    if (coarse?.p_coarse) batch.push({ id: "p_coarse", label: "p_coarse", xyz: coarse.p_coarse, color: ARM_MARKER_COLORS.p_coarse });
    if (coarse?.p_safe) batch.push({ id: "p_safe", label: "p_safe", xyz: coarse.p_safe, color: ARM_MARKER_COLORS.p_safe });
    if (fineServo?.final_c_star) batch.push({ id: "c_star", label: "c*", xyz: fineServo.final_c_star, color: ARM_MARKER_COLORS.c_star });
    if (fineServo?.final_p_pregrasp) batch.push({ id: "p_pregrasp", label: "p_pregrasp", xyz: fineServo.final_p_pregrasp, color: ARM_MARKER_COLORS.p_pregrasp });
    (fineServo?.viewpoints || []).forEach((vp, i) => {
      if (vp?.xyz) batch.push({ id: `vp_${vp.name || i}`, label: `视点 ${vp.name || i}`, xyz: vp.xyz, color: ARM_MARKER_COLORS.viewpoint });
    });
    safePoints.forEach((pt, i) => {
      if (pt?.home || !pt?.xyz) return;
      batch.push({ id: `safe_${pt.name || i}`, label: pt.name || `safe_${i}`, xyz: pt.xyz, color: ARM_MARKER_COLORS.safe_point });
    });
    const n = armSceneMarkers.upsertMany(batch);
    if (!n) message.warning("暂无可显示的点位");
    else message.success(`已在机械臂画布标记 ${n} 个点，可切换到「机械臂」查看`);
  };

  return (
    <Card
      title="检测 / 吸取流程（M2）"
      extra={
        <Space wrap>
          {previewMode && <Tag color="blue">预览中 · 显示全部步骤</Tag>}
          <Button size="small" onClick={showAllArmPoints}>全部点位显示到臂画布</Button>
          <Button size="small" onClick={() => { armSceneMarkers.clear(); message.info("已清除臂画布标记"); }}>清除标记</Button>
          <Text type="secondary">预览模式</Text>
          <Switch
            checked={previewMode}
            onChange={(on) => {
              setPreviewMode(on);
              if (on) message.info("预览模式：展开全部流程内容");
            }}
            checkedChildren="开"
            unCheckedChildren="关"
          />
        </Space>
      }
    >
      {cfg && cfg.weights_ready === false && (
        <Alert
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
          message="尚未安装 YOLO 纸板权重"
          description={`当前 config 指向 ${cfg.weights || "models/cardboard_yolo.pt"}。M2 Step1 分割目标为 cardboard；mock 下会降低 conf 并保留最高分纸板 mask。权重缺失或无检出时才回退中心 ROI / HSV。`}
          action={
            <Button size="small" type="primary" onClick={() => { location.hash = "#/train"; }}>
              去训练
            </Button>
          }
        />
      )}
      <Steps
        current={current}
        items={stepItems}
        style={{ marginBottom: 24 }}
        onChange={(c) => {
          if (previewMode) {
            setCurrent(c);
            return;
          }
          if (c === 0) setCurrent(0);
          else if (c === 1 && coarse?.ok) setCurrent(1);
          else if (c === 2 && (moveSafe?.ok || coarse?.ok)) setCurrent(2);
          else if (c === 3 && fineServo) setCurrent(3);
          else if (c === 4 && (grasp?.ok || fineServo)) setCurrent(4);
          else if (c === 5 && (safeRun?.ok || grasp?.ok)) setCurrent(5);
          else message.info("请按顺序完成前面的步骤");
        }}
      />

      {previewMode && (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
          message="预览模式已开启：下方同时展示 Step1–6 全部内容。关闭后恢复逐步引导。"
        />
      )}

      {stepSection(0, (
        <>
          <Alert
            type="info"
            showIcon
            style={{ marginBottom: 16 }}
            message="第 1 步：D435 拍照并粗定位 → p_coarse；p_safe = p_coarse + [0,0,Δz]。请先在 M0 出流场景相机。"
          />
          <Space wrap style={{ marginBottom: 16 }} align="center">
            <Text>检测方法</Text>
            <Select
              style={{ width: 120 }}
              value={method}
              onChange={setMethod}
              options={["auto", "yolo", "hsv", "mock"].map((m) => ({ value: m, label: m }))}
            />
            <Text>安全点 Z 抬高 Δz</Text>
            <InputNumber min={0.01} max={0.5} step={0.01} value={zLift} addonAfter="m" onChange={onZLiftChange} />
            <Button type="primary" loading={busy} onClick={runStep1Coarse}>拍照并粗定位（D435）</Button>
            {coarse?.ok && <Button onClick={() => setCurrent(1)}>下一步：运动到安全点 →</Button>}
          </Space>
          <Row gutter={16}>
            <Col span={14}>
              <div className="preview-box">
                {coarseViz
                  ? <img src={coarseViz} alt="D435 coarse viz" />
                  : <span style={{ color: "#888" }}>点击「拍照并粗定位」后显示 Mask / 粗吸点</span>}
              </div>
            </Col>
            <Col span={10}>
              <Descriptions column={1} size="small" bordered>
                <Descriptions.Item label="场景串号">{sceneSerial || "未选定"}</Descriptions.Item>
                <Descriptions.Item label="状态">
                  {coarse == null ? "未运行" : (coarse.ok ? <Tag color="green">成功</Tag> : <Tag color="orange">失败</Tag>)}
                </Descriptions.Item>
                <Descriptions.Item label="p_coarse">
                  <Space wrap>
                    <span>{fmtXYZ(coarse?.p_coarse)}</span>
                    <ShowOnArmBtn id="p_coarse" label="p_coarse" xyz={coarse?.p_coarse} color={ARM_MARKER_COLORS.p_coarse} />
                  </Space>
                </Descriptions.Item>
                <Descriptions.Item label="Δz">{Number(coarse?.z_lift_m ?? zLift).toFixed(3)} m</Descriptions.Item>
                <Descriptions.Item label="p_safe">
                  <Space wrap>
                    <Text strong type="success">{fmtXYZ(coarse?.p_safe)}</Text>
                    <ShowOnArmBtn id="p_safe" label="p_safe" xyz={coarse?.p_safe} color={ARM_MARKER_COLORS.p_safe} />
                  </Space>
                </Descriptions.Item>
              </Descriptions>
            </Col>
          </Row>
        </>
      ))}

      {stepSection(1, (
        <>
          <Alert
            type="info"
            showIcon
            style={{ marginBottom: 16 }}
            message="第 2 步：将机械臂 TCP 运动到安全点 p_safe（保持当前姿态，只改 XYZ）。请先在「机械臂」Tab Connect + Arm（dry-run 亦可）。"
          />
          <Descriptions column={1} size="small" bordered style={{ marginBottom: 16, maxWidth: 640 }}>
            <Descriptions.Item label="p_safe 目标">
              <Space wrap>
                <span>{fmtXYZ(coarse?.p_safe)}</span>
                <ShowOnArmBtn id="p_safe" label="p_safe" xyz={coarse?.p_safe} color={ARM_MARKER_COLORS.p_safe} />
              </Space>
            </Descriptions.Item>
            <Descriptions.Item label="上次运动">
              <Space wrap>
                {moveSafe?.ok
                  ? <Tag color="green">成功 · TCP {fmtXYZ(moveSafe?.arm?.cartesian_xyzrpy)}</Tag>
                  : (moveSafe ? <Tag color="orange">{moveSafe.message || "未成功"}</Tag> : "尚未运动")}
                <ShowOnArmBtn
                  id="move_safe_tcp"
                  label="TCP@p_safe"
                  xyz={moveSafe?.arm?.cartesian_xyzrpy}
                  color={ARM_MARKER_COLORS.tcp}
                />
              </Space>
            </Descriptions.Item>
          </Descriptions>
          <Space wrap>
            <Button onClick={() => setCurrent(0)}>← 返回粗定位</Button>
            <Button type="primary" loading={busy} onClick={runStep2MoveSafe}>
              运动到安全点 p_safe
            </Button>
            {moveSafe?.ok && (
              <Button onClick={() => setCurrent(2)}>下一步：多视角精定位 →</Button>
            )}
          </Space>
          {moveSafe && (
            <pre className="mono" style={{ marginTop: 16, maxHeight: 220, overflow: "auto", background: "#fafafa", padding: 10 }}>
              {JSON.stringify(moveSafe, null, 2)}
            </pre>
          )}
        </>
      ))}

      {stepSection(2, (
        <>
          <Alert
            type="info"
            showIcon
            style={{ marginBottom: 16 }}
            message="第 3 步（M3）：腕部多视角采集 → 体素/平面融合 → c*；预抓取点 p_pregrasp = c* + [0,0,Δz]。请先在 M0 出流腕部 D405，并完成 Step2 到 p_safe。"
          />
          <Space wrap style={{ marginBottom: 16 }} align="center">
            <Text>检测方法</Text>
            <Select
              style={{ width: 120 }}
              value={method}
              onChange={setMethod}
              options={["auto", "yolo", "hsv", "mock"].map((m) => ({ value: m, label: m }))}
            />
            <Text>视点数 N</Text>
            <InputNumber min={3} max={9} step={1} value={mvN} onChange={(v) => setMvN(typeof v === "number" ? v : mvN)} />
            <Text>环半径</Text>
            <InputNumber
              min={0.01}
              max={0.1}
              step={0.005}
              value={mvRadius}
              addonAfter="m"
              onChange={(v) => setMvRadius(typeof v === "number" ? v : mvRadius)}
            />
            <Text>预抓取 Δz</Text>
            <InputNumber
              min={0.01}
              max={0.4}
              step={0.01}
              value={pregraspLift}
              addonAfter="m"
              onChange={(v) => setPregraspLift(typeof v === "number" ? v : pregraspLift)}
            />
            <Button type="primary" loading={busy} onClick={runStep3Multiview}>
              开始多视角精定位（M3）
            </Button>
            <Button loading={busy} onClick={runStep3FineServo}>单帧闭环（备用）</Button>
            <Button onClick={() => setCurrent(1)}>← 返回安全点</Button>
          </Space>
          <Row gutter={16}>
            <Col span={14}>
              <div className="preview-box">
                {fineViz
                  ? <img src={fineViz} alt="D405 fine viz" />
                  : <span style={{ color: "#888" }}>多视角运行后显示 D405 检测可视化</span>}
              </div>
            </Col>
            <Col span={10}>
              <Descriptions column={1} size="small" bordered>
                <Descriptions.Item label="腕部串号">{wristSerial || "未选定"}</Descriptions.Item>
                <Descriptions.Item label="模式">{fineServo?.mode === "multiview" ? <Tag color="blue">M3 多视角</Tag> : (fineServo ? <Tag>单帧闭环</Tag> : "—")}</Descriptions.Item>
                <Descriptions.Item label="结果">
                  {fineServo == null ? "未运行" : (fineServo.ok ? <Tag color="green">成功</Tag> : <Tag color="orange">失败</Tag>)}
                </Descriptions.Item>
                <Descriptions.Item label="c* 精定位点">
                  <Space wrap>
                    <span>{fmtXYZ(fineServo?.final_c_star)}</span>
                    <ShowOnArmBtn id="c_star" label="c*" xyz={fineServo?.final_c_star} color={ARM_MARKER_COLORS.c_star} />
                  </Space>
                </Descriptions.Item>
                <Descriptions.Item label="p_pregrasp 预抓取">
                  <Space wrap>
                    <Text strong type="success">{fmtXYZ(fineServo?.final_p_pregrasp)}</Text>
                    <ShowOnArmBtn id="p_pregrasp" label="p_pregrasp" xyz={fineServo?.final_p_pregrasp} color={ARM_MARKER_COLORS.p_pregrasp} />
                  </Space>
                </Descriptions.Item>
                <Descriptions.Item label="成功视点">{fineServo?.num_views_ok != null ? `${fineServo.num_views_ok}/${fineServo.num_views}` : (fineServo?.steps?.length ?? "—")}</Descriptions.Item>
                <Descriptions.Item label="融合点数">{fineServo?.fine?.num_points ?? fineServo?.voxel_stats?.num_voxels ?? "—"}</Descriptions.Item>
                <Descriptions.Item label="说明">{fineServo?.message || "—"}</Descriptions.Item>
                <Descriptions.Item label="视点">
                  <Space wrap>
                    <Button
                      size="small"
                      disabled={!(fineServo?.viewpoints || []).length}
                      onClick={() => {
                        const n = armSceneMarkers.upsertMany(
                          (fineServo?.viewpoints || []).filter((vp) => vp?.xyz).map((vp, i) => ({
                            id: `vp_${vp.name || i}`,
                            label: `视点 ${vp.name || i}`,
                            xyz: vp.xyz,
                            color: ARM_MARKER_COLORS.viewpoint,
                          }))
                        );
                        message.success(n ? `已标记 ${n} 个视点` : "无视点");
                      }}
                    >
                      显示全部视点
                    </Button>
                  </Space>
                </Descriptions.Item>
              </Descriptions>
            </Col>
          </Row>
          {!!(fineServo?.viewpoints || []).length && (
            <div style={{ marginTop: 12 }}>
              <Text type="secondary">多视角路点</Text>
              <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 6, maxWidth: 720 }}>
                {(fineServo.viewpoints || []).map((vp, i) => (
                  <Space key={vp.name || i} wrap>
                    <Tag>{vp.name || `vp_${i}`}</Tag>
                    <span className="mono">{fmtXYZ(vp.xyz)}</span>
                    <ShowOnArmBtn
                      id={`vp_${vp.name || i}`}
                      label={`视点 ${vp.name || i}`}
                      xyz={vp.xyz}
                      color={ARM_MARKER_COLORS.viewpoint}
                    />
                  </Space>
                ))}
              </div>
            </div>
          )}
          {fineServo?.steps && (
            <pre className="mono" style={{ marginTop: 16, maxHeight: 220, overflow: "auto", background: "#fafafa", padding: 10 }}>
              {JSON.stringify(fineServo.steps, null, 2)}
            </pre>
          )}
        </>
      ))}

      {stepSection(3, (
        <>
          <Alert
            type={fineServo?.ok ? "info" : "warning"}
            showIcon
            style={{ marginBottom: 16 }}
            message="第 4 步：抓取（占位）。标记吸盘开启；下降到 c* / 真实 IO 后续接入。"
            description={
              <div>
                <Space wrap style={{ marginBottom: 4 }}>
                  <span>c* = {fmtXYZ(fineServo?.final_c_star)}</span>
                  <ShowOnArmBtn id="c_star" label="c*" xyz={fineServo?.final_c_star} color={ARM_MARKER_COLORS.c_star} />
                </Space>
                <Space wrap>
                  <span>预抓取 p_pregrasp = {fmtXYZ(fineServo?.final_p_pregrasp)}</span>
                  <ShowOnArmBtn id="p_pregrasp" label="p_pregrasp" xyz={fineServo?.final_p_pregrasp} color={ARM_MARKER_COLORS.p_pregrasp} />
                </Space>
              </div>
            }
          />
          <Descriptions column={1} size="small" bordered style={{ marginBottom: 16, maxWidth: 640 }}>
            <Descriptions.Item label="精定位">{fineServo?.ok ? <Tag color="green">收敛</Tag> : <Tag color="orange">未完全收敛</Tag>}</Descriptions.Item>
            <Descriptions.Item label="抓取状态">
              {grasp?.ok
                ? <Tag color="blue">占位已触发 · {grasp?.message || "ok"}</Tag>
                : (grasp ? <Tag color="orange">{grasp.message || "未成功"}</Tag> : "尚未抓取")}
            </Descriptions.Item>
            <Descriptions.Item label="吸盘">{suctionOn ? <Tag color="green">ON（占位）</Tag> : <Tag>OFF</Tag>}</Descriptions.Item>
          </Descriptions>
          <Space wrap>
            <Button onClick={() => setCurrent(2)}>← 返回精定位</Button>
            <Button onClick={runStep3Multiview} loading={busy}>重新多视角精定位</Button>
            <Button type="primary" loading={busy} onClick={runStep4Grasp} disabled={!fineServo}>
              抓取（占位）
            </Button>
            {grasp?.ok && (
              <Button onClick={() => setCurrent(4)}>下一步：安全点列表 →</Button>
            )}
          </Space>
          {grasp && (
            <pre className="mono" style={{ marginTop: 16, maxHeight: 220, overflow: "auto", background: "#fafafa", padding: 10 }}>
              {JSON.stringify(grasp, null, 2)}
            </pre>
          )}
        </>
      ))}

      {stepSection(4, (
        <>
          <Alert
            type="info"
            showIcon
            style={{ marginBottom: 16 }}
            message="第 5 步：设置安全点列表，点击「下发依次运动」让机械臂按顺序走完。支持 xyz（保持姿态）/ home / 从当前 TCP 追加。"
          />
          <Space wrap style={{ marginBottom: 12 }}>
            <Button
              onClick={() => setSafePoints((prev) => [...prev, { name: `p${prev.length + 1}`, xyz: [0.4, 0.05, 0.5] }])}
            >
              添加 xyz 点
            </Button>
            <Button
              onClick={() => setSafePoints((prev) => [...prev, { name: `home_${prev.length + 1}`, home: true }])}
            >
              添加 Home
            </Button>
            <Button loading={busy} onClick={captureSafePoint}>追加当前 TCP</Button>
            <Button loading={busy} onClick={saveSafePoints}>保存列表</Button>
            <Button type="primary" loading={busy} onClick={runStep5SafeSequence} disabled={!safePoints.length}>
              下发依次运动
            </Button>
            <Button onClick={() => setCurrent(3)}>← 返回抓取</Button>
            {safeRun?.ok && <Button onClick={() => setCurrent(5)}>下一步：放开吸盘 →</Button>}
          </Space>
          <div style={{ display: "flex", flexDirection: "column", gap: 8, maxWidth: 820 }}>
            {safePoints.map((pt, idx) => (
              <Space key={idx} wrap align="center" style={{ background: "#fafafa", padding: "8px 10px", borderRadius: 6 }}>
                <Text type="secondary">#{idx + 1}</Text>
                <Input
                  style={{ width: 120 }}
                  value={pt.name || ""}
                  onChange={(e) => updateSafePoint(idx, { name: e.target.value })}
                  placeholder="名称"
                />
                {pt.home ? (
                  <Tag color="purple">HOME</Tag>
                ) : (
                  <>
                    <Text>x</Text>
                    <InputNumber
                      step={0.01}
                      value={Number((pt.xyz || [0, 0, 0])[0])}
                      onChange={(v) => {
                        const xyz = [...(pt.xyz || [0, 0, 0])];
                        xyz[0] = typeof v === "number" ? v : xyz[0];
                        updateSafePoint(idx, { xyz, home: undefined, joints_rad: undefined });
                      }}
                    />
                    <Text>y</Text>
                    <InputNumber
                      step={0.01}
                      value={Number((pt.xyz || [0, 0, 0])[1])}
                      onChange={(v) => {
                        const xyz = [...(pt.xyz || [0, 0, 0])];
                        xyz[1] = typeof v === "number" ? v : xyz[1];
                        updateSafePoint(idx, { xyz, home: undefined, joints_rad: undefined });
                      }}
                    />
                    <Text>z</Text>
                    <InputNumber
                      step={0.01}
                      value={Number((pt.xyz || [0, 0, 0])[2])}
                      onChange={(v) => {
                        const xyz = [...(pt.xyz || [0, 0, 0])];
                        xyz[2] = typeof v === "number" ? v : xyz[2];
                        updateSafePoint(idx, { xyz, home: undefined, joints_rad: undefined });
                      }}
                    />
                  </>
                )}
                <Button
                  size="small"
                  disabled={idx === 0}
                  onClick={() => setSafePoints((prev) => {
                    const next = [...prev];
                    [next[idx - 1], next[idx]] = [next[idx], next[idx - 1]];
                    return next;
                  })}
                >↑</Button>
                <Button
                  size="small"
                  disabled={idx >= safePoints.length - 1}
                  onClick={() => setSafePoints((prev) => {
                    const next = [...prev];
                    [next[idx + 1], next[idx]] = [next[idx], next[idx + 1]];
                    return next;
                  })}
                >↓</Button>
                <Button
                  size="small"
                  danger
                  onClick={() => setSafePoints((prev) => prev.filter((_, i) => i !== idx))}
                >删</Button>
                {!pt.home && (
                  <ShowOnArmBtn
                    id={`safe_${pt.name || idx}`}
                    label={pt.name || `safe_${idx}`}
                    xyz={pt.xyz}
                    color={ARM_MARKER_COLORS.safe_point}
                  />
                )}
              </Space>
            ))}
            {!safePoints.length && <Text type="secondary">列表为空，请添加安全点</Text>}
          </div>
          <Space wrap style={{ marginTop: 10 }}>
            <Button
              size="small"
              onClick={() => {
                const n = armSceneMarkers.upsertMany(
                  safePoints.filter((pt) => !pt.home && pt.xyz).map((pt, i) => ({
                    id: `safe_${pt.name || i}`,
                    label: pt.name || `safe_${i}`,
                    xyz: pt.xyz,
                    color: ARM_MARKER_COLORS.safe_point,
                  }))
                );
                message.success(n ? `已标记 ${n} 个安全点` : "无可显示的 xyz 安全点");
              }}
            >
              显示全部安全点
            </Button>
          </Space>
          {safeRun && (
            <pre className="mono" style={{ marginTop: 16, maxHeight: 240, overflow: "auto", background: "#fafafa", padding: 10 }}>
              {JSON.stringify(safeRun, null, 2)}
            </pre>
          )}
        </>
      ))}

      {stepSection(5, (
        <>
          <Alert
            type={suctionOn ? "warning" : "success"}
            showIcon
            style={{ marginBottom: 16 }}
            message="最后一步：放开吸盘。当前为占位 IO（无真实真空阀），点击后状态变为 OFF。"
          />
          <Descriptions column={1} size="small" bordered style={{ marginBottom: 16, maxWidth: 640 }}>
            <Descriptions.Item label="吸盘状态">
              {suctionOn ? <Tag color="green">ON（占位）</Tag> : <Tag>OFF</Tag>}
            </Descriptions.Item>
            <Descriptions.Item label="安全点下发">
              {safeRun?.ok
                ? <Tag color="green">{safeRun.message}</Tag>
                : (safeRun ? <Tag color="orange">{safeRun.message}</Tag> : "—")}
            </Descriptions.Item>
            <Descriptions.Item label="放开结果">
              {suctionRelease?.ok
                ? <Tag color="blue">{suctionRelease.message}</Tag>
                : "尚未放开"}
            </Descriptions.Item>
          </Descriptions>
          <Space wrap>
            <Button onClick={() => setCurrent(4)}>← 返回安全点</Button>
            <Button type="primary" danger loading={busy} onClick={runStep6Release}>
              放开吸盘
            </Button>
          </Space>
          {suctionRelease && (
            <pre className="mono" style={{ marginTop: 16, maxHeight: 220, overflow: "auto", background: "#fafafa", padding: 10 }}>
              {JSON.stringify(suctionRelease, null, 2)}
            </pre>
          )}
        </>
      ))}
    </Card>
  );
}

