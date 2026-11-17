import { useEffect, useState, useCallback, useRef } from 'react';
import { Card, Button, Space, Tag, Alert, Col, Descriptions, Image, Input, InputNumber, Progress, Row, Select, Switch, Table, Statistic, Typography, Upload, message } from 'antd';
import { api, API } from '../lib/api.js';

const { Text } = Typography;

function fmtMetric(v) {
  return v == null ? '—' : Number(v).toFixed(3);
}

export default function TrainPanel({ active }) {
  const [status, setStatus] = useState(null);
  const [epochs, setEpochs] = useState(50);
  const [imgsz, setImgsz] = useState(640);
  const [batch, setBatch] = useState(8);
  const [device, setDevice] = useState('cpu');
  const [model, setModel] = useState('models/yolo11n-seg.pt');
  const [busy, setBusy] = useState(false);
  const [verifySource, setVerifySource] = useState('installed');
  const [verifyConf, setVerifyConf] = useState(0.15);
  const [verifyImgsz, setVerifyImgsz] = useState(640);
  const [verifyDevice, setVerifyDevice] = useState('cpu');
  const [verifyFilter, setVerifyFilter] = useState(false);
  const [verifyClassFilter, setVerifyClassFilter] = useState('cardboard');
  const [verifyBusy, setVerifyBusy] = useState(false);
  const [verifyResult, setVerifyResult] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const logRef = useRef(null);
  const seededRef = useRef(false);
  const verifySeededRef = useRef(false);

  const refresh = useCallback(async () => {
    const st = await api('/api/train/yolo/status');
    setStatus(st);
    if (!seededRef.current && st?.defaults) {
      seededRef.current = true;
      const d = st.defaults;
      if (d.epochs != null) setEpochs(Number(d.epochs));
      if (d.imgsz != null) setImgsz(Number(d.imgsz));
      if (d.batch != null) setBatch(Number(d.batch));
      if (d.device) setDevice(String(d.device));
      if (d.model) setModel(String(d.model));
    }
    if (!verifySeededRef.current && st?.verify?.defaults) {
      verifySeededRef.current = true;
      const vd = st.verify.defaults;
      if (vd.conf != null) setVerifyConf(Number(vd.conf));
      if (vd.imgsz != null) setVerifyImgsz(Number(vd.imgsz));
      if (vd.device) setVerifyDevice(String(vd.device));
      if (Array.isArray(vd.class_filter) && vd.class_filter.length) {
        setVerifyClassFilter(vd.class_filter.join(','));
      }
      const choices = st.verify.weight_choices || [];
      const preferred = choices.find((c) => c.value === 'installed' && c.exists)
        || choices.find((c) => c.value === 'best' && c.exists)
        || choices.find((c) => c.exists);
      if (preferred?.value) setVerifySource(preferred.value);
    }
    return st;
  }, []);

  useEffect(() => () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  useEffect(() => {
    if (!active) return undefined;
    let cancelled = false;
    const tick = () => refresh().catch((e) => {
      if (!cancelled && !String(e.message || e).includes('Failed to fetch')) {
        /* keep quiet on poll errors */
      }
    });
    tick();
    const t = setInterval(tick, 2000);
    return () => { cancelled = true; clearInterval(t); };
  }, [active, refresh]);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [status?.log_tail]);

  const startTrain = async () => {
    setBusy(true);
    try {
      const st = await api('/api/train/yolo/start', {
        method: 'POST',
        body: JSON.stringify({
          epochs,
          imgsz,
          batch,
          device,
          model,
          task: 'segment',
          data: 'data/yolo_seg/dataset.yaml',
          project: 'runs/segment',
          name: 'cardboard_seg',
        }),
      });
      setStatus(st);
      message.success('YOLO11-seg 训练已启动');
    } catch (e) {
      message.error(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const stopTrain = async () => {
    setBusy(true);
    try {
      const st = await api('/api/train/yolo/stop', { method: 'POST', body: '{}' });
      setStatus(st);
      message.info('已请求停止训练');
    } catch (e) {
      message.error(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const installWeights = async () => {
    setBusy(true);
    try {
      const st = await api('/api/train/yolo/install', { method: 'POST', body: '{}' });
      setStatus(st);
      message.success(`已安装权重 → ${st.installed_to || 'models/cardboard_yolo.pt'}`);
    } catch (e) {
      message.error(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const runVerifyUpload = async (file) => {
    setVerifyBusy(true);
    try {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl(URL.createObjectURL(file));
      const fd = new FormData();
      fd.append('file', file);
      fd.append('weights_source', verifySource);
      fd.append('conf', String(verifyConf));
      fd.append('imgsz', String(verifyImgsz));
      fd.append('device', verifyDevice);
      fd.append('apply_class_filter', verifyFilter ? 'true' : 'false');
      if (verifyFilter && verifyClassFilter.trim()) {
        fd.append('class_filter', verifyClassFilter.trim());
      }
      fd.append('keep_top_k', '0');
      const res = await fetch(`${API}/api/train/yolo/predict`, { method: 'POST', body: fd });
      if (!res.ok) {
        let detail = await res.text();
        try { detail = JSON.parse(detail).detail || detail; } catch (_) { /* keep text */ }
        throw new Error(detail || res.statusText);
      }
      const result = await res.json();
      setVerifyResult(result);
      message.success(result.count > 0 ? `检出 ${result.count} 个目标` : '未检出目标');
    } catch (e) {
      message.error(String(e.message || e));
    } finally {
      setVerifyBusy(false);
    }
    return false;
  };

  const ds = status?.dataset || {};
  const w = status?.weights || {};
  const m = status?.metrics || {};
  const running = !!status?.running;
  const state = status?.state || 'idle';
  const stateColor = ({
    idle: 'default',
    running: 'processing',
    succeeded: 'success',
    failed: 'error',
    stopped: 'warning',
  })[state] || 'default';
  const history = Array.isArray(m.history) ? m.history : [];
  const classNames = ds.names && typeof ds.names === 'object'
    ? Object.entries(ds.names).map(([k, v]) => `${k}:${v}`).join(', ')
    : '—';
  const bootstrapName = (w.bootstrap || 'models/yolo11n-seg.pt').split('/').pop();
  const installedName = (w.installed || 'models/cardboard_yolo.pt').split('/').pop();
  const metricCols = [
    { title: 'epoch', dataIndex: 'epoch', width: 70 },
    { title: 'mAP50(M)', dataIndex: 'metrics/mAP50(M)', width: 95, render: fmtMetric },
    { title: 'mAP50-95(M)', dataIndex: 'metrics/mAP50-95(M)', width: 110, render: fmtMetric },
    { title: 'mAP50(B)', dataIndex: 'metrics/mAP50(B)', width: 95, render: fmtMetric },
    { title: 'seg_loss', dataIndex: 'train/seg_loss', width: 90, render: fmtMetric },
    { title: 'box_loss', dataIndex: 'train/box_loss', width: 90, render: fmtMetric },
  ];
  const detectCols = [
    { title: '#', width: 48, render: (_v, _r, i) => i + 1 },
    { title: 'class', dataIndex: 'class_name', width: 120 },
    { title: 'score', dataIndex: 'score', width: 90, render: fmtMetric },
    {
      title: 'polygon',
      dataIndex: 'polygon_points',
      width: 90,
      render: (n, row) => (row?.has_polygon ? `${n} pts` : '—'),
    },
    {
      title: 'box xyxy',
      dataIndex: 'box',
      render: (box) => (Array.isArray(box) ? box.map((v) => Number(v).toFixed(1)).join(', ') : '—'),
    },
  ];
  const imageArts = (status?.artifacts || []).filter((a) => a.url);
  const weightChoices = (status?.verify?.weight_choices || []).map((c) => ({
    value: c.value,
    label: c.exists ? c.label : `${c.label}（缺失）`,
    disabled: !c.exists,
  }));
  const selectedWeight = (status?.verify?.weight_choices || []).find((c) => c.value === verifySource);
  return (
    <Card
      title="YOLO11-seg 训练（instance segmentation）"
      extra={(
        <Space wrap>
          <Tag color={stateColor}>{state}</Tag>
          <Tag color="blue">task=segment</Tag>
          {w.m2_ready
            ? <Tag color="green">M2 权重就绪</Tag>
            : <Tag color="orange">M2 尚无 cardboard_yolo.pt</Tag>}
          <Button size="small" onClick={() => refresh().catch((e) => message.error(String(e.message || e)))}>刷新</Button>
        </Space>
      )}
    >
      <Alert
        type={w.m2_ready ? 'success' : 'warning'}
        showIcon
        style={{ marginBottom: 16 }}
        message={
          w.m2_ready
            ? `M2 将使用 ${w.config_weights || 'models/cardboard_yolo.pt'}（已存在）。可在此用 yolo_seg 重训并覆盖安装。`
            : '数据集已就绪：data/yolo_seg（polygon 分割）。以 yolo11n-seg 训练后自动安装 best.pt → models/cardboard_yolo.pt 供 M2 使用（推理优先用 mask）。'
        }
      />

      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col xs={24} md={8}>
          <Card size="small" title="数据集 · yolo_seg">
            <Descriptions column={1} size="small">
              <Descriptions.Item label="task">{ds.task || 'segment'}</Descriptions.Item>
              <Descriptions.Item label="classes">{classNames}</Descriptions.Item>
              <Descriptions.Item label="train 图">{ds.train_images ?? '—'}</Descriptions.Item>
              <Descriptions.Item label="train 标注">{ds.train_labels ?? '—'}</Descriptions.Item>
              <Descriptions.Item label="val 图">{ds.val_images ?? '—'}</Descriptions.Item>
              <Descriptions.Item label="yaml">{ds.data_yaml || 'data/yolo_seg/dataset.yaml'}</Descriptions.Item>
            </Descriptions>
            {!ds.ready && (
              <Alert type="error" showIcon message={ds.hint || '数据集未就绪'} style={{ marginTop: 8 }} />
            )}
            {ds.ready && Number(ds.val_images || 0) < 1 && (
              <Alert type="info" showIcon style={{ marginTop: 8 }} message="val 为空时 Ultralytics 通常会用 train 做校验。" />
            )}
          </Card>
        </Col>
        <Col xs={24} md={8}>
          <Card size="small" title="权重">
            <Descriptions column={1} size="small">
              <Descriptions.Item label="基座">
                {w.bootstrap_exists ? <Tag color="green">{bootstrapName}</Tag> : <Tag color="red">缺失 {bootstrapName}</Tag>}
              </Descriptions.Item>
              <Descriptions.Item label="安装目标">
                {w.installed_exists ? <Tag color="green">{installedName}</Tag> : <Tag>未安装</Tag>}
              </Descriptions.Item>
              <Descriptions.Item label="配置路径">{w.config_weights || '—'}</Descriptions.Item>
            </Descriptions>
          </Card>
        </Col>
        <Col xs={24} md={8}>
          <Card size="small" title="进度">
            <Progress
              percent={Number(m.percent || 0)}
              status={running ? 'active' : (state === 'failed' ? 'exception' : (state === 'succeeded' ? 'success' : 'normal'))}
            />
            <Space size="large" style={{ marginTop: 8 }} wrap>
              <Statistic title="Epoch" value={`${m.epochs_done || 0} / ${m.epochs_total || epochs}`} />
              <Statistic title="mAP50(M)" value={fmtMetric(m.map50_mask ?? m.map50)} />
              <Statistic title="mAP50(B)" value={fmtMetric(m.map50_box)} />
            </Space>
          </Card>
        </Col>
      </Row>

      <Card size="small" title="训练参数" style={{ marginBottom: 16 }}>
        <Space wrap align="center">
          <Text>epochs</Text>
          <InputNumber min={1} max={1000} value={epochs} disabled={running} onChange={(v) => setEpochs(typeof v === 'number' ? v : epochs)} />
          <Text>imgsz</Text>
          <InputNumber min={32} max={2048} step={32} value={imgsz} disabled={running} onChange={(v) => setImgsz(typeof v === 'number' ? v : imgsz)} />
          <Text>batch</Text>
          <InputNumber min={1} max={256} value={batch} disabled={running} onChange={(v) => setBatch(typeof v === 'number' ? v : batch)} />
          <Text>device</Text>
          <Select
            style={{ width: 100 }}
            value={device}
            disabled={running}
            onChange={setDevice}
            options={['cpu', '0', '0,1'].map((d) => ({ value: d, label: d }))}
          />
          <Text>model</Text>
          <Input style={{ width: 240 }} value={model} disabled={running} onChange={(e) => setModel(e.target.value)} />
          <Button type="primary" loading={busy} disabled={running || !ds.ready} onClick={startTrain}>开始训练</Button>
          <Button danger loading={busy} disabled={!running} onClick={stopTrain}>停止</Button>
          <Button loading={busy} disabled={running} onClick={installWeights}>安装 best → M2</Button>
        </Space>
        <Text type="secondary" style={{ display: 'block', marginTop: 8, fontSize: 12 }}>
          等价于 yolo segment train（data/yolo_seg + yolo11n-seg）。成功后自动复制 best.pt → models/cardboard_yolo.pt；M2 class_filter 仍可只保留 cardboard。
        </Text>
        {status?.error && (
          <Alert type="error" showIcon style={{ marginTop: 12 }} message={status.error} />
        )}
      </Card>

      <Card
        size="small"
        title="手动验证 · 上传图片检验模型"
        style={{ marginBottom: 16 }}
        extra={verifyResult ? <Tag color={verifyResult.count > 0 ? 'green' : 'default'}>hits={verifyResult.count}</Tag> : null}
      >
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 12 }}
          message="上传 RGB 图（jpg/png）对选定权重做一次 YOLO-seg 推理。分割结果以多边形轮廓+半透明填充显示（细矩形仅为 AABB 辅助）。默认可看全部类别；可开启与 M2 相同的 class_filter。"
        />
        <Space wrap align="center" style={{ marginBottom: 12 }}>
          <Text>权重</Text>
          <Select
            style={{ minWidth: 260 }}
            value={verifySource}
            onChange={setVerifySource}
            options={weightChoices.length ? weightChoices : [
              { value: 'installed', label: 'M2 已安装' },
              { value: 'best', label: '本次训练 best.pt' },
              { value: 'bootstrap', label: '基座' },
            ]}
          />
          <Text>conf</Text>
          <InputNumber
            min={0.01}
            max={1}
            step={0.05}
            value={verifyConf}
            onChange={(v) => setVerifyConf(typeof v === 'number' ? v : verifyConf)}
          />
          <Text>imgsz</Text>
          <InputNumber
            min={32}
            max={2048}
            step={32}
            value={verifyImgsz}
            onChange={(v) => setVerifyImgsz(typeof v === 'number' ? v : verifyImgsz)}
          />
          <Text>device</Text>
          <Select
            style={{ width: 100 }}
            value={verifyDevice}
            onChange={setVerifyDevice}
            options={['cpu', '0', '0,1'].map((d) => ({ value: d, label: d }))}
          />
          <Text>class_filter</Text>
          <Switch checked={verifyFilter} onChange={setVerifyFilter} checkedChildren="开" unCheckedChildren="关" />
          <Input
            style={{ width: 160 }}
            disabled={!verifyFilter}
            value={verifyClassFilter}
            onChange={(e) => setVerifyClassFilter(e.target.value)}
            placeholder="cardboard"
          />
          <Upload
            accept="image/*,.jpg,.jpeg,.png,.bmp,.webp"
            showUploadList={false}
            beforeUpload={runVerifyUpload}
            disabled={verifyBusy || (selectedWeight && selectedWeight.exists === false)}
          >
            <Button type="primary" loading={verifyBusy} disabled={selectedWeight && selectedWeight.exists === false}>
              上传并推理
            </Button>
          </Upload>
        </Space>
        {selectedWeight && (
          <Text type="secondary" style={{ display: 'block', marginBottom: 12, fontSize: 12 }}>
            当前权重路径：{selectedWeight.path || '—'}{selectedWeight.exists ? '' : '（文件不存在，请先训练或安装）'}
          </Text>
        )}
        <Row gutter={16}>
          <Col xs={24} md={12}>
            <div className="train-verify-pair">
              <div className="train-verify-frame">
                {previewUrl
                  ? <img src={previewUrl} alt="upload preview" />
                  : <div className="train-verify-empty">原图预览</div>}
                <div className="cap">上传原图</div>
              </div>
            </div>
          </Col>
          <Col xs={24} md={12}>
            <div className="train-verify-pair">
              <div className="train-verify-frame">
                {verifyResult?.viz_url
                  ? <img src={verifyResult.viz_url} alt="yolo viz" />
                  : <div className="train-verify-empty">推理叠加结果</div>}
                <div className="cap">
                  {verifyResult
                    ? `${verifyResult.message || 'done'} · ${verifyResult.weights || ''}`
                    : '推理叠加结果'}
                </div>
              </div>
            </div>
          </Col>
        </Row>
        {verifyResult && (
          <Table
            size="small"
            style={{ marginTop: 12 }}
            pagination={false}
            rowKey={(_r, i) => String(i)}
            columns={detectCols}
            dataSource={verifyResult.detections || []}
            locale={{ emptyText: '无检出' }}
            scroll={{ x: 480 }}
          />
        )}
      </Card>

      <Row gutter={16}>
        <Col xs={24} lg={12}>
          <Card size="small" title="训练日志" style={{ marginBottom: 16 }}>
            <div className="train-log" ref={logRef}>
              {(status?.log_tail || []).join('\n') || '（尚无日志）'}
            </div>
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card size="small" title="指标历史（Mask / Box）" style={{ marginBottom: 16 }}>
            <Table
              size="small"
              pagination={{ pageSize: 8, hideOnSinglePage: true }}
              rowKey={(r, i) => String(r.epoch ?? i)}
              columns={metricCols}
              dataSource={history.slice().reverse()}
              scroll={{ x: 560 }}
            />
          </Card>
        </Col>
      </Row>

      <Card size="small" title={`结果可视化 · ${imageArts.length}`}>
        {imageArts.length === 0 ? (
          <Text type="secondary">训练产生 results.png / MaskPR 曲线等后会显示在这里。点击缩略图可放大查看。</Text>
        ) : (
          <Image.PreviewGroup>
            <div className="train-artifact-grid">
              {imageArts.map((a) => {
                const src = `${a.url}&t=${encodeURIComponent(a.mtime || '')}`;
                return (
                  <div className="train-artifact-card" key={a.path}>
                    <Image
                      src={src}
                      alt={a.name}
                      className="train-artifact-thumb"
                      rootClassName="train-artifact-image"
                    />
                    <div className="cap">{a.name}</div>
                  </div>
                );
              })}
            </div>
          </Image.PreviewGroup>
        )}
      </Card>
    </Card>
  );
}
