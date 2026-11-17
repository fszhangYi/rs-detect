import { useEffect, useState, useCallback, useRef } from 'react';
import { Card, Row, Col, Button, Space, Tag, Alert, Input, Upload, Typography, message } from 'antd';
import { api } from '../lib/api.js';
import { IDENTITY, cloneMat4 } from '../lib/matrix.js';
import { parseExtrinsicsJson, parseExtrinsicsCsv, downloadTextFile, extrinsicsJsonTemplate, extrinsicsCsvTemplate } from '../lib/extrinsicsParse.js';
import MatrixGrid from './MatrixGrid.jsx';

const { Text } = Typography;
const { TextArea } = Input;

export default function ExtrinsicsPanel({ selection }) {
  const [d405T, setD405T] = useState(IDENTITY.map(r => r.slice()));
  const [d435T, setD435T] = useState(IDENTITY.map(r => r.slice()));
  const [notes, setNotes] = useState("");
  const [importHint, setImportHint] = useState("");

  const load = async () => {
    const data = await api("/api/extrinsics");
    setD405T((data.d405?.T || IDENTITY).map(r => r.slice()));
    setD435T((data.d435?.T || IDENTITY).map(r => r.slice()));
    setNotes(data.notes || "");
  };
  useEffect(() => { load().catch(e => message.error(String(e.message || e))); }, []);

  const applyParsed = (parsed, sourceLabel) => {
    const parts = [];
    if (parsed.d405_T) {
      setD405T(cloneMat4(parsed.d405_T));
      parts.push("D405");
    }
    if (parsed.d435_T) {
      setD435T(cloneMat4(parsed.d435_T));
      parts.push("D435");
    }
    if (parsed.notes != null) setNotes(String(parsed.notes));
    const tip = `已从 ${sourceLabel} 回填 ${parts.join(" + ") || "无矩阵"}（尚未写入磁盘，请点保存）`;
    setImportHint(tip);
    message.success(tip);
  };

  const onUploadFile = async (file) => {
    try {
      const name = String(file.name || "").toLowerCase();
      const text = await file.text();
      let parsed;
      if (name.endsWith(".json") || text.trim().startsWith("{")) {
        parsed = parseExtrinsicsJson(text);
      } else if (name.endsWith(".csv") || name.endsWith(".txt")) {
        parsed = parseExtrinsicsCsv(text);
      } else {
        // sniff
        try { parsed = parseExtrinsicsJson(text); }
        catch (_) { parsed = parseExtrinsicsCsv(text); }
      }
      applyParsed(parsed, file.name || "上传文件");
    } catch (e) {
      message.error(String(e.message || e));
    }
    return false; // prevent antd auto upload
  };

  const downloadJsonTemplate = () => {
    const payload = extrinsicsJsonTemplate(selection);
    downloadTextFile(
      "extrinsics_template.json",
      JSON.stringify(payload, null, 2) + "\n",
      "application/json;charset=utf-8",
    );
    message.success("已下载 JSON 模板");
  };

  const downloadCsvTemplate = () => {
    downloadTextFile(
      "extrinsics_template.csv",
      extrinsicsCsvTemplate(),
      "text/csv;charset=utf-8",
    );
    message.success("已下载 CSV 模板");
  };

  const exportCurrentJson = () => {
    const payload = extrinsicsJsonTemplate(selection);
    payload.d405.T = cloneMat4(d405T);
    payload.d435.T = cloneMat4(d435T);
    payload.notes = notes || payload.notes;
    downloadTextFile(
      "extrinsics_export.json",
      JSON.stringify(payload, null, 2) + "\n",
      "application/json;charset=utf-8",
    );
  };

  const save = async () => {
    try {
      const d405_serial = selection?.d405_serial || selection?.wrist || "";
      const d435_serial = selection?.d435_serial || selection?.scene || "";
      if (!d405_serial || !d435_serial) {
        throw new Error("请先在 M0 选定腕部/场景相机，序列号会自动联动到此处");
      }
      await api("/api/extrinsics", {
        method: "PUT",
        body: JSON.stringify({
          d405_serial,
          d435_serial,
          d405_T: d405T,
          d435_T: d435T,
          notes,
        }),
      });
      message.success("外参已保存");
      setImportHint("");
    } catch (e) { message.error(String(e.message || e)); }
  };

  return (
    <Card
      title="外参（M1 · 用户输入）"
      extra={
        <Space wrap>
          <Button onClick={() => { setD405T(IDENTITY.map(r => r.slice())); setD435T(IDENTITY.map(r => r.slice())); }}>
            复位单位阵
          </Button>
          <Button onClick={() => load().catch(e => message.error(String(e.message || e)))}>重新加载</Button>
        </Space>
      }
    >
      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        message="序列号由 M0 选定相机自动联动（只读）。可手动填 4×4，或上传 JSON / CSV 解析回填；模板可下载。"
      />

      <Card size="small" title="导入 / 模板" style={{ marginBottom: 16 }}>
        <Space wrap style={{ marginBottom: 8 }}>
          <Upload
            accept=".json,.csv,.txt,application/json,text/csv,text/plain"
            showUploadList={false}
            beforeUpload={onUploadFile}
          >
            <Button type="primary">上传 JSON / CSV</Button>
          </Upload>
          <Button onClick={downloadJsonTemplate}>下载 JSON 模板</Button>
          <Button onClick={downloadCsvTemplate}>下载 CSV 模板</Button>
          <Button onClick={exportCurrentJson}>导出当前为 JSON</Button>
        </Space>
        <Text type="secondary" style={{ display: "block", fontSize: 12 }}>
          JSON：完整结构 d405.T / d435.T，或扁平字段 d405_T·d435_T，或单矩阵 which+T。
          CSV：推荐「camera + 16 列」；也支持「# d405 / # d435」分块 4×4。
        </Text>
        {!!importHint && (
          <Alert type="success" showIcon style={{ marginTop: 10 }} message={importHint} />
        )}
      </Card>

      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={12}>
          <Text type="secondary">D405 序列号（腕部）</Text>
          <Input value={selection?.d405_serial || selection?.wrist || ""} readOnly placeholder="请在 M0 选择腕部相机" />
        </Col>
        <Col span={12}>
          <Text type="secondary">D435 序列号（场景）</Text>
          <Input value={selection?.d435_serial || selection?.scene || ""} readOnly placeholder="请在 M0 选择场景相机" />
        </Col>
      </Row>
      <Row gutter={16}>
        <Col span={12}>
          <Text strong>D405 T_flange_cam</Text>
          <div style={{ marginTop: 8 }}><MatrixGrid value={d405T} onChange={setD405T} /></div>
        </Col>
        <Col span={12}>
          <Text strong>D435 T_base_cam</Text>
          <div style={{ marginTop: 8 }}><MatrixGrid value={d435T} onChange={setD435T} /></div>
        </Col>
      </Row>
      <div style={{ marginTop: 16 }}>
        <Text type="secondary">备注</Text>
        <TextArea rows={2} value={notes} onChange={e => setNotes(e.target.value)} />
      </div>
      <Button type="primary" style={{ marginTop: 16 }} onClick={save}>保存外参</Button>
    </Card>
  );
}

