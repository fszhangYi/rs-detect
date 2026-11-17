import { useEffect, useState, useCallback, useRef } from 'react';
import { Modal, Form, Input, Button, Space, Alert, Typography, message } from 'antd';
import { api } from '../lib/api.js';

const { Text } = Typography;

export default function SettingsModal({ open, onClose }) {
  const [capturesDir, setCapturesDir] = useState("");
  const [recordingsDir, setRecordingsDir] = useState("");
  const [defaults, setDefaults] = useState({ captures_dir: "", recordings_dir: "" });
  const [resolved, setResolved] = useState({ captures_dir: "", recordings_dir: "" });
  const [busy, setBusy] = useState(false);

  const load = async () => {
    try {
      const s = await api("/api/settings/storage");
      setCapturesDir(s.captures_dir_display || s.captures_dir || "");
      setRecordingsDir(s.recordings_dir_display || s.recordings_dir || "");
      setDefaults(s.defaults || {});
      setResolved({ captures_dir: s.captures_dir || "", recordings_dir: s.recordings_dir || "" });
    } catch (e) {
      message.error(String(e.message || e));
    }
  };

  useEffect(() => {
    if (open) load();
  }, [open]);

  const save = async () => {
    if (!String(capturesDir || "").trim() || !String(recordingsDir || "").trim()) {
      message.warning("照片路径与视频路径均不能为空");
      return;
    }
    setBusy(true);
    try {
      const s = await api("/api/settings/storage", {
        method: "PUT",
        body: JSON.stringify({
          captures_dir: String(capturesDir).trim(),
          recordings_dir: String(recordingsDir).trim(),
        }),
      });
      setCapturesDir(s.captures_dir_display || s.captures_dir || "");
      setRecordingsDir(s.recordings_dir_display || s.recordings_dir || "");
      setResolved({ captures_dir: s.captures_dir || "", recordings_dir: s.recordings_dir || "" });
      message.success("保存路径已更新");
      onClose?.();
    } catch (e) {
      message.error(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const resetDefaults = () => {
    setCapturesDir(defaults.captures_dir || "data/captures");
    setRecordingsDir(defaults.recordings_dir || "data/recordings");
  };

  return (
    <Modal
      title="设置 · M1 媒体保存路径"
      open={open}
      onCancel={onClose}
      onOk={save}
      okText="保存"
      cancelText="取消"
      confirmLoading={busy}
      destroyOnClose
      width={640}
    >
      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        message="拍照与录像文件的保存目录。可用相对路径（相对仓库根）或绝对路径；目录不存在时会自动创建。"
      />
      <Form layout="vertical">
        <Form.Item label="照片保存路径（snapshot / PNG + 点云 PLY）" required>
          <Input
            value={capturesDir}
            onChange={(e) => setCapturesDir(e.target.value)}
            placeholder="data/captures"
          />
          {resolved.captures_dir && (
            <Text type="secondary" style={{ fontSize: 12 }}>当前解析：{resolved.captures_dir}</Text>
          )}
        </Form.Item>
        <Form.Item label="视频保存路径（录像 / MP4）" required>
          <Input
            value={recordingsDir}
            onChange={(e) => setRecordingsDir(e.target.value)}
            placeholder="data/recordings"
          />
          {resolved.recordings_dir && (
            <Text type="secondary" style={{ fontSize: 12 }}>当前解析：{resolved.recordings_dir}</Text>
          )}
        </Form.Item>
      </Form>
      <Button onClick={resetDefaults}>恢复默认</Button>
    </Modal>
  );
}

