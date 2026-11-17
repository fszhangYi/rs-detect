import { Button, message } from 'antd';
import { armSceneMarkers, ARM_MARKER_COLORS } from '../lib/armMarkers.js';

export default function ShowOnArmBtn({ id, label, xyz, color, size = "small" }) {
  const ready = xyz && xyz.length >= 3 && [0, 1, 2].every((i) => Number.isFinite(Number(xyz[i])));
  return (
    <Button
      size={size}
      disabled={!ready}
      onClick={() => {
        const ok = armSceneMarkers.upsert({
          id: id || label,
          label: label || id,
          xyz,
          color: color != null ? color : ARM_MARKER_COLORS.default,
        });
        if (!ok) {
          message.warning("点位无效，无法显示");
          return;
        }
        message.success(`已标记「${label || id}」到机械臂画布`);
      }}
    >
      显示
    </Button>
  );
}

