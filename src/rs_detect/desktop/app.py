from __future__ import annotations

import logging
import sys
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from qfluentwidgets import (
    BodyLabel,
    ComboBox,
    FluentIcon,
    FluentWindow,
    InfoBar,
    InfoBarPosition,
    LineEdit,
    NavigationItemPosition,
    PrimaryPushButton,
    PushButton,
    SubtitleLabel,
    TextEdit,
)

from rs_detect.camera.recorder import colorize_depth
from rs_detect.service import DetectService, get_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rs_detect.desktop")


def numpy_bgr_to_qpixmap(img: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class CameraLaneWidget(QWidget):
    """One lane: wrist (D405) or scene (D435)."""

    def __init__(self, title: str, role: str, service: DetectService, parent=None):
        super().__init__(parent)
        self.role = role  # wrist | scene
        self.service = service
        self.serial: Optional[str] = None
        self._recording = False

        layout = QVBoxLayout(self)
        layout.addWidget(SubtitleLabel(title))

        row = QHBoxLayout()
        self.combo = ComboBox(self)
        self.combo.setMinimumWidth(280)
        row.addWidget(self.combo, 1)
        self.btn_refresh = PushButton(FluentIcon.SYNC, "刷新扫描")
        self.btn_start = PrimaryPushButton(FluentIcon.PLAY, "开始出流")
        self.btn_stop = PushButton(FluentIcon.PAUSE, "停止")
        row.addWidget(self.btn_refresh)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)
        layout.addLayout(row)

        self.preview = QLabel("无画面")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(480, 360)
        self.preview.setStyleSheet("background:#1e1e1e;color:#aaa;border-radius:8px;")
        layout.addWidget(self.preview, 1)

        acts = QHBoxLayout()
        self.btn_snap = PushButton(FluentIcon.CAMERA, "拍照")
        self.btn_rec = PushButton(FluentIcon.VIDEO, "开始录像")
        self.btn_intr = PushButton(FluentIcon.INFO, "读取内参")
        acts.addWidget(self.btn_snap)
        acts.addWidget(self.btn_rec)
        acts.addWidget(self.btn_intr)
        acts.addStretch(1)
        layout.addLayout(acts)

        self.status = BodyLabel("状态：空闲")
        layout.addWidget(self.status)

        self.btn_refresh.clicked.connect(self.refresh_devices)
        self.btn_start.clicked.connect(self.start_stream)
        self.btn_stop.clicked.connect(self.stop_stream)
        self.btn_snap.clicked.connect(self.take_snapshot)
        self.btn_rec.clicked.connect(self.toggle_record)
        self.btn_intr.clicked.connect(self.show_intrinsics)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update_preview)
        self._timer.start(66)  # ~15 FPS UI

    def set_candidates(self, devices: list) -> None:
        current = self.combo.currentText()
        self.combo.clear()
        for d in devices:
            label = f"{d['name']}  [{d['serial']}]"
            self.combo.addItem(label, userData=d["serial"])
        # restore
        for i in range(self.combo.count()):
            if self.combo.itemData(i) and current.endswith(f"[{self.combo.itemData(i)}]"):
                self.combo.setCurrentIndex(i)
                break

    def refresh_devices(self) -> None:
        data = self.service.scan()
        key = "wrist_candidates" if self.role == "wrist" else "scene_candidates"
        self.set_candidates(data.get(key, []))
        InfoBar.success("扫描完成", f"{self.role} 候选 {self.combo.count()} 台", parent=self.window(),
                        position=InfoBarPosition.TOP, duration=2000)

    def _selected_serial(self) -> Optional[str]:
        if self.combo.count() == 0:
            return None
        return self.combo.currentData()

    def start_stream(self) -> None:
        sn = self._selected_serial()
        if not sn:
            InfoBar.warning("未选择相机", "请先扫描并选择设备", parent=self.window(),
                            position=InfoBarPosition.TOP)
            return
        try:
            info = self.service.start_stream(sn, role=self.role)
            self.serial = sn
            mock = "MOCK" if info.get("is_mock") else "HW"
            self.status.setText(f"状态：出流中 ({mock}) SN={sn}")
            InfoBar.success("出流已启动", sn, parent=self.window(), position=InfoBarPosition.TOP, duration=2000)
        except Exception as exc:
            InfoBar.error("出流失败", str(exc), parent=self.window(), position=InfoBarPosition.TOP)

    def stop_stream(self) -> None:
        sn = self.serial or self._selected_serial()
        if not sn:
            return
        if self._recording:
            self.service.stop_recording(sn)
            self._recording = False
            self.btn_rec.setText("开始录像")
        self.service.stop_stream(sn)
        self.serial = None
        self.preview.setText("无画面")
        self.status.setText("状态：已停止")

    def update_preview(self) -> None:
        sn = self.serial
        if not sn:
            return
        st = self.service.streams.get(sn)
        if st is None:
            return
        pkt = st.get_latest()
        if pkt is None:
            return
        pix = numpy_bgr_to_qpixmap(pkt.color_bgr)
        self.preview.setPixmap(pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.status.setText(f"状态：出流中  frames={st.frame_count}")

    def take_snapshot(self) -> None:
        sn = self.serial
        if not sn:
            InfoBar.warning("未出流", "请先开始出流", parent=self.window(), position=InfoBarPosition.TOP)
            return
        try:
            paths = self.service.snapshot(sn)
            InfoBar.success("已拍照", paths.get("color", ""), parent=self.window(),
                            position=InfoBarPosition.TOP, duration=3000)
        except Exception as exc:
            InfoBar.error("拍照失败", str(exc), parent=self.window(), position=InfoBarPosition.TOP)

    def toggle_record(self) -> None:
        sn = self.serial
        if not sn:
            InfoBar.warning("未出流", "请先开始出流", parent=self.window(), position=InfoBarPosition.TOP)
            return
        try:
            if not self._recording:
                info = self.service.start_recording(sn)
                self._recording = True
                self.btn_rec.setText("停止录像")
                InfoBar.success("录像中", info.get("path", ""), parent=self.window(),
                                position=InfoBarPosition.TOP, duration=3000)
            else:
                info = self.service.stop_recording(sn)
                self._recording = False
                self.btn_rec.setText("开始录像")
                InfoBar.success("录像已保存", f"{info.get('path')} ({info.get('frames')} frames)",
                                parent=self.window(), position=InfoBarPosition.TOP, duration=4000)
        except Exception as exc:
            InfoBar.error("录像失败", str(exc), parent=self.window(), position=InfoBarPosition.TOP)

    def show_intrinsics(self) -> None:
        sn = self.serial or self._selected_serial()
        if not sn:
            return
        try:
            data = self.service.get_intrinsics(sn, from_live=True)
            import json
            text = json.dumps(data, indent=2, ensure_ascii=False)
            InfoBar.info("内参 (SDK)", text[:800], parent=self.window(),
                         position=InfoBarPosition.TOP, duration=8000)
            logger.info("intrinsics: %s", text)
        except Exception as exc:
            InfoBar.error("读取内参失败", str(exc), parent=self.window(), position=InfoBarPosition.TOP)


class ExtrinsicsPage(QWidget):
    """M2: user inputs 4x4 extrinsics."""

    def __init__(self, service: DetectService, parent=None):
        super().__init__(parent)
        self.service = service
        self.setObjectName("extrinsics")
        layout = QVBoxLayout(self)
        layout.addWidget(SubtitleLabel("外参配置（用户输入）"))
        layout.addWidget(BodyLabel("D405: T_flange_cam（眼在手）; D435: T_base_cam。矩阵为 4×4 行主序，单位米。"))

        form = QFormLayout()
        self.d405_serial = LineEdit()
        self.d435_serial = LineEdit()
        form.addRow("D405 序列号", self.d405_serial)
        form.addRow("D435 序列号", self.d435_serial)
        layout.addLayout(form)

        layout.addWidget(BodyLabel("D405 外参矩阵 T（每行 4 个数，空格/逗号分隔）"))
        self.d405_mat = TextEdit()
        self.d405_mat.setPlaceholderText("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1")
        layout.addWidget(self.d405_mat)

        layout.addWidget(BodyLabel("D435 外参矩阵 T"))
        self.d435_mat = TextEdit()
        self.d435_mat.setPlaceholderText("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1")
        layout.addWidget(self.d435_mat)

        row = QHBoxLayout()
        self.btn_load = PushButton(FluentIcon.FOLDER, "重新加载")
        self.btn_save = PrimaryPushButton(FluentIcon.SAVE, "保存外参")
        row.addWidget(self.btn_load)
        row.addWidget(self.btn_save)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addStretch(1)

        self.btn_load.clicked.connect(self.load)
        self.btn_save.clicked.connect(self.save)
        self.load()

    @staticmethod
    def _mat_to_text(mat) -> str:
        lines = []
        for row in mat:
            lines.append(" ".join(f"{float(x):.6g}" for x in row))
        return "\n".join(lines)

    @staticmethod
    def _text_to_mat(text: str):
        rows = []
        for line in text.strip().splitlines():
            line = line.replace(",", " ").strip()
            if not line:
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) != 4:
                raise ValueError(f"每行需要 4 个数，收到: {line}")
            rows.append(vals)
        if len(rows) != 4:
            raise ValueError("需要 4 行组成 4×4 矩阵")
        return rows

    def load(self) -> None:
        data = self.service.get_extrinsics()
        d405 = data.get("d405", {})
        d435 = data.get("d435", {})
        self.d405_serial.setText(d405.get("serial", ""))
        self.d435_serial.setText(d435.get("serial", ""))
        self.d405_mat.setPlainText(self._mat_to_text(d405.get("T", [])))
        self.d435_mat.setPlainText(self._mat_to_text(d435.get("T", [])))

    def save(self) -> None:
        try:
            payload = {
                "d405_serial": self.d405_serial.text().strip(),
                "d435_serial": self.d435_serial.text().strip(),
                "d405_T": self._text_to_mat(self.d405_mat.toPlainText()),
                "d435_T": self._text_to_mat(self.d435_mat.toPlainText()),
            }
            self.service.set_extrinsics(payload)
            InfoBar.success("已保存", "extrinsics.yaml 已更新", parent=self.window(),
                            position=InfoBarPosition.TOP, duration=2500)
        except Exception as exc:
            InfoBar.error("保存失败", str(exc), parent=self.window(), position=InfoBarPosition.TOP)


class CamerasPage(QWidget):
    def __init__(self, service: DetectService, parent=None):
        super().__init__(parent)
        self.setObjectName("cameras")
        self.service = service
        layout = QHBoxLayout(self)
        self.wrist = CameraLaneWidget("腕部相机待选区（D405）", "wrist", service)
        self.scene = CameraLaneWidget("场景相机待选区（D435）", "scene", service)
        layout.addWidget(self.wrist)
        layout.addWidget(self.scene)
        self.refresh_all()

    def refresh_all(self) -> None:
        data = self.service.scan()
        self.wrist.set_candidates(data.get("wrist_candidates", []))
        self.scene.set_candidates(data.get("scene_candidates", []))


class MainWindow(FluentWindow):
    def __init__(self, service: DetectService):
        super().__init__()
        self.service = service
        self.setWindowTitle("rs-detect  ·  D435/D405")
        self.resize(1280, 800)

        self.cameras_page = CamerasPage(service, self)
        self.extrinsics_page = ExtrinsicsPage(service, self)

        self.addSubInterface(self.cameras_page, FluentIcon.CAMERA, "相机出流")
        self.addSubInterface(self.extrinsics_page, FluentIcon.LIBRARY, "外参标定",
                             position=NavigationItemPosition.SCROLL)

    def closeEvent(self, event):
        self.service.stop_all()
        super().closeEvent(event)


def main() -> None:
    # HiDPI
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("rs-detect")
    service = get_service()
    win = MainWindow(service)
    win.show()
    code = app.exec_()
    service.stop_all()
    sys.exit(code)


if __name__ == "__main__":
    main()
