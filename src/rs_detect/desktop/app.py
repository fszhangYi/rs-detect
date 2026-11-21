from __future__ import annotations

import logging
import sys
from typing import Callable, List, Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
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

from rs_detect.paths import ensure_workspace_imports

ensure_workspace_imports()

from rs_detect.service import DetectService, get_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rs_detect.desktop")


def numpy_bgr_to_qpixmap(img: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class AspectRatioPreview(QLabel):
    """Letterboxed preview that preserves the source frame aspect ratio."""

    def __init__(self, parent=None, ratio: float = 4.0 / 3.0):
        super().__init__(parent)
        self._pix: Optional[QPixmap] = None
        self._ratio = ratio
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#121212;color:#888;border-radius:8px;")
        self.setText("无画面")

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return max(1, int(width / max(self._ratio, 1e-6)))

    def set_frame(self, pix: Optional[QPixmap], src_w: int = 0, src_h: int = 0) -> None:
        self._pix = pix
        if src_w > 0 and src_h > 0:
            self._ratio = float(src_w) / float(src_h)
        if pix is None:
            self.clear()
            self.setText("无画面")
            return
        self._relayout()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout()

    def _relayout(self) -> None:
        if self._pix is None or self._pix.isNull():
            return
        scaled = self._pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        super().setPixmap(scaled)


class MatrixEditor(QGroupBox):
    """4×4 numeric grid for extrinsics (avoids free-text format errors)."""

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        self._cells: List[List[QDoubleSpinBox]] = []
        for r in range(4):
            row_widgets = []
            for c in range(4):
                spin = QDoubleSpinBox()
                spin.setDecimals(6)
                spin.setRange(-1e6, 1e6)
                spin.setSingleStep(0.001)
                spin.setMaximumWidth(110)
                # identity default
                spin.setValue(1.0 if r == c else 0.0)
                grid.addWidget(spin, r, c)
                row_widgets.append(spin)
            self._cells.append(row_widgets)

    def get_matrix(self) -> List[List[float]]:
        return [[float(self._cells[r][c].value()) for c in range(4)] for r in range(4)]

    def set_matrix(self, mat: List[List[float]]) -> None:
        if not mat or len(mat) != 4:
            mat = [[1 if i == j else 0 for j in range(4)] for i in range(4)]
        for r in range(4):
            row = mat[r] if r < len(mat) else [0, 0, 0, 0]
            for c in range(4):
                val = float(row[c]) if c < len(row) else (1.0 if r == c else 0.0)
                self._cells[r][c].setValue(val)


class CameraLaneWidget(QWidget):
    """One lane: wrist (D405) or scene (D435)."""

    selectionChanged = pyqtSignal(str, str)  # role, serial

    def __init__(self, title: str, role: str, service: DetectService, parent=None):
        super().__init__(parent)
        self.role = role  # wrist | scene
        self.service = service
        self.serial: Optional[str] = None
        self._recording = False
        self._suppress_bind = False

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

        self.preview = AspectRatioPreview(self, ratio=4.0 / 3.0)
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
        self.combo.currentIndexChanged.connect(self._on_combo_changed)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update_preview)
        self._timer.start(66)

    def set_candidates(self, devices: list, preferred: Optional[str] = None) -> None:
        self._suppress_bind = True
        current = preferred or self.combo.currentData() or self.serial
        self.combo.clear()
        for d in devices:
            label = f"{d['name']}  [{d['serial']}]"
            self.combo.addItem(label, userData=d["serial"])
        idx = 0
        if current:
            for i in range(self.combo.count()):
                if self.combo.itemData(i) == current:
                    idx = i
                    break
        if self.combo.count():
            self.combo.setCurrentIndex(idx)
        self._suppress_bind = False
        self._bind_current(persist=True)

    def refresh_devices(self) -> None:
        data = self.service.scan()
        key = "wrist_candidates" if self.role == "wrist" else "scene_candidates"
        pref = data.get("selection", {}).get(self.role) or None
        self.set_candidates(data.get(key, []), preferred=pref)
        InfoBar.success(
            "扫描完成",
            f"{self.role} 候选 {self.combo.count()} 台",
            parent=self.window(),
            position=InfoBarPosition.TOP,
            duration=2000,
        )

    def _selected_serial(self) -> Optional[str]:
        if self.combo.count() == 0:
            return None
        return self.combo.currentData()

    def _on_combo_changed(self, _index: int = 0) -> None:
        if self._suppress_bind:
            return
        self._bind_current(persist=True)

    def _bind_current(self, persist: bool = True) -> None:
        sn = self._selected_serial()
        if not sn:
            return
        self.service.bind_selection(self.role, sn, persist=persist)
        self.selectionChanged.emit(self.role, sn)

    def start_stream(self) -> None:
        sn = self._selected_serial()
        if not sn:
            InfoBar.warning(
                "未选择相机",
                "请先扫描并选择设备",
                parent=self.window(),
                position=InfoBarPosition.TOP,
            )
            return
        try:
            info = self.service.start_stream(sn, role=self.role)
            self.serial = sn
            self.selectionChanged.emit(self.role, sn)
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
        self.preview.set_frame(None)
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
        h, w = pkt.color_bgr.shape[:2]
        pix = numpy_bgr_to_qpixmap(pkt.color_bgr)
        self.preview.set_frame(pix, src_w=w, src_h=h)
        self.status.setText(f"状态：出流中  frames={st.frame_count}  {w}×{h}")

    def take_snapshot(self) -> None:
        sn = self.serial
        if not sn:
            InfoBar.warning("未出流", "请先开始出流", parent=self.window(), position=InfoBarPosition.TOP)
            return
        try:
            paths = self.service.snapshot(sn)
            InfoBar.success(
                "已拍照",
                paths.get("color", ""),
                parent=self.window(),
                position=InfoBarPosition.TOP,
                duration=3000,
            )
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
                InfoBar.success(
                    "录像中",
                    info.get("path", ""),
                    parent=self.window(),
                    position=InfoBarPosition.TOP,
                    duration=3000,
                )
            else:
                info = self.service.stop_recording(sn)
                self._recording = False
                self.btn_rec.setText("开始录像")
                InfoBar.success(
                    "录像已保存",
                    f"{info.get('path')} ({info.get('frames')} frames)",
                    parent=self.window(),
                    position=InfoBarPosition.TOP,
                    duration=4000,
                )
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
            InfoBar.info("内参 (SDK)", text[:800], parent=self.window(), position=InfoBarPosition.TOP, duration=8000)
            logger.info("intrinsics: %s", text)
        except Exception as exc:
            InfoBar.error("读取内参失败", str(exc), parent=self.window(), position=InfoBarPosition.TOP)


class ExtrinsicsPage(QWidget):
    """M1: serials locked from M0; 4×4 grid editors for T."""

    def __init__(self, service: DetectService, parent=None):
        super().__init__(parent)
        self.service = service
        self.setObjectName("extrinsics")
        layout = QVBoxLayout(self)
        layout.addWidget(SubtitleLabel("外参配置（M1）"))
        layout.addWidget(
            BodyLabel("序列号由 M0 选定相机自动联动（只读）。矩阵用 4×4 数值格填写，单位米，行主序。")
        )

        form = QFormLayout()
        self.d405_serial = LineEdit()
        self.d435_serial = LineEdit()
        self.d405_serial.setReadOnly(True)
        self.d435_serial.setReadOnly(True)
        self.d405_serial.setPlaceholderText("请在「相机出流」中选择腕部 D405")
        self.d435_serial.setPlaceholderText("请在「相机出流」中选择场景 D435")
        form.addRow("D405 序列号（腕部）", self.d405_serial)
        form.addRow("D435 序列号（场景）", self.d435_serial)
        layout.addLayout(form)

        mats = QHBoxLayout()
        self.d405_mat = MatrixEditor("D405  T_flange_cam（眼在手）")
        self.d435_mat = MatrixEditor("D435  T_base_cam")
        mats.addWidget(self.d405_mat)
        mats.addWidget(self.d435_mat)
        layout.addLayout(mats)

        layout.addWidget(BodyLabel("备注"))
        self.notes = TextEdit()
        self.notes.setMaximumHeight(72)
        layout.addWidget(self.notes)

        row = QHBoxLayout()
        self.btn_load = PushButton(FluentIcon.FOLDER, "重新加载")
        self.btn_identity = PushButton(FluentIcon.CANCEL, "复位为单位阵")
        self.btn_save = PrimaryPushButton(FluentIcon.SAVE, "保存外参")
        row.addWidget(self.btn_load)
        row.addWidget(self.btn_identity)
        row.addWidget(self.btn_save)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addStretch(1)

        self.btn_load.clicked.connect(self.load)
        self.btn_identity.clicked.connect(self.reset_identity)
        self.btn_save.clicked.connect(self.save)
        self.load()

        self._sync_timer = QTimer(self)
        self._sync_timer.timeout.connect(self.sync_serials_from_selection)
        self._sync_timer.start(500)

    def sync_serials_from_selection(self) -> None:
        sel = self.service.get_selection()
        self.d405_serial.setText(sel.get("d405_serial") or "")
        self.d435_serial.setText(sel.get("d435_serial") or "")

    def on_selection_changed(self, role: str, serial: str) -> None:
        if role == "wrist":
            self.d405_serial.setText(serial)
        elif role == "scene":
            self.d435_serial.setText(serial)

    def load(self) -> None:
        data = self.service.get_extrinsics()
        self.sync_serials_from_selection()
        # if file has serials and selection empty, show file values once
        if not self.d405_serial.text():
            self.d405_serial.setText((data.get("d405") or {}).get("serial", ""))
        if not self.d435_serial.text():
            self.d435_serial.setText((data.get("d435") or {}).get("serial", ""))
        self.d405_mat.set_matrix((data.get("d405") or {}).get("T") or [])
        self.d435_mat.set_matrix((data.get("d435") or {}).get("T") or [])
        self.notes.setPlainText(data.get("notes") or "")

    def reset_identity(self) -> None:
        ident = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        self.d405_mat.set_matrix(ident)
        self.d435_mat.set_matrix(ident)

    def save(self) -> None:
        try:
            self.sync_serials_from_selection()
            payload = {
                "d405_serial": self.d405_serial.text().strip(),
                "d435_serial": self.d435_serial.text().strip(),
                "d405_T": self.d405_mat.get_matrix(),
                "d435_T": self.d435_mat.get_matrix(),
                "notes": self.notes.toPlainText().strip(),
            }
            if not payload["d405_serial"] or not payload["d435_serial"]:
                raise ValueError("请先在 M0 相机出流页选定腕部/场景相机，序列号会自动填入")
            self.service.set_extrinsics(payload)
            InfoBar.success(
                "已保存",
                "extrinsics.yaml 已更新",
                parent=self.window(),
                position=InfoBarPosition.TOP,
                duration=2500,
            )
        except Exception as exc:
            InfoBar.error("保存失败", str(exc), parent=self.window(), position=InfoBarPosition.TOP)


class DetectPage(QWidget):
    """M2 plan flow: D435 coarse → D405 fine → gate (no depth blend)."""

    def __init__(self, service: DetectService, parent=None):
        super().__init__(parent)
        self.service = service
        self.setObjectName("detect")
        self._show_role = "scene"

        layout = QVBoxLayout(self)
        layout.addWidget(SubtitleLabel("检测通路（M2）"))
        layout.addWidget(
            BodyLabel(
                "按方案：D435 粗定位给出 p_coarse（导航/门控）→ D405 精定位给出吸点 → 一致性门控。"
                "禁止两相机深度平均。请先在 M0 同时出流腕部+场景。"
            )
        )

        row = QHBoxLayout()
        self.method_combo = ComboBox(self)
        for label, data in [
            ("auto", "auto"),
            ("yolo", "yolo"),
            ("hsv", "hsv"),
            ("mock", "mock"),
        ]:
            self.method_combo.addItem(label, userData=data)
        self.btn_pipeline = PrimaryPushButton(FluentIcon.PLAY, "运行 粗→精→门控")
        self.btn_coarse = PushButton(FluentIcon.SEARCH, "仅 D435 粗定位")
        self.btn_fine = PushButton(FluentIcon.CAMERA, "仅 D405 精定位")
        self.viz_combo = ComboBox(self)
        self.viz_combo.addItem("预览 D435 粗", userData="scene")
        self.viz_combo.addItem("预览 D405 精", userData="wrist")
        row.addWidget(BodyLabel("方法"))
        row.addWidget(self.method_combo)
        row.addWidget(self.btn_pipeline)
        row.addWidget(self.btn_coarse)
        row.addWidget(self.btn_fine)
        row.addWidget(self.viz_combo)
        row.addStretch(1)
        layout.addLayout(row)

        self.preview = AspectRatioPreview(self, ratio=4.0 / 3.0)
        layout.addWidget(self.preview, 1)
        self.result = TextEdit()
        self.result.setReadOnly(True)
        self.result.setMaximumHeight(200)
        layout.addWidget(self.result)

        self.btn_pipeline.clicked.connect(self.run_pipeline)
        self.btn_coarse.clicked.connect(self.run_coarse_only)
        self.btn_fine.clicked.connect(self.run_fine_only)
        self.viz_combo.currentIndexChanged.connect(self._refresh_viz)

    def _method(self) -> str:
        return self.method_combo.currentData() or "auto"

    def _show_payload(self, payload: dict, prefer_role: str = "scene") -> None:
        import json

        self.result.setPlainText(json.dumps(payload, indent=2, ensure_ascii=False))
        self._show_role = prefer_role
        # sync combo without re-entrancy issues
        idx = 0 if prefer_role == "scene" else 1
        if self.viz_combo.currentIndex() != idx:
            self.viz_combo.setCurrentIndex(idx)
        else:
            self._refresh_viz()

    def _refresh_viz(self) -> None:
        role = self.viz_combo.currentData() or self._show_role
        sel = self.service.get_selection()
        serial = sel.get("scene") if role == "scene" else sel.get("wrist")
        if not serial:
            return
        viz = self.service.get_detect_viz_jpeg(serial)
        if not viz:
            return
        arr = np.frombuffer(viz, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
            self.preview.set_frame(numpy_bgr_to_qpixmap(img), src_w=w, src_h=h)

    def run_pipeline(self) -> None:
        try:
            payload = self.service.run_m2_pipeline(method=self._method())
            self._show_payload(payload, prefer_role="scene")
            gate = payload.get("gate") or {}
            if payload.get("ok"):
                InfoBar.success(
                    "粗→精完成",
                    f"gate_ok dist={gate.get('distance_m')}",
                    parent=self.window(),
                    position=InfoBarPosition.TOP,
                    duration=3500,
                )
            else:
                InfoBar.warning(
                    "管线未通过",
                    payload.get("message") or gate.get("gate_reason") or "",
                    parent=self.window(),
                    position=InfoBarPosition.TOP,
                    duration=5000,
                )
        except Exception as exc:
            InfoBar.error("管线失败", str(exc), parent=self.window(), position=InfoBarPosition.TOP)

    def run_coarse_only(self) -> None:
        try:
            payload = self.service.run_coarse(method=self._method())
            self._show_payload(payload, prefer_role="scene")
            if payload.get("ok"):
                InfoBar.success(
                    "粗定位 OK",
                    f"p_coarse={payload.get('p_coarse')}",
                    parent=self.window(),
                    position=InfoBarPosition.TOP,
                    duration=3500,
                )
            else:
                InfoBar.warning(
                    "粗定位失败",
                    payload.get("message", ""),
                    parent=self.window(),
                    position=InfoBarPosition.TOP,
                    duration=4000,
                )
        except Exception as exc:
            InfoBar.error("粗定位失败", str(exc), parent=self.window(), position=InfoBarPosition.TOP)

    def run_fine_only(self) -> None:
        try:
            payload = self.service.run_fine(method=self._method())
            self._show_payload(payload, prefer_role="wrist")
            if payload.get("ok"):
                InfoBar.success(
                    "精定位 OK",
                    f"c_star={payload.get('c_star')}",
                    parent=self.window(),
                    position=InfoBarPosition.TOP,
                    duration=3500,
                )
            else:
                InfoBar.warning(
                    "精定位失败",
                    payload.get("message", ""),
                    parent=self.window(),
                    position=InfoBarPosition.TOP,
                    duration=4000,
                )
        except Exception as exc:
            InfoBar.error("精定位失败", str(exc), parent=self.window(), position=InfoBarPosition.TOP)


class CamerasPage(QWidget):
    selectionChanged = pyqtSignal(str, str)

    def __init__(self, service: DetectService, parent=None):
        super().__init__(parent)
        self.setObjectName("cameras")
        self.service = service
        layout = QHBoxLayout(self)
        self.wrist = CameraLaneWidget("腕部相机待选区（D405）", "wrist", service)
        self.scene = CameraLaneWidget("场景相机待选区（D435）", "scene", service)
        layout.addWidget(self.wrist)
        layout.addWidget(self.scene)
        self.wrist.selectionChanged.connect(self.selectionChanged.emit)
        self.scene.selectionChanged.connect(self.selectionChanged.emit)
        self.refresh_all()

    def refresh_all(self) -> None:
        data = self.service.scan()
        sel = data.get("selection", {})
        self.wrist.set_candidates(data.get("wrist_candidates", []), preferred=sel.get("wrist") or None)
        self.scene.set_candidates(data.get("scene_candidates", []), preferred=sel.get("scene") or None)


class MainWindow(FluentWindow):
    def __init__(self, service: DetectService):
        super().__init__()
        self.service = service
        self.setWindowTitle("rs-detect  ·  D435/D405")
        self.resize(1280, 800)

        self.cameras_page = CamerasPage(service, self)
        self.extrinsics_page = ExtrinsicsPage(service, self)
        self.detect_page = DetectPage(service, self)
        self.cameras_page.selectionChanged.connect(self.extrinsics_page.on_selection_changed)

        self.addSubInterface(self.cameras_page, FluentIcon.CAMERA, "相机出流 (M0)")
        self.addSubInterface(
            self.extrinsics_page,
            FluentIcon.LIBRARY,
            "外参标定 (M1)",
            position=NavigationItemPosition.SCROLL,
        )
        self.addSubInterface(
            self.detect_page,
            FluentIcon.SEARCH,
            "检测通路 (M2)",
            position=NavigationItemPosition.SCROLL,
        )

    def closeEvent(self, event):
        self.service.stop_all()
        super().closeEvent(event)


def main() -> None:
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
