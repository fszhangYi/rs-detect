from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    method: str
    boxes: List[List[float]] = field(default_factory=list)  # xyxy
    scores: List[float] = field(default_factory=list)
    class_ids: List[int] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    mask: Optional[np.ndarray] = None  # HxW uint8 0/255
    # Per-instance segmentation polygons in image pixels: [[x,y], ...]
    polygons: List[List[List[float]]] = field(default_factory=list)
    message: str = ""

    def to_dict(self, include_mask: bool = False, include_polygons: bool = False) -> Dict[str, Any]:
        d = {
            "method": self.method,
            "boxes": self.boxes,
            "scores": self.scores,
            "class_ids": self.class_ids,
            "class_names": self.class_names,
            "message": self.message,
            "has_mask": self.mask is not None and int(np.count_nonzero(self.mask)) > 0,
            "mask_pixels": int(np.count_nonzero(self.mask)) if self.mask is not None else 0,
            "polygon_counts": [len(p) for p in (self.polygons or [])],
            "has_polygons": bool(self.polygons),
        }
        if include_polygons:
            d["polygons"] = self.polygons
        if include_mask and self.mask is not None:
            d["mask_shape"] = list(self.mask.shape)
        return d


def _box_to_mask(h: int, w: int, box: List[float], pad: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1 : y2 + 1, x1 : x2 + 1] = 255
    return mask


def detect_hsv_cardboard(
    color_bgr: np.ndarray,
    hsv_low: Tuple[int, int, int] = (10, 30, 80),
    hsv_high: Tuple[int, int, int] = (35, 180, 255),
    min_area_ratio: float = 0.01,
) -> DetectionResult:
    """Fallback detector for brownish cardboard under 'no similar-color distractors'."""
    h, w = color_bgr.shape[:2]
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_high))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_ratio * h * w
    best = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        if area > best_area:
            best_area = area
            best = cnt
    if best is None:
        return DetectionResult(method="hsv", message="no contour found")
    x, y, bw, bh = cv2.boundingRect(best)
    box = [float(x), float(y), float(x + bw), float(y + bh)]
    out_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(out_mask, [best], -1, 255, thickness=-1)
    return DetectionResult(
        method="hsv",
        boxes=[box],
        scores=[float(min(1.0, best_area / (h * w)))],
        class_ids=[0],
        class_names=["cardboard"],
        mask=out_mask,
        message=f"hsv area={int(best_area)}",
    )


def detect_mock_roi(color_bgr: np.ndarray, margin: float = 0.2) -> DetectionResult:
    """Center ROI for mock cameras / offline smoke tests."""
    h, w = color_bgr.shape[:2]
    x1, y1 = int(w * margin), int(h * margin)
    x2, y2 = int(w * (1 - margin)), int(h * (1 - margin))
    box = [float(x1), float(y1), float(x2), float(y2)]
    mask = _box_to_mask(h, w, box, pad=0)
    # paint a "cardboard" depth-friendly region already exists in mock depth
    return DetectionResult(
        method="mock",
        boxes=[box],
        scores=[1.0],
        class_ids=[0],
        class_names=["cardboard"],
        mask=mask,
        message="mock center ROI",
    )


_YOLO_MODEL = None
_YOLO_PATH = None


def _load_yolo(weights: Path):
    global _YOLO_MODEL, _YOLO_PATH
    if _YOLO_MODEL is not None and _YOLO_PATH == str(weights):
        return _YOLO_MODEL
    from ultralytics import YOLO

    _YOLO_MODEL = YOLO(str(weights))
    _YOLO_PATH = str(weights)
    logger.info("Loaded YOLO weights: %s", weights)
    return _YOLO_MODEL


def detect_yolo(
    color_bgr: np.ndarray,
    weights: Path,
    conf: float = 0.25,
    imgsz: int = 640,
    class_filter: Optional[List[str]] = None,
    device: Optional[str] = None,
    keep_top_k: int = 1,
) -> DetectionResult:
    """Run Ultralytics YOLO-seg. ROI mask uses top-k by score (default 1).

    Boxes in the result are the kept instances (for viz). Low-conf mock runs can
    otherwise OR hundreds of masks into a useless full-frame ROI.
    """
    model = _load_yolo(Path(weights))
    # ultralytics expects RGB
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    predict_kw = {"source": rgb, "conf": conf, "imgsz": imgsz, "verbose": False}
    if device is not None and str(device).strip() != "":
        predict_kw["device"] = device
    results = model.predict(**predict_kw)
    if not results:
        return DetectionResult(method="yolo", message="empty yolo result")
    r0 = results[0]
    names = r0.names or {}
    h, w = color_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if r0.boxes is None or len(r0.boxes) == 0:
        return DetectionResult(method="yolo", message="no detections", mask=mask)

    xyxy = r0.boxes.xyxy.cpu().numpy()
    confs = r0.boxes.conf.cpu().numpy()
    cls = r0.boxes.cls.cpu().numpy().astype(int)

    seg_masks = None
    seg_xy: Optional[List[Any]] = None
    if getattr(r0, "masks", None) is not None and r0.masks is not None:
        try:
            seg_masks = r0.masks.data.cpu().numpy()
        except Exception:
            seg_masks = None
        try:
            # Ultralytics polygon vertices already in original image coords.
            seg_xy = list(r0.masks.xy) if getattr(r0.masks, "xy", None) is not None else None
        except Exception:
            seg_xy = None

    filter_l = [c.lower() for c in class_filter] if class_filter else None
    hits: List[Tuple[float, int]] = []  # (score, raw_index)
    for i in range(len(xyxy)):
        cname = str(names.get(int(cls[i]), str(int(cls[i]))))
        if filter_l is not None and cname.lower() not in filter_l:
            continue
        hits.append((float(confs[i]), i))

    if not hits:
        return DetectionResult(method="yolo", message="filtered to empty", mask=mask)

    hits.sort(key=lambda t: t[0], reverse=True)
    k = max(1, int(keep_top_k)) if keep_top_k and keep_top_k > 0 else len(hits)
    kept = hits[:k]

    boxes, scores, cids, cnames = [], [], [], []
    polygons: List[List[List[float]]] = []
    for score, i in kept:
        cname = str(names.get(int(cls[i]), str(int(cls[i]))))
        box = [float(v) for v in xyxy[i].tolist()]
        boxes.append(box)
        scores.append(float(score))
        cids.append(int(cls[i]))
        cnames.append(cname)
        poly: List[List[float]] = []
        if seg_xy is not None and i < len(seg_xy) and seg_xy[i] is not None:
            pts = np.asarray(seg_xy[i], dtype=np.float32)
            if pts.ndim == 2 and pts.shape[0] >= 3:
                poly = [[float(x), float(y)] for x, y in pts.tolist()]
        polygons.append(poly)
        if seg_masks is not None and i < len(seg_masks):
            m = seg_masks[i]
            if m.shape[:2] != (h, w):
                m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                m = m.astype(np.float32)
            mask[m > 0.5] = 255
        elif poly:
            cnt = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [cnt], 255)
        else:
            mask = cv2.bitwise_or(mask, _box_to_mask(h, w, box))

    n_poly = sum(1 for p in polygons if len(p) >= 3)
    return DetectionResult(
        method="yolo",
        boxes=boxes,
        scores=scores,
        class_ids=cids,
        class_names=cnames,
        mask=mask,
        polygons=polygons,
        message=f"yolo hits={len(boxes)} polys={n_poly} best={scores[0]:.3f}",
    )


# Ultralytics still applies NMS; near-zero conf keeps almost every box for mock viz.
_MOCK_YOLO_CONF = 1e-3


def detect_cardboard(
    color_bgr: np.ndarray,
    method: str = "auto",
    weights: Optional[Path] = None,
    conf: float = 0.25,
    class_filter: Optional[List[str]] = None,
    is_mock: bool = False,
    device: Optional[str] = None,
    imgsz: int = 640,
) -> DetectionResult:
    """Unified detector entry.

    method:
      - auto: try YOLO if weights exist; else HSV (mock: YOLO with near-zero conf)
      - yolo / hsv / mock

    Mock cameras still run YOLO when method=auto/yolo. Conf is floored so the
    cardboard instance below the configured threshold can still surface (mock
    frame best cardboard ≈0.21 < default 0.25). class_filter stays applied
    (cardboard). ROI mask = top-1 by score. Falls back to center ROI only if
    weights are missing, YOLO errors, or returns empty.
    """
    method = (method or "auto").lower()
    if method == "auto":
        if weights and Path(weights).exists():
            method = "yolo"
        else:
            method = "mock" if is_mock else "hsv"

    if method == "mock":
        return detect_mock_roi(color_bgr)
    if method == "hsv":
        return detect_hsv_cardboard(color_bgr)
    if method == "yolo":
        yolo_conf = min(float(conf), _MOCK_YOLO_CONF) if is_mock else float(conf)
        if not weights or not Path(weights).exists():
            logger.warning("YOLO weights missing (%s), fallback %s", weights, "mock" if is_mock else "hsv")
            if is_mock:
                det = detect_mock_roi(color_bgr)
                det.message = f"yolo weights missing → mock: {det.message}"
                return det
            det = detect_hsv_cardboard(color_bgr)
            det.message = f"yolo weights missing → hsv: {det.message}"
            return det
        try:
            det = detect_yolo(
                color_bgr,
                Path(weights),
                conf=yolo_conf,
                imgsz=imgsz,
                class_filter=class_filter,
                device=device,
                keep_top_k=1,
            )
            if is_mock and not det.boxes:
                # Still nothing at near-zero conf — keep smoke tests usable.
                fallback = detect_mock_roi(color_bgr)
                fallback.message = f"yolo empty @conf={yolo_conf} → mock: {det.message}"
                return fallback
            if is_mock and det.boxes:
                det.message = f"{det.message} (mock conf={yolo_conf})"
            return det
        except Exception as exc:
            logger.exception("YOLO failed: %s", exc)
            if is_mock:
                det = detect_mock_roi(color_bgr)
                det.message = f"yolo error → mock: {exc}"
                return det
            det = detect_hsv_cardboard(color_bgr)
            det.message = f"yolo error → hsv: {exc}"
            return det
    raise ValueError(f"unknown detect method: {method}")
