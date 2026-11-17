from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from rs_detect.pipeline.coarse_localize import CoarseResult
from rs_detect.pipeline.fine_localize import FineResult


@dataclass
class GateResult:
    gate_ok: bool
    gate_reason: str
    distance_m: Optional[float] = None
    normal_angle_deg: Optional[float] = None
    thresholds: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_ok": self.gate_ok,
            "gate_reason": self.gate_reason,
            "distance_m": self.distance_m,
            "normal_angle_deg": self.normal_angle_deg,
            "thresholds": self.thresholds,
        }


def gate_fine_vs_coarse(
    coarse: CoarseResult,
    fine: FineResult,
    *,
    max_distance_m: float = 0.08,
    max_normal_angle_deg: float = 25.0,
    require_coarse_detect: bool = True,
    require_fine_detect: bool = True,
) -> GateResult:
    """Plan stage E (lite): consistency gate — never blend D435/D405 depths."""
    thr = {
        "max_distance_m": float(max_distance_m),
        "max_normal_angle_deg": float(max_normal_angle_deg),
    }
    if require_coarse_detect and not coarse.ok:
        return GateResult(False, f"coarse not ok: {coarse.message}", thresholds=thr)
    if require_fine_detect and not fine.ok:
        return GateResult(False, f"fine not ok: {fine.message}", thresholds=thr)
    if coarse.p_coarse is None or fine.c_star is None:
        return GateResult(
            False,
            "missing p_coarse or c_star in base (need extrinsics / T_base_flange)",
            thresholds=thr,
        )

    d = float(np.linalg.norm(np.asarray(fine.c_star) - np.asarray(coarse.p_coarse)))
    ang = None
    if coarse.n_coarse is not None and fine.n_star is not None:
        n0 = np.asarray(coarse.n_coarse, dtype=np.float64)
        n1 = np.asarray(fine.n_star, dtype=np.float64)
        n0 = n0 / (np.linalg.norm(n0) + 1e-12)
        n1 = n1 / (np.linalg.norm(n1) + 1e-12)
        c = float(np.clip(np.dot(n0, n1), -1.0, 1.0))
        ang = float(np.degrees(np.arccos(c)))

    if d > max_distance_m:
        return GateResult(
            False,
            f"distance {d:.4f} m > {max_distance_m} m",
            distance_m=d,
            normal_angle_deg=ang,
            thresholds=thr,
        )
    if ang is not None and ang > max_normal_angle_deg:
        return GateResult(
            False,
            f"normal angle {ang:.1f}° > {max_normal_angle_deg}°",
            distance_m=d,
            normal_angle_deg=ang,
            thresholds=thr,
        )
    return GateResult(
        True,
        "ok",
        distance_m=d,
        normal_angle_deg=ang,
        thresholds=thr,
    )
