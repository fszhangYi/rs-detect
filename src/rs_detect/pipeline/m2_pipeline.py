from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rs_detect.camera.models import FramePacket
from rs_detect.pipeline.coarse_localize import CoarseResult, coarse_localize
from rs_detect.pipeline.fine_localize import FineResult, fine_localize_single
from rs_detect.pipeline.gate import GateResult, gate_fine_vs_coarse


@dataclass
class M2PipelineResult:
    """Plan flow A → (B single-frame) → E-lite: D435 coarse then D405 fine + gate."""

    ok: bool
    coarse: CoarseResult
    fine: Optional[FineResult]
    gate: Optional[GateResult]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        fine_d = self.fine.to_dict() if self.fine else None
        gate_d = self.gate.to_dict() if self.gate else None
        # flatten plan fields for FineResult consumers
        if fine_d is not None and gate_d is not None:
            fine_d["gate_ok"] = gate_d["gate_ok"]
            fine_d["gate_reason"] = gate_d["gate_reason"]
        return {
            "ok": self.ok,
            "message": self.message,
            "flow": ["d435_coarse", "d405_fine_single", "gate"],
            "coarse": self.coarse.to_dict(),
            "fine": fine_d,
            "gate": gate_d,
            # convenience aliases matching plan names
            "p_coarse": self.coarse.p_coarse,
            "n_coarse": self.coarse.n_coarse,
            "c_star": fine_d.get("c_star") if fine_d else None,
            "n_star": fine_d.get("n_star") if fine_d else None,
            "gate_ok": gate_d.get("gate_ok") if gate_d else False,
            "gate_reason": gate_d.get("gate_reason") if gate_d else "not run",
        }


def run_m2_coarse_then_fine(
    scene_packet: FramePacket,
    wrist_packet: FramePacket,
    *,
    T_base_d435: List[List[float]],
    T_flange_d405: List[List[float]],
    T_base_flange: Optional[List[List[float]]] = None,
    method: str = "auto",
    weights: Optional[Path] = None,
    conf: float = 0.25,
    class_filter: Optional[List[str]] = None,
    scene_is_mock: bool = False,
    wrist_is_mock: bool = False,
    depth_scale: float = 0.001,
    ransac_thresh: float = 0.008,
    max_points: int = 20000,
    work_distance_m: float = 0.25,
    max_gate_distance_m: float = 0.08,
    max_gate_normal_angle_deg: float = 25.0,
    skip_fine_if_coarse_fails: bool = True,
    make_viz: bool = True,
) -> M2PipelineResult:
    """Canonical M2 entry: D435 coarse reference first, then D405 fine, then gate.

    Does **not** average D435/D405 depths. Fine suction comes from D405 only.
    """
    coarse = coarse_localize(
        scene_packet,
        T_base_d435,
        method=method,
        weights=weights,
        conf=conf,
        class_filter=class_filter,
        is_mock=scene_is_mock,
        depth_scale=depth_scale,
        ransac_thresh=ransac_thresh,
        max_points=max_points,
        work_distance_m=work_distance_m,
        make_viz=make_viz,
    )
    if not coarse.ok and skip_fine_if_coarse_fails:
        return M2PipelineResult(
            ok=False,
            coarse=coarse,
            fine=None,
            gate=GateResult(False, f"skipped fine: {coarse.message}"),
            message="coarse failed — abort before wrist fine (plan stage A)",
        )

    fine = fine_localize_single(
        wrist_packet,
        T_flange_d405=T_flange_d405,
        T_base_flange=T_base_flange,
        method=method,
        weights=weights,
        conf=conf,
        class_filter=class_filter,
        is_mock=wrist_is_mock,
        depth_scale=depth_scale,
        ransac_thresh=ransac_thresh,
        max_points=max_points,
        make_viz=make_viz,
    )
    gate = gate_fine_vs_coarse(
        coarse,
        fine,
        max_distance_m=max_gate_distance_m,
        max_normal_angle_deg=max_gate_normal_angle_deg,
    )
    ok = bool(coarse.ok and fine.ok and gate.gate_ok)
    msg = "ok" if ok else f"pipeline incomplete: coarse={coarse.ok} fine={fine.ok} gate={gate.gate_ok} ({gate.gate_reason})"
    return M2PipelineResult(ok=ok, coarse=coarse, fine=fine, gate=gate, message=msg)
