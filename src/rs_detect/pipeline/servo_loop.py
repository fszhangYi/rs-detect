"""Closed-loop fine localization: D405 measure → move to pre-grasp → re-measure."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass
class ServoStep:
    iter: int
    c_star: Optional[List[float]]
    p_pregrasp: Optional[List[float]]
    tcp: Optional[List[float]]
    error_xy_m: Optional[float]
    commanded_xyz: Optional[List[float]]
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iter": self.iter,
            "c_star": self.c_star,
            "p_pregrasp": self.p_pregrasp,
            "tcp": self.tcp,
            "error_xy_m": self.error_xy_m,
            "commanded_xyz": self.commanded_xyz,
            "message": self.message,
        }


@dataclass
class ServoLoopResult:
    ok: bool
    steps: List[ServoStep] = field(default_factory=list)
    final_c_star: Optional[List[float]] = None
    final_p_pregrasp: Optional[List[float]] = None
    pregrasp_z_lift_m: float = 0.05
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "message": self.message,
            "final_c_star": self.final_c_star,
            "final_p_pregrasp": self.final_p_pregrasp,
            "pregrasp_z_lift_m": self.pregrasp_z_lift_m,
            "steps": [s.to_dict() for s in self.steps],
        }


def _xy_err(a: Sequence[float], b: Sequence[float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def run_d405_servo_loop(
    *,
    detect_fine: Callable[[], Dict[str, Any]],
    move_xyz: Callable[[Sequence[float]], Dict[str, Any]],
    get_tcp_xyz: Callable[[], Optional[List[float]]],
    pregrasp_z_lift_m: float = 0.05,
    max_iters: int = 5,
    pos_tol_m: float = 0.005,
    settle_s: float = 0.35,
) -> ServoLoopResult:
    """D405 fine → p_pregrasp = c* + Z lift → move → re-detect until XY stable.

    Does **not** use D435 depth. Fine point comes from wrist camera only.
    """
    lift = float(pregrasp_z_lift_m)
    steps: List[ServoStep] = []
    last_c: Optional[List[float]] = None
    last_pre: Optional[List[float]] = None

    for i in range(max(1, int(max_iters))):
        fine = detect_fine()
        if not fine.get("ok") or not fine.get("c_star"):
            steps.append(
                ServoStep(
                    iter=i,
                    c_star=None,
                    p_pregrasp=None,
                    tcp=get_tcp_xyz(),
                    error_xy_m=None,
                    commanded_xyz=None,
                    message=f"fine failed: {fine.get('message')}",
                )
            )
            return ServoLoopResult(
                ok=False,
                steps=steps,
                final_c_star=last_c,
                final_p_pregrasp=last_pre,
                pregrasp_z_lift_m=lift,
                message=f"fine detect failed at iter {i}: {fine.get('message')}",
            )

        c_star = [float(x) for x in fine["c_star"][:3]]
        p_pre = [c_star[0], c_star[1], c_star[2] + lift]
        last_c, last_pre = c_star, p_pre

        try:
            move_xyz(p_pre)
        except Exception as exc:  # noqa: BLE001
            steps.append(
                ServoStep(
                    iter=i,
                    c_star=c_star,
                    p_pregrasp=p_pre,
                    tcp=get_tcp_xyz(),
                    error_xy_m=None,
                    commanded_xyz=p_pre,
                    message=f"move failed: {exc}",
                )
            )
            return ServoLoopResult(
                ok=False,
                steps=steps,
                final_c_star=c_star,
                final_p_pregrasp=p_pre,
                pregrasp_z_lift_m=lift,
                message=f"move to pregrasp failed: {exc}",
            )

        if settle_s > 0:
            time.sleep(float(settle_s))

        tcp = get_tcp_xyz()
        err = _xy_err(tcp, c_star) if tcp and len(tcp) >= 2 else None
        steps.append(
            ServoStep(
                iter=i,
                c_star=c_star,
                p_pregrasp=p_pre,
                tcp=tcp,
                error_xy_m=err,
                commanded_xyz=p_pre,
                message="ok" if (err is not None and err <= pos_tol_m) else "continue",
            )
        )
        if err is not None and err <= float(pos_tol_m):
            return ServoLoopResult(
                ok=True,
                steps=steps,
                final_c_star=c_star,
                final_p_pregrasp=p_pre,
                pregrasp_z_lift_m=lift,
                message=f"converged iter={i} err_xy={err:.4f} m",
            )

    err_last = steps[-1].error_xy_m if steps else None
    return ServoLoopResult(
        ok=False,
        steps=steps,
        final_c_star=last_c,
        final_p_pregrasp=last_pre,
        pregrasp_z_lift_m=lift,
        message=f"max iters reached; last err_xy={err_last}",
    )
