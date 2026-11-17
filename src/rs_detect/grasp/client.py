"""Thin HTTP client for grasp-service (S0 / S1 / S2 / Final + helpers)."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def format_http_detail(detail: Any) -> str:
    """Normalize FastAPI ``detail`` (str | list[validation] | dict) to a single string."""
    if detail is None:
        return ""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        parts: List[str] = []
        for item in detail:
            if isinstance(item, dict):
                loc = item.get("loc")
                loc_s = ".".join(str(x) for x in loc) if isinstance(loc, (list, tuple)) else ""
                msg = item.get("msg") or item.get("message") or str(item)
                parts.append(f"{loc_s}: {msg}" if loc_s else str(msg))
            else:
                parts.append(str(item))
        return "; ".join(parts) if parts else str(detail)
    if isinstance(detail, dict):
        if "detail" in detail:
            return format_http_detail(detail["detail"])
        return json.dumps(detail, ensure_ascii=False)
    return str(detail)


class GraspServiceError(RuntimeError):
    def __init__(self, status_code: int, detail: Union[str, Any]):
        self.status_code = int(status_code)
        self.detail = format_http_detail(detail) if not isinstance(detail, str) else detail
        self.raw_detail = detail
        super().__init__(f"HTTP {status_code}: {self.detail}")


class GraspServiceClient:
    def __init__(self, base_url: str = "http://127.0.0.1:6010", timeout_s: float = 60.0):
        self.base_url = str(base_url or "").rstrip("/")
        self.timeout_s = float(timeout_s)

    def set_base_url(self, base_url: str) -> None:
        self.base_url = str(base_url or "").rstrip("/")

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"

    def request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = self._url(path)
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            parsed: Any
            try:
                parsed = json.loads(raw)
            except Exception:  # noqa: BLE001
                parsed = raw
            if isinstance(parsed, dict) and "detail" in parsed:
                msg = parsed["detail"]
            else:
                msg = parsed
            raise GraspServiceError(int(exc.code), msg) from exc
        except urllib.error.URLError as exc:
            raise GraspServiceError(0, f"connect failed: {exc.reason}") from exc

        if not raw:
            return {"ok": True}
        try:
            detail = json.loads(raw)
        except Exception:  # noqa: BLE001
            return {"ok": True, "raw": raw}
        if not isinstance(detail, dict):
            return {"ok": True, "raw": detail}
        return detail

    def health(self) -> Dict[str, Any]:
        return self.request("GET", "/health")

    def site_config(self) -> Dict[str, Any]:
        """GET /v1/site/config — field schema + current effective site summary."""
        return self.request("GET", "/v1/site/config")

    def site_configure(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST /v1/site/configure — hot-update + persist under server_grasp/runtime/."""
        return self.request("POST", "/v1/site/configure", payload)

    def hole_detect(self, frame: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
        """HoleDetectRequest: FrameUpload + sam_text (material mask) — NOT hole_sam_text."""
        body = {**frame, **{k: v for k, v in extra.items() if v is not None}}
        body.pop("hole_sam_text", None)  # ignored by server; avoid confusion
        return self.request("POST", "/v1/hole/detect", body)

    def materials_detect(self, frame: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
        body = {**frame, **{k: v for k, v in extra.items() if v is not None}}
        return self.request("POST", "/v1/materials/detect", body)

    def s0_plan(self, frame: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
        body = {**frame, **{k: v for k, v in extra.items() if v is not None}}
        return self.request("POST", "/v1/s0/plan", body)

    def s1_latch(self, session_id: str, frame: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
        body = {"session_id": session_id, **frame, **{k: v for k, v in extra.items() if v is not None}}
        return self.request("POST", "/v1/s1/latch", body)

    def s1_step(self, session_id: str, frame: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
        body = {"session_id": session_id, **frame, **{k: v for k, v in extra.items() if v is not None}}
        return self.request("POST", "/v1/s1/step", body)

    def s2_latch(self, session_id: str, frame: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
        body = {"session_id": session_id, **frame, **{k: v for k, v in extra.items() if v is not None}}
        return self.request("POST", "/v1/s2/latch", body)

    def s2_step(self, session_id: str, frame: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
        body = {"session_id": session_id, **frame, **{k: v for k, v in extra.items() if v is not None}}
        return self.request("POST", "/v1/s2/step", body)

    def final_approach(self, session_id: str, **extra: Any) -> Dict[str, Any]:
        body = {"session_id": session_id, **{k: v for k, v in extra.items() if v is not None}}
        return self.request("POST", "/v1/final/approach", body)
