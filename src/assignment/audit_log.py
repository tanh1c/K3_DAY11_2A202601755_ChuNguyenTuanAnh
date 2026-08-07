"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict[str, object]] = []
        self._open: dict[str, float] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input and start timestamp, returning its correlation ID."""
        request_id = request_id or str(uuid4())
        self._open[request_id] = time.monotonic()
        self.logs.append({
            "request_id": request_id,
            "user_id": user_id,
            "input": text,
            "input_timestamp": utc_now_iso(),
        })
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, decision layer, and latency on the correlated record."""
        request_id = request_id or next(
            (key for key in reversed(self._open)), str(uuid4())
        )
        started = self._open.pop(request_id, None)
        existing = next(
            (item for item in reversed(self.logs) if item["request_id"] == request_id),
            None,
        )
        if existing is None:
            record: dict[str, object] = {
                "request_id": request_id,
                "user_id": user_id,
            }
            self.logs.append(record)
        else:
            record = existing
        record.update({
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "output_timestamp": utc_now_iso(),
            "latency_ms": round((time.monotonic() - started) * 1000, 3)
            if started is not None else None,
        })
        return request_id

    def export_json(self, filepath: str = "outputs/audit_log.json"):
        """Write logs to disk (JSON array)."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
