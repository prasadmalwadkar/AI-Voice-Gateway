"""
Per-call intake audit trail (PII).

Plan (extend incrementally):
- **Storage:** one NDJSON file per call, e.g. ``logs/intake_audit/{call_sid}.jsonl``,
  or a single merged file with ``call_sid`` on every line.
- **Events:** ``user_final``, ``deterministic_preline``, ``tool_args`` (sanitized),
  ``tool_result_excerpt``, ``tts_out``, ``stage_change``, ``error``.
- **Payload:** ``ts_iso``, ``call_sid``, ``seq``, ``browser_stage``, ``user_text``,
  ``missing_fields_excerpt``, ``handled_field`` (if any), ``state_patch`` (field names
  only or redacted values — your compliance choice).
- **Never log:** ``password``, ``confirm_password``; strip from any dict before write.
- **Retention / encryption:** policy outside this module (S3, KMS, 30-day delete, etc.).

Set ``INTAKE_AUDIT_NDJSON_DIR`` to a writable directory to enable append logging.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

_AUDIT_DENY = frozenset({"password", "confirm_password"})


def _audit_dir() -> Optional[Path]:
    raw = (os.getenv("INTAKE_AUDIT_NDJSON_DIR") or "").strip()
    if not raw:
        return None
    return Path(raw)


def _scrub(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: _scrub(v)
            for k, v in obj.items()
            if k not in _AUDIT_DENY
        }
    if isinstance(obj, list):
        return [_scrub(x) for x in obj]
    return obj


def log_intake_event(
    call_sid: str,
    event: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one JSON line if ``INTAKE_AUDIT_NDJSON_DIR`` is set."""
    base = _audit_dir()
    if base is None or not call_sid:
        return
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "call_sid": call_sid,
        "event": event,
        **(_scrub(dict(payload)) if payload else {}),
    }
    path = base / f"{call_sid}.jsonl"
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except OSError:
        pass
