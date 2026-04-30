"""
Per-call conversation transcript (user STT finals + agent TTS lines).

NDJSON: one JSON object per line in ``{UTC-datetime}_{call_sid}.jsonl``
(datetime prefix set on first append — sortable by recency).
Lines may include ``role: ui`` with ``scope`` (``page`` during call = current wizard step only;
``full`` once at hang-up = entire captured snapshot), optional ``browser_stage``, and ``fields``.

End-of-call **voice field map** (schema field → verbatim user utterance last accepted for that field):
``<dir>/voice_fields/<call_sid>.json``. Written when ``scope: full`` snapshot runs; omitted when
transcripts disabled.

Env:
  ``CONVERSATION_TRANSCRIPT_DIR`` — unset → ``data/conversations``;
  set to empty → disabled; otherwise use that path.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

Role = Literal["user", "agent"]

# Stable path per call_sid for duration of process (first append picks UTC stamp in filename).
_call_transcript_paths: dict[str, Path] = {}


def _transcript_dir() -> Optional[Path]:
    raw = os.getenv("CONVERSATION_TRANSCRIPT_DIR")
    if raw is None:
        return Path("data/conversations")
    s = raw.strip()
    if not s:
        return None
    return Path(s)


def append_transcript_turn(call_sid: str, role: Role, text: str) -> None:
    """Append one transcript line. No-op if dir disabled or ``call_sid`` / text empty."""
    base = _transcript_dir()
    if base is None or not call_sid:
        return
    t = (text or "").strip()
    if not t:
        return
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "text": t,
    }
    if call_sid not in _call_transcript_paths:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        _call_transcript_paths[call_sid] = base / f"{stamp}_{call_sid}.jsonl"
    path = _call_transcript_paths[call_sid]
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


def append_transcript_ui_fields(
    call_sid: str | None,
    fields: dict[str, str],
    *,
    scope: str = "page",
    browser_stage: str | None = None,
) -> None:
    """Append browser/UI field snapshot (same file as user/agent lines).

    ``scope`` is ``page`` while the call progresses (current wizard step only) or
    ``full`` once at hang-up (entire captured application snapshot).
    """
    from app.path2_transcript_debug import _path2_debug_get_call

    sid = (call_sid or "").strip() or (_path2_debug_get_call() or "")
    if not sid:
        return
    base = _transcript_dir()
    if base is None:
        return
    clean: dict[str, str] = {}
    for k, v in (fields or {}).items():
        if v is None:
            if scope == "full":
                clean[k] = ""
            continue
        s = str(v).strip()
        if not s:
            if scope == "full":
                clean[k] = ""
            continue
        clean[k] = s
    if not clean:
        return
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    rec: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "role": "ui",
        "scope": scope,
        "fields": clean,
    }
    if browser_stage:
        rec["browser_stage"] = browser_stage
    if sid not in _call_transcript_paths:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        _call_transcript_paths[sid] = base / f"{stamp}_{sid}.jsonl"
    path = _call_transcript_paths[sid]
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


def write_voice_sourced_fields_json(
    call_sid: str | None,
    fields: dict[str, str],
) -> None:
    """Write field → verbatim user transcript map at call end (not UI-normalized values).

    Stored under ``<CONVERSATION_TRANSCRIPT_DIR>/voice_fields/<call_sid>.json``.
    """
    base = _transcript_dir()
    if base is None:
        return
    sid = (call_sid or "").strip()
    if not sid:
        return
    try:
        out_dir = base / "voice_fields"
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    payload = {
        "call_sid": sid,
        "ended_ts": datetime.now(timezone.utc).isoformat(),
        "fields": dict(sorted((fields or {}).items())),
    }
    path = out_dir / f"{sid}.json"
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass
