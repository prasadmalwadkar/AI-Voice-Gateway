"""
Per-call conversation transcript (user STT finals + agent TTS lines).

NDJSON: one JSON object per line in ``{call_sid}.jsonl``.

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
    path = base / f"{call_sid}.jsonl"
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass
