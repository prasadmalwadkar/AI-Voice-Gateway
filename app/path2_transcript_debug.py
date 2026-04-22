"""
Path2 live STT transcript fan-out for local debugging (orchestrator CLI).

Subscribe: ``WS /debug/path2-transcripts``. Server pushes JSON:
``{"kind": str, "text": str, "call_sid": str|null}``.

Kinds: ``user_partial``, ``user_final``, ``user_final_echo_skip``,
``user_final_noise_dropped`` (when bridge drops junk finals),
``agent_tts`` (text sent to Sonic for speech), ``agent_fallback`` (tool Ask-next shortcut),
``call_ended`` (session finished — close mimic / UI; ``text`` holds reason slug).

Disable broadcasting with ``PATH2_TRANSCRIPT_DEBUG_WS=0`` (connections get no messages).

**Deeper debug** (LLM re-ask / Continue stuck): set ``PATH2_DEBUG_DETAILED=1``.
You get structured ``path2_debug`` events to the same WS and ``voip.path2_debug`` log:
browser sync phase ms, prior-task drain, STT→Line handoff, optional DOM snapshot on
incomplete register form.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import time
from typing import Any, Set

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

_clients: Set[WebSocket] = set()
_lock = asyncio.Lock()
_path2_debug = logging.getLogger("voip.path2_debug")
_call_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("path2_call", default=None)


def path2_transcript_debug_enabled() -> bool:
    v = os.getenv("PATH2_TRANSCRIPT_DEBUG_WS", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def path2_debug_detailed() -> bool:
    return os.getenv("PATH2_DEBUG_DETAILED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def path2_set_debug_call(sid: str | None) -> contextvars.Token[str | None]:
    return _call_ctx.set(sid)


def path2_reset_debug_call(token: contextvars.Token[str | None]) -> None:
    _call_ctx.reset(token)


def _path2_debug_get_call() -> str | None:
    return _call_ctx.get()


def path2_debug_envelope(
    event: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    t = time.time()
    env: dict[str, Any] = {
        "event": "path2_debug",
        "kind": event,
        "call_sid": _path2_debug_get_call(),
        "ts": t,
    }
    if extra:
        env["data"] = extra
    return env


async def path2_debug_log(event: str, **extra: Any) -> None:
    """If PATH2_DEBUG_DETAILED=1: same as path2_debug_log_sync, plus WebSocket to debug clients when enabled."""
    if not path2_debug_detailed():
        return
    e = path2_debug_envelope(event, extra=extra)
    _path2_debug.info("path2_debug | %s", json.dumps(e, default=str, ensure_ascii=False))
    if not path2_transcript_debug_enabled():
        return
    text = json.dumps(
        {
            "kind": "path2_debug",
            "text": e.get("kind", ""),
            "call_sid": e.get("call_sid"),
            "envelope": e,
        },
        default=str,
        ensure_ascii=False,
    )
    async with _lock:
        if not _clients:
            return
        snapshot = list(_clients)
    dead: list[WebSocket] = []
    for ws in snapshot:
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    if dead:
        async with _lock:
            for ws in dead:
                _clients.discard(ws)


async def register_client(websocket: WebSocket) -> None:
    """Register an already-accepted WebSocket (route handler calls ``accept()`` once)."""
    async with _lock:
        _clients.add(websocket)


async def unregister_client(websocket: WebSocket) -> None:
    async with _lock:
        _clients.discard(websocket)


async def broadcast_stt(
    *,
    kind: str,
    text: str,
    call_sid: str | None = None,
) -> None:
    if not path2_transcript_debug_enabled():
        return
    async with _lock:
        if not _clients:
            return
        snapshot = list(_clients)
    payload = json.dumps(
        {"kind": kind, "text": text, "call_sid": call_sid},
        ensure_ascii=False,
    )
    dead: list[WebSocket] = []
    for ws in snapshot:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    if dead:
        async with _lock:
            for ws in dead:
                _clients.discard(ws)


async def keepalive_loop(websocket: WebSocket) -> None:
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
