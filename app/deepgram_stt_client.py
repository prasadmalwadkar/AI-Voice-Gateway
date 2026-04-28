"""
Deepgram Flux streaming STT - WebSocket client for Twilio mu-law @ 8 kHz.

This module mirrors ``CartesiaInkSTTSession`` so the Twilio bridge can switch
STT providers without changing downstream Line, state, or TTS behavior.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Optional
from urllib.parse import urlencode

import websockets
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosed

from app.logging_config import CallLogger


DEEPGRAM_FLUX_EVENTS = {
    "StartOfTurn",
    "Update",
    "EagerEndOfTurn",
    "TurnResumed",
    "EndOfTurn",
}


def _getenv_float(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    float(value)
    return value


def _getenv_int(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    int(value)
    return value


def _build_deepgram_websocket_url() -> str:
    q: dict[str, Any] = {
        "model": os.getenv("DEEPGRAM_STT_MODEL", "flux-general-en"),
        "encoding": "mulaw",
        "sample_rate": "8000",
        "eot_threshold": _getenv_float("DEEPGRAM_STT_EOT_THRESHOLD", "0.7"),
        "eot_timeout_ms": _getenv_int("DEEPGRAM_STT_EOT_TIMEOUT_MS", "3000"),
    }

    eager = os.getenv("DEEPGRAM_STT_EAGER_EOT_THRESHOLD", "0.4").strip()
    if eager:
        float(eager)
        q["eager_eot_threshold"] = eager

    keyterms = [
        term.strip()
        for term in os.getenv("DEEPGRAM_STT_KEYTERMS", "").split(",")
        if term.strip()
    ]
    if keyterms:
        q["keyterm"] = keyterms

    return f"wss://api.deepgram.com/v2/listen?{urlencode(q, doseq=True)}"


def _normalize_flux_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    event = msg.get("event")
    if event not in DEEPGRAM_FLUX_EVENTS:
        if msg.get("type") in {"Error", "FatalError"}:
            return {"type": "error", "provider": "deepgram", "raw": msg}
        return None

    text = (msg.get("transcript") or "").strip()
    return {
        "type": "transcript",
        "text": text,
        "is_final": event == "EndOfTurn",
        "provider": "deepgram",
        "deepgram_event": event,
        "confidence": msg.get("end_of_turn_confidence"),
        "raw": msg,
    }


class DeepgramFluxSTTSession:
    """Single-call STT session; send mu-law chunks, yield Ink-style events."""

    def __init__(self, call_logger: CallLogger) -> None:
        self._call_logger = call_logger
        self._ws: Optional[WebSocketClientProtocol] = None
        self._closed = False

    async def connect(self) -> None:
        api_key = os.getenv("DEEPGRAM_API_KEY", "")
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is required for Deepgram STT")

        url = _build_deepgram_websocket_url()
        self._ws = await websockets.connect(
            url,
            extra_headers={"Authorization": f"Token {api_key}"},
            ping_interval=20,
            ping_timeout=20,
            max_size=2**20,
        )
        self._call_logger.info(
            (
                "Deepgram STT WebSocket connected | "
                f"model={os.getenv('DEEPGRAM_STT_MODEL', 'flux-general-en')} "
                f"eot_threshold={os.getenv('DEEPGRAM_STT_EOT_THRESHOLD', '0.7')} "
                f"eager_eot_threshold={os.getenv('DEEPGRAM_STT_EAGER_EOT_THRESHOLD', '0.4')} "
                f"eot_timeout_ms={os.getenv('DEEPGRAM_STT_EOT_TIMEOUT_MS', '3000')}"
            ),
            event="deepgram.stt.connected",
        )

    async def send_mulaw_bytes(self, data: bytes) -> bool:
        """
        Send one mu-law audio chunk. Returns False when the STT socket is no
        longer usable.
        """
        if self._ws is None or self._closed:
            return False
        try:
            await self._ws.send(data)
            return True
        except ConnectionClosed:
            self._closed = True
            self._call_logger.info(
                "STT WebSocket closed during send (stop forwarding)",
                event="deepgram.stt.send_closed",
            )
            return False

    async def iter_events(self) -> AsyncIterator[dict[str, Any]]:
        """Yield normalized transcript/error objects from Deepgram Flux."""
        if self._ws is None:
            return
        try:
            async for raw in self._ws:
                if self._closed:
                    break
                if isinstance(raw, bytes):
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                normalized = _normalize_flux_message(msg)
                if normalized is not None:
                    yield normalized
        except Exception as e:
            if not self._closed:
                self._call_logger.warning(
                    f"STT recv ended: {e}",
                    event="deepgram.stt.recv_ended",
                )

    async def close(self) -> None:
        ws = self._ws
        if ws:
            try:
                await ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass
            try:
                await ws.close()
            except Exception:
                pass
            self._ws = None
        self._closed = True
        self._call_logger.info("Deepgram STT session closed", event="deepgram.stt.closed")
