"""
Cartesia Ink streaming STT — WebSocket client for Twilio µ-law @ 8 kHz.

Sends raw µ-law bytes as binary frames; receives JSON with type=transcript
and is_final. See Cartesia STT streaming docs.

Model default is ``ink-whisper`` (``CARTESIA_STT_MODEL``). On 8 kHz phone audio,
short homophones are often wrong (e.g. "sole" → "stole" / "phone"). Tunables:
``CARTESIA_STT_MAX_SILENCE_SECS`` (higher = wait longer before ``is_final``, better for full addresses; default 0.45 for PSTN),
``CARTESIA_STT_LANGUAGE`` (en).
The agent also maps common STT garbles in ``state_manager.infer_ownership_from_transcript``.
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


def _build_stt_websocket_url() -> str:
    # Lower = faster turn-end (snappier), higher = fewer mid-sentence cuts.
    # Recommended: 0.25–0.30 for local mic, 0.40–0.55 for Twilio PSTN.
    silence = max(0.20, float(os.getenv("CARTESIA_STT_MAX_SILENCE_SECS", "0.45")))
    q: dict[str, str] = {
        "model": os.getenv("CARTESIA_STT_MODEL", "ink-whisper"),
        "language": os.getenv("CARTESIA_STT_LANGUAGE", "en"),
        "encoding": "pcm_mulaw",
        "sample_rate": "8000",
        "min_volume": os.getenv("CARTESIA_STT_MIN_VOLUME", "0.0"),
        "max_silence_duration_secs": str(silence),
        "cartesia_version": os.getenv("CARTESIA_VERSION", "2026-03-01"),
    }
    return f"wss://api.cartesia.ai/stt/websocket?{urlencode(q)}"


class CartesiaInkSTTSession:
    """Single-call STT session; send µ-law chunks, iterate JSON events from server."""

    def __init__(self, call_logger: CallLogger) -> None:
        self._call_logger = call_logger
        self._ws: Optional[WebSocketClientProtocol] = None
        self._closed = False

    async def connect(self) -> None:
        api_key = os.getenv("CARTESIA_API_KEY", "")
        if not api_key:
            raise RuntimeError("CARTESIA_API_KEY is required for Cartesia STT")

        url = _build_stt_websocket_url()
        self._ws = await websockets.connect(
            url,
            extra_headers={
                "Authorization": f"Bearer {api_key}",
                "Cartesia-Version": os.getenv("CARTESIA_VERSION", "2026-03-01"),
            },
            ping_interval=20,
            ping_timeout=20,
            max_size=2**20,
        )
        self._call_logger.info(
            (
                "Cartesia STT WebSocket connected | "
                f"max_silence={os.getenv('CARTESIA_STT_MAX_SILENCE_SECS', '0.45')} "
                f"min_volume={os.getenv('CARTESIA_STT_MIN_VOLUME', '0.0')} "
                f"model={os.getenv('CARTESIA_STT_MODEL', 'ink-whisper')}"
            ),
            event="cartesia.stt.connected",
        )

    async def send_mulaw_bytes(self, data: bytes) -> bool:
        """
        Send one chunk of µ-law audio. Returns False if the STT socket is no longer
        usable (closed locally, by Cartesia, or mid-flight). Caller should stop
        forwarding Twilio media when this returns False.
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
                event="cartesia.stt.send_closed",
            )
            return False

    async def iter_events(self) -> AsyncIterator[dict[str, Any]]:
        """Yield parsed JSON objects from the STT server (text or binary ignored)."""
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
                yield msg
        except Exception as e:
            if not self._closed:
                self._call_logger.warning(
                    f"STT recv ended: {e}",
                    event="cartesia.stt.recv_ended",
                )

    async def close(self) -> None:
        ws = self._ws
        if ws:
            try:
                await ws.send("finalize")
                await ws.send("done")
            except Exception:
                pass
            try:
                await ws.close()
            except Exception:
                pass
            self._ws = None
        self._closed = True
        self._call_logger.info("Cartesia STT session closed", event="cartesia.stt.closed")
