"""
Cartesia Sonic TTS over WebSocket — stream µ-law @ 8 kHz for Twilio Media Streams.

Uses AsyncCartesia.tts.websocket_connect; one connection per call, new context per utterance.
"""

from __future__ import annotations

import os
import time
from typing import Any, Awaitable, Callable, Dict, Optional

from cartesia import AsyncCartesia

from app.logging_config import CallLogger

_TTS_MAX_WALL_SEC = float(os.getenv("CARTESIA_TTS_MAX_WALL_SEC", "15"))
_TTS_IDLE_SEC = float(os.getenv("CARTESIA_TTS_IDLE_SEC", "4"))


OutputFormatMulaw8k = Dict[str, Any]


def default_mulaw_output_format() -> OutputFormatMulaw8k:
    return {
        "container": "raw",
        "encoding": "pcm_mulaw",
        "sample_rate": 8000,
    }


def default_voice() -> Dict[str, str]:
    vid = os.getenv("CARTESIA_TTS_VOICE_ID", "6ccbfb76-1fc6-48f7-b71d-91ac6298247b")
    return {"mode": "id", "id": vid}


class CartesiaSonicMulawTTS:
    """Manage a long-lived TTS WebSocket and synthesize text to 8 kHz µ-law chunks."""

    def __init__(self, call_logger: CallLogger) -> None:
        self._call_logger = call_logger
        self._client: Optional[AsyncCartesia] = None
        self._manager: Any = None
        self._conn: Any = None
        self._current_ctx: Any = None

    async def connect(self) -> None:
        api_key = os.getenv("CARTESIA_API_KEY", "")
        if not api_key:
            raise RuntimeError("CARTESIA_API_KEY is required for Cartesia TTS")

        self._client = AsyncCartesia(api_key=api_key)
        self._manager = self._client.tts.websocket_connect()
        self._conn = await self._manager.__aenter__()
        self._call_logger.info("Cartesia TTS WebSocket connected", event="cartesia.tts.connected")

    async def close(self) -> None:
        if self._conn is not None:
            try:
                await self._conn.close()
            except Exception:
                pass
            self._conn = None
        if self._manager is not None:
            try:
                await self._manager.__aexit__(None, None, None)
            except Exception:
                pass
            self._manager = None
        self._client = None
        self._call_logger.info("Cartesia TTS closed", event="cartesia.tts.closed")

    async def cancel_current(self) -> None:
        ctx = self._current_ctx
        if ctx is not None:
            try:
                await ctx.cancel()
            except Exception:
                pass
            self._current_ctx = None

    async def speak_text(
        self,
        text: str,
        on_mulaw_chunk: Callable[[bytes], Awaitable[None]],
    ) -> None:
        """
        Synthesize `text` and invoke callback for each raw µ-law chunk (decoded from API).
        """
        if not text.strip() or self._conn is None:
            return

        model_id = os.getenv("CARTESIA_TTS_MODEL_ID", "sonic-3")
        voice = default_voice()
        output_format = default_mulaw_output_format()

        ctx = self._conn.context(
            model_id=model_id,
            voice=voice,
            output_format=output_format,
        )
        self._current_ctx = ctx

        t0 = time.monotonic()
        chunks = 0
        last_chunk_at = t0
        try:
            await ctx.send(
                model_id=model_id,
                transcript=text,
                voice=voice,
                output_format=output_format,
                continue_=False,
            )
            async for event in ctx.receive():
                now = time.monotonic()
                if now - t0 > _TTS_MAX_WALL_SEC:
                    self._call_logger.warning(
                        f"TTS wall timeout ({_TTS_MAX_WALL_SEC}s) after {chunks} chunks",
                        event="cartesia.tts.wall_timeout",
                    )
                    break
                if now - last_chunk_at > _TTS_IDLE_SEC:
                    self._call_logger.warning(
                        f"TTS idle timeout ({_TTS_IDLE_SEC}s) after {chunks} chunks",
                        event="cartesia.tts.idle_timeout",
                    )
                    break
                et = getattr(event, "type", None)
                if et == "chunk":
                    audio = event.audio
                    if audio:
                        chunks += 1
                        last_chunk_at = now
                        await on_mulaw_chunk(audio)
                elif et == "error":
                    err = getattr(event, "error", str(event))
                    self._call_logger.warning(
                        f"TTS chunk error: {err}",
                        event="cartesia.tts.chunk_error",
                    )
                    break
            elapsed = time.monotonic() - t0
            self._call_logger.debug(
                f"TTS done: {chunks} chunks in {elapsed:.2f}s",
                event="cartesia.tts.utterance_done",
            )
        finally:
            self._current_ctx = None
