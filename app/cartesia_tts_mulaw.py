"""
Cartesia Sonic TTS over WebSocket — stream µ-law @ 8 kHz for Twilio Media Streams.

Uses AsyncCartesia.tts.websocket_connect; one connection per call, new context per utterance.
"""

from __future__ import annotations

import asyncio
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


def default_generation_config() -> Dict[str, float]:
    """
    Sonic-3 generation_config: speed + volume.

    Speed: [0.6, 1.5] — 0.6 is slowest, 1.0 is Cartesia default.
    Volume: [0.5, 2.0] — boost above 1.0 for PSTN audibility.
    """
    speed = max(0.6, min(1.5, float(os.getenv("CARTESIA_TTS_SPEED", "1.05"))))
    volume = max(0.5, min(2.0, float(os.getenv("CARTESIA_TTS_VOLUME", "2.0"))))
    return {"speed": speed, "volume": volume}


class CartesiaSonicMulawTTS:
    """Manage a long-lived TTS WebSocket and synthesize text to 8 kHz µ-law chunks."""

    def __init__(self, call_logger: CallLogger) -> None:
        self._call_logger = call_logger
        self._client: Optional[AsyncCartesia] = None
        self._manager: Any = None
        self._conn: Any = None
        self._current_ctx: Any = None
        self._stream_task: Optional[asyncio.Task[None]] = None
        self._cancelled = False

    async def connect(self) -> None:
        api_key = os.getenv("CARTESIA_API_KEY", "")
        if not api_key:
            raise RuntimeError("CARTESIA_API_KEY is required for Cartesia TTS")

        self._client = AsyncCartesia(api_key=api_key)
        self._manager = self._client.tts.websocket_connect()
        self._conn = await self._manager.__aenter__()

        gen_cfg = default_generation_config()
        self._call_logger.info(
            f"Cartesia TTS WebSocket connected | generation_config={gen_cfg}",
            event="cartesia.tts.connected",
        )

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
        self._cancelled = True
        ctx = self._current_ctx
        if ctx is not None:
            try:
                await ctx.cancel()
            except Exception:
                pass
            self._current_ctx = None
        task = self._stream_task
        if task is not None and not task.done():
            task.cancel()

    async def speak_text(
        self,
        text: str,
        on_mulaw_chunk: Callable[[bytes], Awaitable[None]],
    ) -> None:
        """
        Synthesize `text` and invoke callback for each raw µ-law chunk (decoded from API).

        The receive loop runs in a child task so that cancel_current() can
        interrupt a blocked ``ctx.receive()`` immediately via task.cancel().
        """
        if not text.strip() or self._conn is None:
            return

        self._cancelled = False

        model_id = os.getenv("CARTESIA_TTS_MODEL_ID", "sonic-3")
        voice = default_voice()
        output_format = default_mulaw_output_format()
        gen_cfg = default_generation_config()

        ctx = self._conn.context(
            model_id=model_id,
            voice=voice,
            output_format=output_format,
            generation_config=gen_cfg,
        )
        self._current_ctx = ctx

        t0 = time.monotonic()
        chunks = 0
        last_chunk_at = t0

        async def _do_stream() -> None:
            nonlocal chunks, last_chunk_at
            await ctx.send(
                model_id=model_id,
                transcript=text,
                voice=voice,
                output_format=output_format,
                continue_=False,
                generation_config=gen_cfg,
            )
            async for event in ctx.receive():
                if self._cancelled:
                    break
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
                    if audio and not self._cancelled:
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

        self._stream_task = asyncio.create_task(_do_stream())
        try:
            await self._stream_task
        except asyncio.CancelledError:
            if not self._cancelled:
                raise
            self._call_logger.info(
                f"TTS cancelled after {chunks} chunks",
                event="cartesia.tts.cancelled",
            )
        finally:
            task = self._stream_task
            self._stream_task = None
            self._current_ctx = None
            if task is not None and not task.done():
                task.cancel()

        elapsed = time.monotonic() - t0
        self._call_logger.debug(
            f"TTS done: {chunks} chunks in {elapsed:.2f}s",
            event="cartesia.tts.utterance_done",
        )
