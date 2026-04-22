"""
Twilio Media Streams ↔ Cartesia Ink STT ↔ Line /ws (LlmAgent) ↔ Sonic TTS.

Path2 single-brain pipeline: no Cartesia Calls API.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
import uuid
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from agent.state_manager import (
    apply_voice_intake_heuristics,
    apply_voice_intake_heuristics_before_line,
    ensure_browser_ready,
    get_missing_fields,
    reset_state,
)
from app.cartesia_stt_client import CartesiaInkSTTSession
from app.cartesia_tts_mulaw import CartesiaSonicMulawTTS
from app.line_session_client import LineSessionClient
from app.logging_config import CallLogger
from app.path2_stt_noise import path2_stt_final_is_drop_noise
from app.path2_transcript_debug import (
    broadcast_stt,
    path2_debug_log,
    path2_reset_debug_call,
    path2_set_debug_call,
)

ECHO_GUARD_SECONDS = max(0.30, float(os.getenv("TWILIO_PATH2_ECHO_GUARD_SECS", "0.45")))
BARGE_IN_COOLDOWN_SECS = 0.35
ANSWER_GUARD_SECS = float(os.getenv("TWILIO_PATH2_ANSWER_GUARD_SECS", "2.0"))
DETERMINISTIC_GOODBYE_TEXT = (
    "Thanks, we have all the information we need. Goodbye!"
)
DETERMINISTIC_GOODBYE_DRAIN_SEC = float(
    os.getenv("DETERMINISTIC_GOODBYE_DRAIN_SEC", "6.0")
)
DETERMINISTIC_FOLLOWUP_POLL_SEC = 0.25
DETERMINISTIC_FOLLOWUP_MAX_WAIT_SEC = 8.0


def _is_status_check_utterance(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    t = re.sub(r"[^\w\s?']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    patterns = (
        r"\b(you there|are you there|still there|hello)\b",
        r"\b(what'?s next|next question|go ahead|continue)\b",
        r"\b(what'?s the question|what is the question|which question)\b",
        r"\b(repeat|say that again|say again|come again)\b",
        r"\b(i didn'?t hear|didn'?t catch|what did you say)\b",
        r"\b(i'?m lost|i'?m confused|what do you need|what do i do)\b",
        r"\b(pardon|excuse me|sorry what)\b",
    )
    return any(re.search(p, t) for p in patterns)


class TwilioLineBridge:
    """Bridge one Twilio call through Ink STT + Line LlmAgent + Sonic TTS."""

    def __init__(
        self,
        twilio_ws: WebSocket,
        call_logger: CallLogger,
    ) -> None:
        self.twilio_ws = twilio_ws
        self.call_logger = call_logger

        self.stream_sid: Optional[str] = None
        self.call_sid: Optional[str] = None

        self._running = False
        self._session_ready = False
        self._first_audio_sent = False

        self._stt: Optional[CartesiaInkSTTSession] = None
        self._line: Optional[LineSessionClient] = None
        self._tts: Optional[CartesiaSonicMulawTTS] = None

        self._stt_task: Optional[asyncio.Task[None]] = None
        self._tts_task: Optional[asyncio.Task[None]] = None
        self._end_watch_task: Optional[asyncio.Task[None]] = None
        self._deterministic_followup_task: Optional[asyncio.Task[None]] = None

        self._agent_playing = False
        self._last_agent_audio_time = 0.0
        self._last_barge_at = 0.0
        self._deterministic_answer_guard_until = 0.0

    async def run(self) -> None:
        self._running = True
        try:
            while self._running:
                try:
                    raw = await self.twilio_ws.receive_text()
                except WebSocketDisconnect:
                    self.call_logger.info(
                        "Twilio disconnected (caller hung up)",
                        event="call.dropped",
                    )
                    break

                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                event_type = message.get("event")

                if event_type == "connected":
                    self.call_logger.info(
                        "Twilio transport connected",
                        event="twilio.connected",
                    )

                elif event_type == "start":
                    start_data = message.get("start", {})
                    self.stream_sid = start_data.get("streamSid")
                    self.call_sid = start_data.get("callSid", self.call_sid)
                    if self.call_sid:
                        self.call_logger.call_sid = self.call_sid
                    self.call_logger.info(
                        f"Media stream started | stream_sid={self.stream_sid}",
                        event="twilio.stream.start",
                        stream_sid=self.stream_sid,
                    )
                    self.call_logger.mark_connected()
                    if not self._session_ready:
                        await self._setup_session()

                elif event_type == "media" and self._session_ready and self._stt:
                    payload = message["media"]["payload"]
                    data = base64.b64decode(payload)
                    sent = await self._stt.send_mulaw_bytes(data)
                    if not sent:
                        # STT socket closed (Cartesia end or race with shutdown) —
                        # stop forwarding; Twilio side will disconnect or send stop next.
                        self.call_logger.info(
                            "STT closed — ending Twilio receive loop",
                            event="path2.stt.closed_forward",
                        )
                        break

                elif event_type == "stop":
                    self.call_logger.info(
                        "Twilio media stream stopped",
                        event="twilio.stream.stop",
                    )
                    break

        except WebSocketDisconnect:
            self.call_logger.info(
                "Twilio disconnected",
                event="call.dropped",
            )
        except Exception as e:
            self.call_logger.error(
                f"TwilioLineBridge error: {type(e).__name__}: {e}",
                event="path2.error",
                exc_info=True,
            )
            raise
        finally:
            self._running = False
            await self._cleanup()

    async def _setup_session(self) -> None:
        self._session_ready = True
        port = int(os.getenv("PORT", "8000"))
        line_url = f"ws://127.0.0.1:{port}/ws"

        call_id = self.call_sid or str(uuid.uuid4())
        from_num = self.call_sid or "unknown"
        to_num = os.getenv("TWILIO_PATH2_TO_NUMBER", "west-capital")

        self._stt = CartesiaInkSTTSession(self.call_logger)
        await self._stt.connect()

        self._line = LineSessionClient(
            line_url,
            self.call_logger,
            call_id=call_id,
            from_number=from_num,
            to_number=to_num,
        )
        await self._line.connect()

        # Browser only started inside update_application_state when there are field
        # updates; open Playwright as soon as the call/session is live so the form
        # is visible during the intro (runs in parallel — does not block TTS).
        if os.getenv("TWILIO_PATH2_OPEN_BROWSER", "1").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            asyncio.create_task(
                self._launch_playwright_session(),
                name=f"path2_browser:{call_id[:8]}",
            )

        self._tts = CartesiaSonicMulawTTS(self.call_logger)
        await self._tts.connect()

        self._stt_task = asyncio.create_task(
            self._consume_stt(),
            name=f"path2_stt:{call_id[:8]}",
        )
        self._tts_task = asyncio.create_task(
            self._consume_tts_queue(),
            name=f"path2_tts:{call_id[:8]}",
        )
        self._end_watch_task = asyncio.create_task(
            self._watch_line_end_call(),
            name=f"path2_end:{call_id[:8]}",
        )

        self.call_logger.info(
            f"Path2 session ready | line={line_url}",
            event="path2.session.ready",
        )

    async def _launch_playwright_session(self) -> None:
        try:
            await ensure_browser_ready()
            self.call_logger.info(
                "Playwright browser up (HELOC register page)",
                event="path2.browser.ready",
            )
        except Exception as e:
            self.call_logger.warning(
                f"Playwright failed to start — form sync waits for first tool save: {e}",
                event="path2.browser.failed",
                exc_info=True,
            )

    async def _watch_line_end_call(self) -> None:
        if not self._line:
            return
        await self._line.end_call_event.wait()
        self._running = False

    async def _consume_stt(self) -> None:
        if not self._stt or not self._line:
            return
        try:
            async for msg in self._stt.iter_events():
                if not self._running:
                    break
                mtype = msg.get("type")
                if mtype == "transcript":
                    text = (msg.get("text") or "").strip()
                    is_final = bool(msg.get("is_final"))
                    if not text:
                        continue

                    now = time.monotonic()
                    in_echo = (now - self._last_agent_audio_time) < ECHO_GUARD_SECONDS

                    if not is_final:
                        await broadcast_stt(
                            kind="user_partial",
                            text=text,
                            call_sid=self.call_sid,
                        )
                        if (
                            not in_echo
                            and self._agent_playing
                            and (now - self._last_barge_at) >= BARGE_IN_COOLDOWN_SECS
                        ):
                            self._last_barge_at = now
                            await self._barge_in()
                        continue

                    if in_echo:
                        self.call_logger.debug(
                            "Skipping STT final during echo guard",
                            event="path2.stt.echo_skip",
                        )
                        await broadcast_stt(
                            kind="user_final_echo_skip",
                            text=text,
                            call_sid=self.call_sid,
                        )
                        continue

                    if path2_stt_final_is_drop_noise(text):
                        await broadcast_stt(
                            kind="user_final_noise_dropped",
                            text=text,
                            call_sid=self.call_sid,
                        )
                        continue

                    if now < self._deterministic_answer_guard_until:
                        self.call_logger.debug(
                            "Skipping STT final during answer guard",
                            event="path2.stt.answer_guard_skip",
                        )
                        await broadcast_stt(
                            kind="user_final_answer_guard_skip",
                            text=text,
                            call_sid=self.call_sid,
                        )
                        continue

                    await broadcast_stt(
                        kind="user_final",
                        text=text,
                        call_sid=self.call_sid,
                    )
                    ctx = path2_set_debug_call(self.call_sid)
                    t0 = time.perf_counter()
                    try:
                        result = await apply_voice_intake_heuristics_before_line(text)
                        if result.handled and self._line:
                            self._deterministic_answer_guard_until = (
                                time.monotonic() + ANSWER_GUARD_SECS
                            )
                            self._cancel_deterministic_followup_task()
                            await self._line.mark_deterministic_turn(result.ask_next)
                            if result.ask_next is None:
                                missing_now = get_missing_fields() or ""
                                if not missing_now.strip():
                                    await self._send_deterministic_goodbye()
                                else:
                                    self._schedule_deterministic_followup_ask()
                            await broadcast_stt(
                                kind="deterministic_handled",
                                text=text,
                                call_sid=self.call_sid,
                            )
                            await path2_debug_log(
                                "path2.deterministic_handled",
                                field=result.handled_field,
                                stage=result.stage,
                                text_preview=(text or "")[:160],
                                total_ms=round((time.perf_counter() - t0) * 1000, 2),
                            )
                        else:
                            missing_snapshot = get_missing_fields() or ""
                            ask_from_missing = self._extract_ask_next(missing_snapshot)
                            if (
                                ask_from_missing
                                and _is_status_check_utterance(text)
                                and self._line
                            ):
                                await self._line.mark_deterministic_turn(ask_from_missing)
                                continue
                            if result.ask_next and self._line:
                                self._line.inject_ask_next_hint(result.ask_next)
                            t0_line = time.perf_counter()
                            await self._line.send_user_transcript(text)
                            t_after_line = time.perf_counter()
                            await apply_voice_intake_heuristics(text)
                            line_user_ms = round((t_after_line - t0_line) * 1000, 2)
                            heuristics_ms = round(
                                (time.perf_counter() - t_after_line) * 1000, 2
                            )
                            await path2_debug_log(
                                "path2.stt_final_to_line",
                                heuristics_ms=heuristics_ms,
                                line_user_send_ms=line_user_ms,
                                total_handoff_ms=round((time.perf_counter() - t0) * 1000, 2),
                                text_preview=(text or "")[:160],
                            )
                    finally:
                        path2_reset_debug_call(ctx)

                elif mtype == "error":
                    self.call_logger.warning(
                        f"STT error frame: {msg}",
                        event="cartesia.stt.error_frame",
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if self._running:
                self.call_logger.warning(
                    f"STT consumer ended: {e}",
                    event="path2.stt.consumer_end",
                )

    async def _barge_in(self) -> None:
        self.call_logger.info("Barge-in: cancel TTS + Line + Twilio clear", event="path2.barge_in")
        if self._tts:
            await self._tts.cancel_current()
        self._agent_playing = False
        if self._line:
            await self._line.send_user_speaking_start()
        await self._send_twilio_clear()

    async def _send_deterministic_goodbye(self) -> None:
        if not self._line:
            return
        await self._line.tts_queue.put(DETERMINISTIC_GOODBYE_TEXT)
        self.call_logger.info(
            "Deterministic goodbye enqueued",
            event="path2.deterministic_goodbye",
        )
        await asyncio.sleep(DETERMINISTIC_GOODBYE_DRAIN_SEC)
        if self._line:
            self._line.end_call_event.set()

    @staticmethod
    def _extract_ask_next(text: str) -> Optional[str]:
        m = re.search(r'Ask next:\s*"([^"]+)"', text or "")
        if not m:
            return None
        ask = (m.group(1) or "").strip()
        return ask or None

    def _cancel_deterministic_followup_task(self) -> None:
        t = self._deterministic_followup_task
        if t and not t.done():
            t.cancel()
        self._deterministic_followup_task = None

    def _schedule_deterministic_followup_ask(self) -> None:
        if not self._line:
            return
        self._cancel_deterministic_followup_task()
        self._deterministic_followup_task = asyncio.create_task(
            self._wait_and_queue_deterministic_followup_ask(),
            name="path2_det_followup_ask",
        )

    async def _wait_and_queue_deterministic_followup_ask(self) -> None:
        if not self._line:
            return
        attempts = max(
            1, int(DETERMINISTIC_FOLLOWUP_MAX_WAIT_SEC / DETERMINISTIC_FOLLOWUP_POLL_SEC)
        )
        try:
            for _ in range(attempts):
                await asyncio.sleep(DETERMINISTIC_FOLLOWUP_POLL_SEC)
                if not self._running or not self._line:
                    return
                missing = get_missing_fields() or ""
                ask = self._extract_ask_next(missing)
                if ask:
                    await self._line.mark_deterministic_turn(ask)
                    self.call_logger.info(
                        "Queued deterministic follow-up ask after async navigation",
                        event="path2.deterministic_followup.ask",
                    )
                    return
                if not missing.strip():
                    await self._send_deterministic_goodbye()
                    return
        except asyncio.CancelledError:
            raise

    async def _consume_tts_queue(self) -> None:
        if not self._line or not self._tts:
            return
        while self._running:
            if self._line.end_call_event.is_set():
                break
            try:
                text = await asyncio.wait_for(self._line.tts_queue.get(), timeout=0.3)
            except asyncio.TimeoutError:
                continue

            await broadcast_stt(
                kind="agent_tts",
                text=text,
                call_sid=self.call_sid,
            )
            self._agent_playing = True
            self._deterministic_answer_guard_until = 0.0
            try:
                await self._tts.speak_text(
                    text,
                    self._send_mulaw_to_twilio,
                )
            finally:
                self._agent_playing = False
                await self._send_twilio_clear()
                if (text or "").strip() == DETERMINISTIC_GOODBYE_TEXT.strip():
                    await broadcast_stt(
                        kind="call_ended",
                        text="deterministic_goodbye",
                        call_sid=self.call_sid,
                    )

    async def _send_mulaw_to_twilio(self, mulaw: bytes) -> None:
        self._last_agent_audio_time = time.monotonic()
        payload = base64.b64encode(mulaw).decode("ascii")
        await self._send_twilio_media(payload)

    async def _send_twilio_media(self, base64_audio: str) -> None:
        if not self.stream_sid:
            return
        if not self._first_audio_sent:
            self._first_audio_sent = True
            self.call_logger.mark_first_audio()

        media_event = {
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {"payload": base64_audio},
        }
        try:
            await self.twilio_ws.send_text(json.dumps(media_event))
        except (WebSocketDisconnect, RuntimeError):
            self.call_logger.info(
                "Cannot send audio — Twilio WS closed",
                event="twilio.send_closed",
            )
            self._running = False

    async def _send_twilio_clear(self) -> None:
        if not self.stream_sid:
            return
        try:
            await self.twilio_ws.send_text(
                json.dumps({"event": "clear", "streamSid": self.stream_sid})
            )
        except Exception:
            pass

    async def _cleanup(self) -> None:
        self._running = False
        try:
            await broadcast_stt(
                kind="call_ended",
                text="cleanup",
                call_sid=self.call_sid,
            )
        except Exception:
            pass

        self._cancel_deterministic_followup_task()
        for t in (self._stt_task, self._tts_task, self._end_watch_task):
            if t and not t.done():
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

        self._stt_task = None
        self._tts_task = None
        self._end_watch_task = None

        if self._line:
            try:
                await self._line.close()
            except Exception:
                pass
            self._line = None

        if self._tts:
            try:
                await self._tts.close()
            except Exception:
                pass
            self._tts = None

        if self._stt:
            try:
                await self._stt.close()
            except Exception:
                pass
            self._stt = None

        try:
            await reset_state()
        except Exception:
            pass

        self.call_logger.info(
            f"Path2 cleanup | TTFA={self.call_logger.ttfa_ms}ms",
            event="path2.cleanup",
        )
