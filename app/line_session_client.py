"""
Line /ws client for Path2: connect to VoiceAgentApp ``/ws``, drive LlmAgent with
transcripts, surface agent text to a TTS queue with debounced streaming chunks.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Optional

import websockets
from websockets.client import WebSocketClientProtocol

from app.logging_config import CallLogger
from app.path2_transcript_debug import broadcast_stt

class LineSessionClient:
    """Async client for the Line VoiceAgentApp websocket protocol."""

    # Re-ask same pending question without replaying long scripted prompts (watchdogs / status).
    BRIEF_REPEAT_PROMPT = "Can you repeat that?"
    STATUS_ACK_PREFIX = "Yes, I'm here."

    def __init__(
        self,
        ws_url: str,
        call_logger: CallLogger,
        *,
        call_id: str,
        from_number: str,
        to_number: str,
        debounce_ms: Optional[float] = None,
    ) -> None:
        self._ws_url = ws_url
        self._call_logger = call_logger
        self._call_id = call_id
        self._from_number = from_number
        self._to_number = to_number
        # Slightly longer debounce = fewer TTS utterances on streamed tokens (Sonic per call).
        self._debounce_ms = (
            debounce_ms
            if debounce_ms is not None
            else float(os.getenv("LINE_TTS_DEBOUNCE_SEC", "0.10"))
        )

        self._ws: Optional[WebSocketClientProtocol] = None
        self._recv_task: Optional[asyncio.Task[None]] = None
        self._closed = False

        self.tts_queue: asyncio.Queue[str] = asyncio.Queue()
        self.end_call_event = asyncio.Event()

        self._text_buffer: str = ""
        self._debounce_task: Optional[asyncio.Task[None]] = None
        self._pending_ask_fallback_task: Optional[asyncio.Task[None]] = None
        self._ask_fallback_delay_sec = max(
            0.25,
            float(os.getenv("LINE_TOOL_ASK_FALLBACK_SEC", "0.35")),
        )
        self._post_fallback_mute_model_sec = max(
            0.5,
            float(os.getenv("LINE_POST_FALLBACK_MUTE_SEC", "1.8")),
        )
        self._mute_model_text_until = 0.0
        self._last_tool_ask: str = ""
        self._awaiting_turn_response = False
        self._turn_watchdog_task: Optional[asyncio.Task[None]] = None
        self._turn_watchdog_sec = max(
            0.8,
            float(os.getenv("LINE_TURN_WATCHDOG_SEC", "1.4")),
        )
        # Hard guard for empty LLM turns (no tool_call + no message).
        self._no_response_watchdog_sec = max(
            1.2,
            float(os.getenv("LINE_NO_RESPONSE_WATCHDOG_SEC", "2.4")),
        )
        self._ask_dedup_window_sec = max(
            0.8,
            float(os.getenv("LINE_ASK_DEDUP_SEC", "3.0")),
        )
        self._fallback_spoken_text: str = ""
        self._no_response_watchdog_task: Optional[asyncio.Task[None]] = None
        self._last_spoken_ask_text: str = ""
        self._last_spoken_ask_ts = 0.0
        self._clarification_ask_key: str = ""
        self._clarification_count: int = 0
        self._ask_spoken_this_turn = False

    async def connect(self) -> None:
        """Connect to /ws and send the required ``start`` frame."""
        if self._closed:
            return
        self._ws = await websockets.connect(
            self._ws_url,
            ping_interval=20,
            ping_timeout=20,
            max_size=2**20,
        )
        start = {
            "type": "start",
            "call_id": self._call_id,
            "agent_call_id": self._call_id,
            "from": self._from_number,
            "to": self._to_number,
            "agent": {},
            "metadata": {"source": "twilio-path2-bridge"},
        }
        await self._ws.send(json.dumps(start))
        self._recv_task = asyncio.create_task(self._receive_loop(), name="line_session_recv")
        self._call_logger.info("Connected Line session client", event="line.session.connected")

    async def _flush_buffer_to_tts(self) -> None:
        t = self._text_buffer.strip()
        self._text_buffer = ""
        if not t:
            return
        t = self._normalize_agent_text_for_tts(t)
        if not t:
            return
        if self._fallback_spoken_text and self._is_fallback_duplicate(t):
            self._call_logger.debug(
                "Dropping model text that duplicates fallback",
                event="line.session.agent_text_dedup_fallback",
            )
            return
        self._fallback_spoken_text = ""
        self._call_logger.info(
            f"Queueing TTS text | chars={len(t)} text='{t[:160]}'",
            event="line.session.tts_enqueue",
        )
        await self.tts_queue.put(t)

    def _normalize_agent_text_for_tts(self, text: str) -> str:
        """
        Strong-pin policy: if ``_last_tool_ask`` is set and model text
        contains a *different* question, replace with authoritative ask.
        Also strips tool-policy scaffolding ("Still needed …").
        """
        t = (text or "").strip()
        if not t:
            return ""
        current_ask = (self._last_tool_ask or "").strip()
        low = t.lower()

        if "still needed" in low and len(t.split()) > 8:
            return current_ask or ""
        m = re.search(r'Ask next:\s*"([^"]+)"', t, flags=re.IGNORECASE)
        if m:
            return current_ask or (m.group(1) or "").strip()

        # Guard against off-stage narration like "I have your income as $325"
        # while current ask is still property address.
        if (
            current_ask
            and "property address" in current_ask.lower()
            and any(tok in low for tok in ("income", "annual", "$"))
        ):
            return current_ask

        # Model sometimes echoes SYSTEM_PROMPT examples (e.g. "full last name")
        # while Still needed is still property / income — never speak that aloud.
        if current_ask and self._model_leaks_name_intake_before_that_stage(t, current_ask):
            return current_ask

        if current_ask and self._model_leaks_offstage_for_personal_ask(t, current_ask):
            return current_ask

        if current_ask and self._text_contains_mismatched_question(t, current_ask):
            return current_ask
        return t

    _NAME_FIELD_LEAK_RE = re.compile(
        r"\b(full\s+)?last\s+name\b|\bfirst\s+name\b|\blegal\s+name\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _is_name_or_contact_intake_ask(current_ask: str) -> bool:
        ca = (current_ask or "").lower()
        return any(
            m in ca
            for m in (
                "first name",
                "last name",
                "date of birth",
                "mobile phone",
                "phone number",
                "email address",
            )
        )

    def _model_leaks_name_intake_before_that_stage(self, text: str, current_ask: str) -> bool:
        if self._is_name_or_contact_intake_ask(current_ask):
            return False
        return bool(self._NAME_FIELD_LEAK_RE.search(text))

    _PERSONAL_FIELD_ASK_MARKERS = (
        "phone",
        "mobile",
        "email",
        "first name",
        "last name",
        "date of birth",
        "birth",
    )
    # Asks with fixed options: broad "short reply" fallback must not apply (STT noise).
    _CHOICE_ASK_MARKERS = (
        "sole or joint",
        "primary, secondary, or investment",
        "for sale",
        "primary income",
        "income source",
    )

    _OFF_STAGE_TOPIC_PATTERNS = (
        r"\bincome\b",
        r"\bannual\b",
        r"\bproperty\b",
        r"\baddress\b",
        r"\bstreet\b",
        r"\bavenue\b",
        r"\bboulevard\b",
        r"\bmortgage\b",
        r"\bownership\b",
        r"\boccupancy\b",
        r"\binvestment property\b",
        r"\bprimary property\b",
        r"\bsecondary property\b",
        r"\bequity\b",
        r"\bfor sale\b",
        r"\blisted\b",
        r"\$\s*\d",
    )

    def _is_personal_field_ask(self, current_ask: str) -> bool:
        ca = (current_ask or "").lower()
        return any(m in ca for m in self._PERSONAL_FIELD_ASK_MARKERS)

    def _model_leaks_offstage_for_personal_ask(self, text: str, current_ask: str) -> bool:
        if not self._is_personal_field_ask(current_ask):
            return False
        low = (text or "").lower()
        ca = (current_ask or "").lower()
        patterns = list(self._OFF_STAGE_TOPIC_PATTERNS)
        if "phone" in ca or "mobile" in ca:
            patterns.extend((r"\bemail\b", r"@"))
        if "email" in ca:
            patterns.extend((r"\bphone\b", r"\bmobile\b"))
        if "first name" in ca:
            patterns.append(r"\blast name\b")
        if "last name" in ca:
            patterns.append(r"\bfirst name\b")
        if "date of birth" in ca or (
            "birth" in ca and "phone" not in ca and "email" not in ca
        ):
            patterns.extend((r"\bphone\b", r"\bmobile\b", r"\bemail\b", r"@"))
        return any(re.search(p, low) for p in patterns)

    @staticmethod
    def _text_contains_mismatched_question(text: str, current_ask: str) -> bool:
        if "?" not in text:
            return False
        ca = current_ask.strip().rstrip("?").lower()
        sentences = re.split(r"(?<=[.?!])\s+", text)
        for s in sentences:
            s_stripped = s.strip()
            if "?" not in s_stripped:
                continue
            q_body = s_stripped.rstrip("?").strip().lower()
            if ca and ca in q_body:
                return False
            return True
        return False

    async def _debounced_flush(self) -> None:
        try:
            await asyncio.sleep(self._debounce_ms)
            await self._flush_buffer_to_tts()
        except asyncio.CancelledError:
            raise

    def _schedule_debounced_flush(self) -> None:
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        self._debounce_task = asyncio.create_task(self._debounced_flush(), name="line_tts_debounce")

    async def clear_pending_speech(self, *, reason: str) -> None:
        """Drop buffered/queued agent speech after caller takes the floor."""
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
            try:
                await self._debounce_task
            except asyncio.CancelledError:
                pass
            self._debounce_task = None
        if (
            self._pending_ask_fallback_task
            and not self._pending_ask_fallback_task.done()
        ):
            self._pending_ask_fallback_task.cancel()
            try:
                await self._pending_ask_fallback_task
            except asyncio.CancelledError:
                pass
            self._pending_ask_fallback_task = None
        self._text_buffer = ""
        self._fallback_spoken_text = ""
        drained = 0
        while True:
            try:
                self.tts_queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break
        if drained:
            self._call_logger.info(
                f"Cleared pending TTS queue | reason={reason} drained={drained}",
                event="line.session.tts_queue_cleared",
            )

    async def _flush_now(self) -> None:
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
            try:
                await self._debounce_task
            except asyncio.CancelledError:
                pass
            self._debounce_task = None
        await self._flush_buffer_to_tts()

    def _is_fallback_duplicate(self, model_text: str) -> bool:
        """True if model_text is a near-duplicate of the fallback that already played."""
        fb = (self._fallback_spoken_text or "").strip().lower()
        mt = model_text.strip().lower()
        if not fb or not mt:
            return False
        if fb in mt or mt in fb:
            return True
        fb_words = set(fb.split())
        mt_words = set(mt.split())
        if not fb_words:
            return False
        return len(fb_words & mt_words) / len(fb_words) >= 0.7

    async def _enqueue_ask_if_needed(self, ask: str, *, event: str) -> bool:
        q = (ask or "").strip()
        if not q:
            return False
        if self._ask_spoken_this_turn:
            self._call_logger.debug(
                f"Turn-level ask already spoken; suppressing ({event}): {q}",
                event="line.session.ask_turn_deduped",
            )
            return False
        now = time.monotonic()
        if (
            self._last_spoken_ask_text
            and self._last_spoken_ask_text.strip().lower() == q.lower()
            and (now - self._last_spoken_ask_ts) < self._ask_dedup_window_sec
        ):
            self._call_logger.debug(
                f"Dedup ask suppressed ({event}): {q}",
                event="line.session.ask_deduped",
            )
            return False
        await self.tts_queue.put(q)
        self._last_spoken_ask_text = q
        self._last_spoken_ask_ts = now
        self._ask_spoken_this_turn = True
        self._awaiting_turn_response = False
        self._mute_model_text_until = now + self._post_fallback_mute_model_sec
        return True

    async def _receive_loop(self) -> None:
        assert self._ws is not None
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

                out_type = msg.get("type", "")
                if out_type == "message":
                    content = msg.get("content") or ""
                    if content:
                        self._awaiting_turn_response = False
                        if (
                            self._no_response_watchdog_task
                            and not self._no_response_watchdog_task.done()
                        ):
                            self._no_response_watchdog_task.cancel()
                        if self._turn_watchdog_task and not self._turn_watchdog_task.done():
                            self._turn_watchdog_task.cancel()
                        if time.monotonic() < self._mute_model_text_until:
                            self._call_logger.debug(
                                "Dropping model text fragment after fallback",
                                event="line.session.agent_text_deduped",
                            )
                            continue
                        # Model produced speech; cancel any pending Ask-next fallback.
                        if (
                            self._pending_ask_fallback_task
                            and not self._pending_ask_fallback_task.done()
                        ):
                            self._pending_ask_fallback_task.cancel()
                        self._text_buffer += content
                        self._schedule_debounced_flush()
                        self._call_logger.debug(
                            (
                                f"Line agent text fragment len={len(content)} "
                                f"buffer_len={len(self._text_buffer)}"
                            ),
                            event="line.session.agent_text",
                        )
                elif out_type == "tool_call":
                    await self._flush_now()
                    self._awaiting_turn_response = False
                    if (
                        self._no_response_watchdog_task
                        and not self._no_response_watchdog_task.done()
                    ):
                        self._no_response_watchdog_task.cancel()
                    if self._turn_watchdog_task and not self._turn_watchdog_task.done():
                        self._turn_watchdog_task.cancel()
                    name = msg.get("name", "")
                    result = msg.get("result")
                    has_result = result is not None
                    self._call_logger.info(
                        f"Line tool: {name} result={has_result}",
                        event="line.session.tool_call",
                    )
                    if (
                        name == "update_application_state"
                        and isinstance(result, str)
                    ):
                        m = re.search(r'Ask next:\s*"([^"]+)"', result)
                        if m:
                            ask = (m.group(1) or "").strip()
                            if ask:
                                self._last_tool_ask = ask
                                if (
                                    self._pending_ask_fallback_task
                                    and not self._pending_ask_fallback_task.done()
                                ):
                                    self._pending_ask_fallback_task.cancel()
                                self._pending_ask_fallback_task = asyncio.create_task(
                                    self._enqueue_ask_fallback(ask),
                                    name="line_tool_ask_fallback",
                                )
                elif out_type == "end_call":
                    await self._flush_now()
                    self.end_call_event.set()
                    self._call_logger.info("Line end_call", event="line.session.end_call")
                    break
                elif out_type == "error":
                    await self._flush_now()
                    self._call_logger.warning(
                        f"Line session error: {msg.get('content')}",
                        event="line.session.error",
                    )
                elif out_type == "config":
                    self._call_logger.debug("Line config output", event="line.session.config")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not self._closed:
                self._call_logger.warning(
                    f"Line session receive loop ended: {e}",
                    event="line.session.receive_ended",
                )

    async def _enqueue_ask_fallback(self, ask: str) -> None:
        try:
            await asyncio.sleep(self._ask_fallback_delay_sec)
            # Only speak fallback if model has not started speaking.
            if self._text_buffer.strip():
                return
            emitted = await self._enqueue_ask_if_needed(
                ask,
                event="line.session.ask_fallback",
            )
            if not emitted:
                return
            self._fallback_spoken_text = ask
            if self._turn_watchdog_task and not self._turn_watchdog_task.done():
                self._turn_watchdog_task.cancel()
                try:
                    await self._turn_watchdog_task
                except asyncio.CancelledError:
                    pass
                self._turn_watchdog_task = None
            self._call_logger.info(
                f"Line fallback ask emitted: {ask}",
                event="line.session.ask_fallback",
            )
            await broadcast_stt(
                kind="agent_fallback",
                text=ask,
                call_sid=self._call_logger.call_sid,
            )
        except asyncio.CancelledError:
            raise

    @staticmethod
    def _looks_like_answer_to_ask(ask: str, user_text: str) -> bool:
        a = (ask or "").strip().lower()
        u = (user_text or "").strip().lower()
        if not a or not u:
            return False

        # Generic binary confirmations used across many pages.
        if u in {"yes", "yes.", "no", "no.", "yep", "nope", "correct", "right"}:
            return True

        # Existing key asks.
        if "primary, secondary, or investment" in a:
            pu = u.strip().rstrip(".,!?").lower()
            return any(k in u for k in ("primary", "secondary", "investment")) or pu in {
                "1",
                "2",
                "3",
                "one",
                "two",
                "three",
                "won",
                "when",
                "too",
                "to",
                "tree",
                "free",
            }
        if "sole or joint" in a:
            pu = u.strip().rstrip(".,!?").lower()
            return (
                any(k in u for k in ("sole", "soul", "joint", "one", "two"))
                or u in {"1", "2"}
                or pu in {"when", "won", "one", "two", "to", "too"}
            )
        if "for sale" in a and "?" in a:
            pu = u.strip().rstrip(".,!?").lower()
            return any(k in u for k in ("yes", "no", "yep", "nope")) or pu in {
                "1",
                "2",
                "one",
                "two",
                "won",
                "when",
                "too",
                "to",
            }

        # Later-page asks.
        if "annual pre-tax income" in a or "annual income" in a:
            if any(ch.isdigit() for ch in u):
                return True
            if any(k in u for k in ("million", "thousand", "k", "m")):
                return True
            # Spoken-digit / number-word replies: "one zero zero zero...", "one fifty"
            if any(
                k in u
                for k in (
                    "zero",
                    "one",
                    "two",
                    "three",
                    "four",
                    "five",
                    "six",
                    "seven",
                    "eight",
                    "nine",
                    "ten",
                    "eleven",
                    "twelve",
                    "thirteen",
                    "fourteen",
                    "fifteen",
                    "sixteen",
                    "seventeen",
                    "eighteen",
                    "nineteen",
                    "twenty",
                    "thirty",
                    "forty",
                    "fifty",
                    "sixty",
                    "seventy",
                    "eighty",
                    "ninety",
                    "hundred",
                    "point",
                )
            ):
                return True
            return False
        if "other income" in a:
            return any(ch.isdigit() for ch in u) or u in {"none", "no", "nope"}
        if "primary income" in a or (
            "income source" in a and "other income" not in a and "annual" not in a
        ):
            pu = u.strip().rstrip(".,!?").lower()
            if pu in {
                "1",
                "2",
                "3",
                "4",
                "5",
                "one",
                "two",
                "three",
                "four",
                "five",
                "won",
                "when",
                "too",
                "to",
                "tree",
                "free",
                "for",
            }:
                return True
            return any(
                k in u
                for k in (
                    "full",
                    "part",
                    "self",
                    "retired",
                    "unemploy",
                    "employ",
                    "alimony",
                    "child support",
                    "public assistance",
                    "separate maintenance",
                    "furlough",
                )
            )
        if "date of birth" in a:
            return any(ch.isdigit() for ch in u)
        if "phone" in a:
            digit_count = sum(ch.isdigit() for ch in u)
            return digit_count >= 7
        if "email" in a:
            return "@" in u or " dot " in u
        if "first name" in a or "last name" in a or "name" in a:
            return len(u.split()) <= 3
        if (
            "agree" in a
            or "consent" in a
            or "privacy policy" in a
            or "terms of service" in a
            or "credit pull" in a
            or "soft credit" in a
        ):
            return any(k in u for k in ("yes", "i agree", "agree", "no", "nope"))

        # Broad fallback: short reply to an unknown question is often an answer — but
        # never for enumerated choice asks: STT noise ("sad", "um") must not skip
        # watchdog / look like a valid sole-vs-joint reply (see path2 logs).
        if "?" in a and len(u.split()) <= 4:
            if any(m in a for m in LineSessionClient._CHOICE_ASK_MARKERS):
                return False
            return True
        return False

    async def _turn_response_watchdog(self) -> None:
        try:
            await asyncio.sleep(self._turn_watchdog_sec)
            if not self._awaiting_turn_response:
                return
            ask = (self._last_tool_ask or "").strip()
            if not ask:
                return
            to_speak = self.BRIEF_REPEAT_PROMPT
            emitted = await self._enqueue_ask_if_needed(
                to_speak,
                event="line.session.turn_watchdog",
            )
            if not emitted:
                return
            self._call_logger.info(
                f"Line watchdog brief repeat (pending ask was): {ask[:80]}",
                event="line.session.turn_watchdog",
            )
        except asyncio.CancelledError:
            raise

    async def _no_response_watchdog(self) -> None:
        """
        Rescue path when LLM returns no message and no tool call.
        This happens sporadically on noisy turns; speak a short prompt so call doesn't stall.
        """
        try:
            await asyncio.sleep(self._no_response_watchdog_sec)
            pending = (self._last_tool_ask or "").strip()
            to_speak = self.BRIEF_REPEAT_PROMPT
            emitted = await self._enqueue_ask_if_needed(
                to_speak,
                event="line.session.no_response_watchdog",
            )
            if not emitted:
                return
            self._call_logger.warning(
                f"Line no-response watchdog brief repeat (pending ask was): {pending[:80]}",
                event="line.session.no_response_watchdog",
            )
            await broadcast_stt(
                kind="agent_watchdog_no_response",
                text=to_speak,
                call_sid=self._call_logger.call_sid,
            )
        except asyncio.CancelledError:
            raise

    async def send_user_speaking_start(self) -> None:
        """Barge-in: cancel agent generation (ConversationRunner cancel_filter)."""
        if not self._ws or self._closed:
            return
        await self._ws.send(json.dumps({"type": "user_state", "value": "speaking"}))

    def inject_ask_next_hint(self, ask: str, *, schedule_fallback: bool = True) -> None:
        """Seed fallback ask from non-Line tool paths (pre-Line heuristics)."""
        q = (ask or "").strip()
        if not q:
            return
        self._last_tool_ask = q
        if not schedule_fallback:
            return
        if (
            self._pending_ask_fallback_task
            and not self._pending_ask_fallback_task.done()
        ):
            self._pending_ask_fallback_task.cancel()
        self._pending_ask_fallback_task = asyncio.create_task(
            self._enqueue_ask_fallback(q),
            name="line_tool_ask_fallback_injected",
        )

    async def mark_deterministic_turn(self, ask_next: Optional[str]) -> None:
        """Reset turn state and speak ask_next without involving LLM."""
        await self.clear_pending_speech(reason="deterministic_turn")
        self._fallback_spoken_text = ""
        self._ask_spoken_this_turn = False
        self._last_tool_ask = ""

        for task_attr in (
            "_no_response_watchdog_task",
            "_turn_watchdog_task",
            "_pending_ask_fallback_task",
        ):
            t = getattr(self, task_attr, None)
            if t and not t.done():
                t.cancel()
            setattr(self, task_attr, None)

        self._awaiting_turn_response = False

        if ask_next:
            self._last_tool_ask = ask_next
            await self._enqueue_ask_if_needed(
                ask_next, event="line.session.deterministic_ask"
            )

    async def enqueue_status_check_response(self) -> bool:
        """Presence check response: acknowledge audibly, then restate current ask."""
        ask = (self._last_tool_ask or "").strip()
        response = f"{self.STATUS_ACK_PREFIX} {ask}" if ask else self.STATUS_ACK_PREFIX
        self._ask_spoken_this_turn = False
        return await self._enqueue_ask_if_needed(
            response,
            event="line.session.status_check_response",
        )

    @staticmethod
    def _clarification_for_ask(ask: str) -> str:
        low = (ask or "").strip().lower()
        if "property address" in low or "street address" in low:
            return "I didn't catch that correctly. Please repeat the street address."
        if "ownership type" in low or "sole or joint" in low:
            return "I didn't catch ownership type. You can say sole or joint."
        if "property type" in low or "primary, secondary, or investment" in low:
            return (
                "I didn't catch property type. "
                "You can say primary, secondary, or investment."
            )
        if "currently for sale" in low or "for sale" in low:
            return "I didn't catch that. Is it currently for sale?"
        if "annual pre-tax income" in low or "annual income" in low:
            return "I didn't catch annual income. Please say the dollar amount."
        if "other income" in low:
            return "I didn't catch other income. You can say an amount or none."
        if "primary income source" in low or "income source" in low:
            return "I didn't catch income source. Please say it another way."
        if "first name" in low:
            return "I didn't catch first name. Please say first name only."
        if "last name" in low:
            return "I didn't catch last name. Please say last name only."
        if "date of birth" in low:
            return "I didn't catch date of birth. Please say month, day, and year."
        if "phone" in low:
            return "I didn't catch phone number. Please say all ten digits."
        if "email" in low:
            return "I didn't catch email. Please say it slowly."
        return "I didn't catch an answer. Please say it another way."

    @staticmethod
    def _spelling_clarification_for_ask(ask: str) -> str:
        low = (ask or "").strip().lower()
        if "first name" in low:
            return (
                "I still didn't catch first name. Please spell it slowly, "
                "letter by letter."
            )
        if "last name" in low:
            return (
                "I still didn't catch last name. Please spell it slowly, "
                "letter by letter."
            )
        if "property address" in low or "street address" in low:
            return "Please spell the street address slowly, one word at a time."
        if "email" in low:
            return "Please spell the email slowly, letter by letter."
        if "date of birth" in low:
            return "Please say date of birth slowly, one word at a time."
        if "annual pre-tax income" in low or "annual income" in low:
            return "Please say the income slowly, one word at a time."
        return "Please say it slowly again, or spell it if needed."

    async def enqueue_clarification_for_current_ask(self) -> bool:
        """Clarify invalid/noisy answer without replacing current Ask next."""
        ask = (self._last_tool_ask or "").strip()
        key = ask.lower()
        if key and key == getattr(self, "_clarification_ask_key", ""):
            self._clarification_count = getattr(self, "_clarification_count", 0) + 1
        else:
            self._clarification_ask_key = key
            self._clarification_count = 1
        if self._clarification_count >= 3:
            response = self._spelling_clarification_for_ask(ask)
        elif self._clarification_count == 1:
            response = ask if ask else self._clarification_for_ask(ask)
        else:
            response = self._clarification_for_ask(ask)
        self._ask_spoken_this_turn = False
        return await self._enqueue_ask_if_needed(
            response,
            event="line.session.non_answer_clarification",
        )

    async def send_user_transcript(self, transcript: str) -> None:
        """
        Send one user turn: user_state speaking → message → user_state idle.
        """
        if not self._ws or self._closed:
            return
        text = transcript.strip()
        if not text:
            return
        self._fallback_spoken_text = ""
        self._ask_spoken_this_turn = False
        lower_text = text.lower()
        ask_l = (self._last_tool_ask or "").strip()
        # Status pings ("you there?", "what's next?") should repeat current ask
        # locally and never reopen LLM context for stale/out-of-stage questions.
        if ask_l and self._is_status_check_utterance(lower_text):
            await self.enqueue_status_check_response()
            return
        frames = [
            {"type": "user_state", "value": "speaking"},
            {"type": "message", "content": text},
            {"type": "user_state", "value": "idle"},
        ]
        # Do not watchdog-reask when user likely answered the currently pending ask.
        # This prevents stale repeats like "Primary, secondary, or investment?"
        # after occupancy was already captured by pre-Line heuristics.
        skip_watchdog = False
        if ask_l and self._looks_like_answer_to_ask(ask_l, lower_text):
            skip_watchdog = True
            self._call_logger.debug(
                "Likely answer to current ask; disabling short watchdog for this turn",
                event="line.session.turn_skip_short_watchdog",
            )

        self._awaiting_turn_response = not skip_watchdog
        if (
            self._no_response_watchdog_task
            and not self._no_response_watchdog_task.done()
        ):
            self._no_response_watchdog_task.cancel()
        # Always keep long watchdog to prevent dead-air turns, even when short
        # watchdog is skipped for likely direct answers.
        self._no_response_watchdog_task = asyncio.create_task(
            self._no_response_watchdog(),
            name="line_no_response_watchdog",
        )
        if self._turn_watchdog_task and not self._turn_watchdog_task.done():
            self._turn_watchdog_task.cancel()
        if self._awaiting_turn_response:
            self._turn_watchdog_task = asyncio.create_task(
                self._turn_response_watchdog(),
                name="line_turn_watchdog",
            )
        else:
            self._turn_watchdog_task = None
        for frame in frames:
            await self._ws.send(json.dumps(frame))

    @staticmethod
    def _is_status_check_utterance(text: str) -> bool:
        from agent.state_manager import transcript_is_status_check
        return transcript_is_status_check(text)

    async def close(self) -> None:
        self._closed = True
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
            try:
                await self._debounce_task
            except asyncio.CancelledError:
                pass
            self._debounce_task = None
        if (
            self._pending_ask_fallback_task
            and not self._pending_ask_fallback_task.done()
        ):
            self._pending_ask_fallback_task.cancel()
            try:
                await self._pending_ask_fallback_task
            except asyncio.CancelledError:
                pass
            self._pending_ask_fallback_task = None
        if self._turn_watchdog_task and not self._turn_watchdog_task.done():
            self._turn_watchdog_task.cancel()
            try:
                await self._turn_watchdog_task
            except asyncio.CancelledError:
                pass
            self._turn_watchdog_task = None
        if (
            self._no_response_watchdog_task
            and not self._no_response_watchdog_task.done()
        ):
            self._no_response_watchdog_task.cancel()
            try:
                await self._no_response_watchdog_task
            except asyncio.CancelledError:
                pass
            self._no_response_watchdog_task = None

        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._call_logger.info("Closed Line session client", event="line.session.closed")
