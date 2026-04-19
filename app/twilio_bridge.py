"""
Twilio ↔ Cartesia Agent Audio Bridge
======================================

Bridges bidirectional audio between a Twilio phone call and Cartesia's
cloud agent via the Calls API (wss://api.cartesia.ai/agents/stream/).

Protocol:
    Twilio sends µ-law 8kHz audio. Cartesia Calls API accepts mulaw_8000
    natively via the input_format config. No transcoding needed — we pass
    Twilio's base64 payload straight through as media_input events.

    Cartesia returns media_output events with base64 audio that we forward
    to Twilio as media events.

Reference: https://docs.cartesia.ai/line/integrations/calls-api

Debug: set CARTESIA_LOG_ALL_EVENTS=1 to log Cartesia JSON frames (payloads
redacted) as cartesia.debug_frame. TTS streams ~50 media_output/sec — those are
skipped by default. Set CARTESIA_LOG_MEDIA_CHUNKS=1 to log every media_output too.
"""

import asyncio
import json
import os
from typing import Any, Optional

from app.logging_config import CallLogger

import websockets
from fastapi import WebSocket, WebSocketDisconnect

# Cloud-compatible HELOC prompt. Unlike agent/main_agent.py's SYSTEM_PROMPT,
# this version works with the cloud agent's available tools (web_search,
# end_call) and drives the intake flow conversationally without requiring
# the local update_application_state tool.
CLOUD_SYSTEM_PROMPT = """\
You are a mortgage intake specialist for West Capital Lending. Your job is \
to collect all the information needed to start a HELOC application over \
the phone.

# Voice and Tone
- Speak like a friendly, professional loan officer — warm but efficient.
- Use contractions and casual phrasing. Sound human, not robotic.
- Keep responses under 2 sentences. This is a phone call, not an email.
- Always end with a question to keep the conversation moving (except goodbye).
- Speak naturally — no bullet points, numbered lists, or special characters.
- Spell out numbers and dates verbally: "January fifteenth" not "1/15".

# Conversation Flow — Collect in This Order
Track what the caller has already provided. Do NOT re-ask for information \
already given. Ask for ONE or TWO related fields at a time.

1. Property address: street, city, state, zip code. Mention apartment or \
   suite number is optional.
2. Ownership type: sole or joint. Occupancy: primary residence, secondary, \
   or investment property.
3. Is the property currently listed for sale?
4. Annual pre-tax income and primary income source. Ask if they have other \
   income to include (optional).
5. Legal name: first, middle (optional), last, suffix (optional).
6. Date of birth.
7. Mobile phone number and email address.
8. Verbal consent to Privacy Policy and Terms of Service. The caller must \
   clearly say "I agree" — vague responses like "sure" are not sufficient.
9. Verbal consent to Electronic Communications Policy and a soft credit \
   pull. CRITICAL: say "soft credit pull" and explain it will NOT affect \
   their credit score. Again, require a clear "I agree."
10. Once all information is collected, read back a brief summary and let \
    them know a secure password has been generated for their new account.

# Normalization
- Expand abbreviations: "Ave" to "Avenue", "Blvd" to "Boulevard", etc.
- State as 2-letter code: CA, NY, IL.
- Phone as 10 digits, no dashes.

# Rules
- If asked about rates or terms: "Our loan officers will discuss specific \
  rates during your follow-up call."
- If the caller goes off-topic, gently redirect to the application.
- NEVER provide interest rates, APR, or financial advice.
- When the caller says goodbye, respond warmly and then call end_call.
- NEVER call end_call without saying goodbye first.
"""

CLOUD_INTRODUCTION = (
    "Hi, I'm your West Capital assistant. "
    "I can help you start your HELOC application today. "
    "To get started, what is the address of the property "
    "you are looking to finance?"
)

_DEBUG_LOG_MAX_CHARS = 32000


def _redact_cartesia_value(obj: Any, parent_key: Optional[str] = None) -> Any:
    """
    Redact large/base64 blobs for safe logging. Replaces `payload` strings
    and any string longer than 500 chars with a length marker.
    """
    if isinstance(obj, dict):
        return {k: _redact_cartesia_value(v, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_cartesia_value(v, parent_key) for v in obj]
    if isinstance(obj, str):
        if parent_key == "payload" or len(obj) > 500:
            return f"<redacted len={len(obj)}>"
        return obj
    return obj


def _log_cartesia_frame_if_enabled(call_logger: CallLogger, data: dict) -> None:
    """
    When CARTESIA_LOG_ALL_EVENTS is 1/true/yes, log full JSON (redacted)
    for Cartesia text frames. Skips ``media_output`` by default (~50/sec TTS).
    Set CARTESIA_LOG_MEDIA_CHUNKS=1 to include every audio chunk.
    """
    flag = os.getenv("CARTESIA_LOG_ALL_EVENTS", "").strip().lower()
    if flag not in ("1", "true", "yes"):
        return
    evt = data.get("event", "")
    if evt == "media_output":
        media_flag = os.getenv("CARTESIA_LOG_MEDIA_CHUNKS", "").strip().lower()
        if media_flag not in ("1", "true", "yes"):
            return
    redacted = _redact_cartesia_value(data)
    line = json.dumps(redacted, default=str, ensure_ascii=False)
    if len(line) > _DEBUG_LOG_MAX_CHARS:
        line = line[:_DEBUG_LOG_MAX_CHARS] + "...[truncated]"
    call_logger.info(line, event="cartesia.debug_frame")


class TwilioBridge:
    """
    Bridges a single phone call between Twilio Media Streams and the
    Cartesia Calls API. Zero transcoding — mulaw passthrough.

    Lifecycle:
        1. __init__()     — Store references, initialize state
        2. run()          — Connect to Cartesia, start bidirectional streams
        3. _forward_*()   — Two concurrent async tasks handle each direction
        4. _cleanup()     — Close all connections on call end
    """

    def __init__(
        self,
        twilio_ws: WebSocket,
        cartesia_ws_url: str,
        call_logger: CallLogger,
    ):
        self.twilio_ws = twilio_ws
        self.cartesia_ws_url = cartesia_ws_url
        self.call_logger = call_logger

        # Twilio stream metadata (populated from Twilio's 'start' event)
        self.stream_sid: Optional[str] = None
        self.call_sid: Optional[str] = None

        # Cartesia stream_id (populated from Cartesia's 'ack' event)
        self._cartesia_stream_id: Optional[str] = None

        # Cartesia WebSocket client connection
        self._cartesia_ws: Optional[websockets.WebSocketClientProtocol] = None

        self._running = False
        self._first_audio_sent = False

    async def run(self) -> None:
        """
        Connect to Cartesia Calls API and run bidirectional audio streaming
        until the call ends or an error occurs.
        """
        self._running = True

        try:
            cartesia_api_key = os.getenv("CARTESIA_API_KEY", "")

            self._cartesia_ws = await websockets.connect(
                self.cartesia_ws_url,
                extra_headers={
                    "Authorization": f"Bearer {cartesia_api_key}",
                    "Cartesia-Version": "2025-04-16",
                },
                ping_interval=20,
                ping_timeout=10,
                max_size=2**20,
            )

            self.call_logger.info(
                "Connected to Cartesia agent stream",
                event="cartesia.connected",
            )

            twilio_to_cartesia = asyncio.create_task(
                self._forward_twilio_to_cartesia(),
                name=f"t2c:{self.call_sid or 'init'}",
            )
            cartesia_to_twilio = asyncio.create_task(
                self._forward_cartesia_to_twilio(),
                name=f"c2t:{self.call_sid or 'init'}",
            )

            done, pending = await asyncio.wait(
                [twilio_to_cartesia, cartesia_to_twilio],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            for task in done:
                exc = task.exception()
                if exc and not isinstance(
                    exc, (WebSocketDisconnect, asyncio.CancelledError)
                ):
                    raise exc

        except WebSocketDisconnect:
            self.call_logger.info(
                "Twilio disconnected (caller hung up)",
                event="call.dropped",
            )
        except websockets.ConnectionClosed as e:
            self.call_logger.info(
                f"Cartesia connection closed: code={e.code} reason={e.reason}",
                event="agent.ended",
            )
        except ConnectionRefusedError:
            self.call_logger.error(
                "Cannot connect to Cartesia agent — is it running?",
                event="cartesia.refused",
            )
        except Exception as e:
            self.call_logger.error(
                f"Unhandled bridge error: {type(e).__name__}: {e}",
                event="bridge.error",
                exc_info=True,
            )
            raise
        finally:
            self._running = False
            await self._cleanup()

    # =========================================================================
    # Direction 1: Twilio → Cartesia (Caller's voice → Agent's ears)
    # =========================================================================

    async def _forward_twilio_to_cartesia(self) -> None:
        """
        Receive audio from Twilio's Media Stream and forward to Cartesia.

        Twilio sends µ-law 8kHz base64 audio in 'media' events. Cartesia
        accepts mulaw_8000 natively, so we pass the payload straight through.
        """
        while self._running:
            try:
                raw = await self.twilio_ws.receive_text()
                message = json.loads(raw)
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

                    await self._send_cartesia_start()

                elif event_type == "media":
                    payload = message["media"]["payload"]
                    await self._forward_audio_to_cartesia(payload)

                elif event_type == "stop":
                    self.call_logger.info(
                        "Twilio media stream stopped",
                        event="twilio.stream.stop",
                    )
                    break

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError as e:
                self.call_logger.warning(
                    f"Malformed JSON from Twilio: {e}",
                    event="twilio.parse_error",
                )

    # =========================================================================
    # Direction 2: Cartesia → Twilio (Agent's voice → Caller's ears)
    # =========================================================================

    async def _forward_cartesia_to_twilio(self) -> None:
        """
        Receive events from Cartesia Calls API and forward audio to Twilio.

        Cartesia Calls API server events:
          - ack: confirms stream config, returns stream_id
          - media_output: base64 audio payload
          - clear: barge-in / interrupt current playback
          - transfer_call: agent wants to transfer
        """
        if not self._cartesia_ws:
            return

        try:
            async for message in self._cartesia_ws:
                if not self._running:
                    break

                try:
                    if isinstance(message, bytes):
                        self.call_logger.info(
                            "Received raw audio bytes from Cartesia",
                            size=len(message),
                        )
                        continue

                    data = json.loads(message)
                    _log_cartesia_frame_if_enabled(self.call_logger, data)
                    event = data.get("event", "")

                    if event == "ack":
                        self._cartesia_stream_id = data.get("stream_id")
                        self.call_logger.info(
                            f"Cartesia ack received | stream_id={self._cartesia_stream_id}",
                            event="cartesia.ack",
                        )

                    elif event == "media_output":
                        audio_payload = data.get("media", {}).get("payload")
                        if audio_payload:
                            await self._send_twilio_media(audio_payload)

                    elif event == "clear":
                        self.call_logger.info(
                            "Cartesia clear (barge-in)",
                            event="cartesia.clear",
                        )
                        if self.stream_sid:
                            try:
                                await self.twilio_ws.send_text(json.dumps({
                                    "event": "clear",
                                    "streamSid": self.stream_sid,
                                }))
                            except Exception:
                                pass

                    elif event == "transfer_call":
                        target = data.get("transfer", {}).get("target_phone_number", "")
                        self.call_logger.info(
                            f"Cartesia transfer request: {target}",
                            event="cartesia.transfer",
                        )

                    elif event == "error":
                        self.call_logger.error(
                            f"Cartesia error: {data.get('message', data)}",
                            event="cartesia.error",
                        )

                    else:
                        self.call_logger.info(
                            f"Cartesia event: {event}",
                            event="cartesia.event",
                        )

                except Exception as e:
                    self.call_logger.warning(
                        f"Error processing Cartesia message: {e}",
                        event="cartesia.process_error",
                    )

        except websockets.ConnectionClosed:
            self.call_logger.info(
                "Cartesia WebSocket closed",
                event="cartesia.ws.closed",
            )

    # =========================================================================
    # Audio Passthrough (Twilio mulaw → Cartesia mulaw — zero transcoding)
    # =========================================================================

    async def _forward_audio_to_cartesia(self, base64_mulaw: str) -> None:
        """
        Forward Twilio's base64 µ-law audio directly to Cartesia.

        Cartesia Calls API accepts mulaw_8000 natively (configured in the
        start event). No decoding, resampling, or re-encoding needed.
        """
        if not self._cartesia_ws:
            return

        msg = {
            "event": "media_input",
            "media": {
                "payload": base64_mulaw,
            },
        }

        if self._cartesia_stream_id:
            msg["stream_id"] = self._cartesia_stream_id

        try:
            await self._cartesia_ws.send(json.dumps(msg))
        except websockets.ConnectionClosed:
            self.call_logger.warning(
                "Cartesia WS closed during audio forward",
                event="cartesia.send_failed",
            )
            self._running = False

    # =========================================================================
    # Cartesia Session Init (Calls API start event)
    # =========================================================================

    async def _send_cartesia_start(self) -> None:
        """
        Send the Calls API 'start' event to initialize the audio stream.

        Per https://docs.cartesia.ai/line/integrations/calls-api:
        - input_format: "mulaw_8000" matches Twilio's native format
        - No output config needed (server decides)
        - stream_id is optional; server generates one in the ack
        """
        if not self._cartesia_ws:
            return

        start_message = {
            "event": "start",
            "config": {
                "input_format": "mulaw_8000",
            },
            "agent": {
                "system_prompt": CLOUD_SYSTEM_PROMPT,
                "introduction": CLOUD_INTRODUCTION,
            },
            "metadata": {
                "from": self.call_sid or "twilio-bridge",
                "stream_sid": self.stream_sid,
            },
        }

        self.call_logger.info(
            "Sending Cartesia start event (mulaw_8000)",
            event="cartesia.start.sent",
        )
        await self._cartesia_ws.send(json.dumps(start_message))

    # =========================================================================
    # Twilio Media Output
    # =========================================================================

    async def _send_twilio_media(self, base64_audio: str) -> None:
        """
        Send an audio chunk back to Twilio as a 'media' event.
        """
        if not self.stream_sid:
            return

        if not self._first_audio_sent:
            self._first_audio_sent = True
            self.call_logger.mark_first_audio()

        media_event = {
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {
                "payload": base64_audio,
            },
        }

        try:
            await self.twilio_ws.send_text(json.dumps(media_event))
        except (WebSocketDisconnect, RuntimeError):
            self.call_logger.info(
                "Cannot send audio — Twilio WS closed",
                event="twilio.send_closed",
            )
            self._running = False

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def _cleanup(self) -> None:
        """Close all connections and release resources."""
        if self._cartesia_ws:
            try:
                await self._cartesia_ws.close()
            except Exception:
                pass

        self.call_logger.info(
            f"Bridge cleanup complete | TTFA={self.call_logger.ttfa_ms}ms",
            event="bridge.cleanup",
        )
