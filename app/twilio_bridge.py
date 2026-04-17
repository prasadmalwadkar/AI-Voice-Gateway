"""
Twilio ↔ Cartesia Agent Audio Bridge
======================================

This module is the beating heart of the VoIP bridge. It handles the
bidirectional flow of raw audio between a Twilio phone call and
Cartesia's voice agent infrastructure.

Core Engineering Challenges:
    1. PROTOCOL TRANSLATION
       Twilio Media Streams use a JSON-framed WebSocket protocol with
       event types (connected, start, media, stop). Cartesia's agent
       streaming uses a different WebSocket protocol. This bridge
       translates between the two.

    2. AUDIO TRANSCODING
       Twilio delivers G.711 µ-law audio at 8kHz (the telephony standard
       since 1972). Cartesia's STT engine expects 16-bit linear PCM at
       16kHz. We transcode in real-time with zero buffering.

       Audio Pipeline (Caller → Agent):
         base64(µ-law 8kHz) → decode → ulaw2lin → PCM 16-bit 8kHz
         → ratecv resample → PCM 16-bit 16kHz → Cartesia STT

       Audio Pipeline (Agent → Caller):
         Cartesia TTS (configured for µ-law 8kHz) → base64 → Twilio

    3. ZERO-LATENCY DESIGN
       We process each audio chunk (~20ms) immediately upon receipt.
       No accumulation buffers. No batch processing. The async
       architecture ensures the transcode→forward path adds <1ms
       of processing latency per chunk.

    4. GRACEFUL DEGRADATION
       Phone calls drop. Networks fail. The bridge handles every
       disconnection scenario without crashing the server or leaking
       resources.

Audio Format Reference:
    Twilio → Server:
        Encoding:    G.711 µ-law (ITU-T standard)
        Sample Rate: 8,000 Hz
        Bit Depth:   8-bit (companded from 14-bit)
        Channels:    1 (mono)
        Chunk Size:  ~160 bytes per 20ms frame
        Transport:   Base64-encoded in JSON payload

    Server → Cartesia STT:
        Encoding:    Linear PCM (signed 16-bit little-endian)
        Sample Rate: 16,000 Hz
        Bit Depth:   16-bit
        Channels:    1 (mono)
        Chunk Size:  ~640 bytes per 20ms frame (after resample)
        Transport:   Raw binary over WebSocket

    Cartesia TTS → Server → Twilio:
        Configured to output µ-law 8kHz directly to avoid a second
        transcode step. Cartesia's TTS API supports this natively.
"""

import asyncio
import base64
import json
import logging
from typing import Optional

# audioop handles G.711 µ-law ↔ linear PCM conversion.
# It's a CPython built-in implemented in C — critical for real-time
# audio where Python-level decoding would be too slow.
# Deprecated in 3.11, removed in 3.13 — audioop-lts is the fallback.
try:
    import audioop
except ImportError:
    import audioop_lts as audioop  # type: ignore[no-redef]

import websockets
from fastapi import WebSocket, WebSocketDisconnect

from app.logging_config import CallLogger

logger = logging.getLogger("voip.bridge")


# =============================================================================
# Audio Constants
# =============================================================================
# These constants define the audio format contract between Twilio, this
# bridge, and Cartesia. Changing these requires updating the Cartesia
# session config in _build_cartesia_start_message() as well.

# Twilio's telephony audio: G.711 µ-law, 8kHz, mono
TWILIO_SAMPLE_RATE = 8000
TWILIO_SAMPLE_WIDTH = 1  # µ-law = 8-bit = 1 byte/sample

# PCM output from µ-law decoding: 16-bit signed, 8kHz initially
PCM_SAMPLE_WIDTH = 2  # 16-bit = 2 bytes/sample

# Cartesia STT input: 16-bit PCM at 16kHz (standard for speech recognition)
CARTESIA_INPUT_SAMPLE_RATE = 16000

# Number of audio channels (telephony is always mono)
CHANNELS = 1


class TwilioBridge:
    """
    Manages the full lifecycle of a single phone call, bridging audio
    between Twilio's Media Stream and Cartesia's Agent Streaming endpoint.

    Architecture:
    ┌──────────────┐          ┌──────────────┐          ┌────────────────┐
    │   Twilio     │   WS     │  TwilioBridge │    WS    │    Cartesia    │
    │  (Caller's   │ ←──────→ │  (This Class) │ ←──────→ │  (STT → LLM   │
    │   Phone)     │  JSON/   │               │  Binary/ │   → TTS)       │
    └──────────────┘  base64  └──────────────┘  JSON     └────────────────┘

    Lifecycle:
        1. __init__()     — Store references, initialize state
        2. run()          — Connect to Cartesia, start bidirectional streams
        3. _forward_*()   — Two concurrent async tasks handle each direction
        4. _cleanup()     — Close all connections on call end

    Thread Safety:
        This class is NOT thread-safe. Each instance handles exactly one
        call, and runs entirely within a single asyncio event loop.
        Concurrent calls each get their own TwilioBridge instance.
    """

    def __init__(
        self,
        twilio_ws: WebSocket,
        cartesia_ws_url: str,
        call_logger: CallLogger,
    ):
        """
        Initialize the bridge for a single call.

        Args:
            twilio_ws: The FastAPI WebSocket connection from Twilio.
            cartesia_ws_url: The Cartesia agent streaming WebSocket URL
                             (includes API key as query parameter).
            call_logger: Per-call structured logger with TTFA tracking.
        """
        self.twilio_ws = twilio_ws
        self.cartesia_ws_url = cartesia_ws_url
        self.call_logger = call_logger

        # -- Stream metadata (populated from Twilio's 'start' event) --
        self.stream_sid: Optional[str] = None
        self.call_sid: Optional[str] = None

        # -- Cartesia WebSocket client connection --
        self._cartesia_ws: Optional[
            websockets.WebSocketClientProtocol
        ] = None

        # -- Bridge state --
        self._running = False
        self._first_audio_sent = False

        # -- Resampling State --
        # audioop.ratecv() is STATEFUL: it tracks the fractional sample
        # position between calls. This is critical because audio chunks
        # don't necessarily align to exact sample boundaries.
        #
        # Without state preservation, you get audible clicks/pops at
        # every chunk boundary (every ~20ms = 50 times per second).
        # The state object carries the interpolation residual across calls.
        self._upsample_state = None

    async def run(self) -> None:
        """
        Main bridge loop. Connects to Cartesia and runs bidirectional
        audio streaming until the call ends or an error occurs.

        This method is the single entry point called by the WebSocket
        handler in app/main.py. It orchestrates the entire call lifecycle.

        Concurrency Architecture:
            Two independent asyncio tasks run concurrently:
            - twilio_to_cartesia: Reads from Twilio, transcodes, sends
                                  to Cartesia
            - cartesia_to_twilio: Reads from Cartesia, sends to Twilio

            Using FIRST_COMPLETED means if either direction dies (caller
            hangs up, Cartesia disconnects, network error), we immediately
            cancel the other direction and clean up. No zombie tasks.
        """
        self._running = True

        try:
            # -- Step 1: Connect to Cartesia's agent streaming endpoint --
            # We're a WebSocket CLIENT connecting to Cartesia's server.
            # The VoiceAgentApp (or Cartesia cloud) is the server.
            self._cartesia_ws = await websockets.connect(
                self.cartesia_ws_url,
                additional_headers={
                    "Cartesia-Version": "2024-11-13",
                },
                # Ping interval keeps the connection alive through
                # load balancers that kill idle WebSockets (AWS ALB: 60s)
                ping_interval=20,
                ping_timeout=10,
                # Max message size: TTS audio chunks can be large
                max_size=2**20,  # 1MB
            )
            self.call_logger.info(
                "Connected to Cartesia agent stream",
                event="cartesia.connected",
            )

            # -- Step 2: Launch bidirectional audio tasks --
            twilio_to_cartesia = asyncio.create_task(
                self._forward_twilio_to_cartesia(),
                name=f"t2c:{self.call_sid or 'init'}",
            )
            cartesia_to_twilio = asyncio.create_task(
                self._forward_cartesia_to_twilio(),
                name=f"c2t:{self.call_sid or 'init'}",
            )

            # -- Step 3: Wait for either direction to complete --
            # FIRST_COMPLETED is intentional: if the caller hangs up,
            # we don't want to keep the Cartesia connection alive
            # (and vice versa). This pattern ensures fast cleanup.
            done, pending = await asyncio.wait(
                [twilio_to_cartesia, cartesia_to_twilio],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # -- Step 4: Cancel the surviving task --
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass  # Expected when we cancel the task

            # -- Step 5: Propagate any exceptions from completed tasks --
            for task in done:
                exc = task.exception()
                if exc and not isinstance(
                    exc, (WebSocketDisconnect, asyncio.CancelledError)
                ):
                    raise exc

        except WebSocketDisconnect:
            # Normal: Twilio disconnected (caller hung up)
            self.call_logger.info(
                "Twilio disconnected (caller hung up)",
                event="call.dropped",
            )
        except websockets.ConnectionClosed as e:
            # Normal: Cartesia closed the connection (agent ended call)
            self.call_logger.info(
                f"Cartesia connection closed: code={e.code} "
                f"reason={e.reason}",
                event="agent.ended",
            )
        except ConnectionRefusedError:
            # Cartesia agent is not running or unreachable
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

        Twilio Media Stream Protocol (JSON over WebSocket):
        ┌─────────────┬────────────────────────────────────┐
        │ Event       │ Purpose                            │
        ├─────────────┼────────────────────────────────────┤
        │ 'connected' │ WebSocket handshake complete        │
        │ 'start'     │ Audio stream started, has metadata  │
        │ 'media'     │ Audio payload (base64 µ-law)        │
        │ 'stop'      │ Audio stream ended                  │
        └─────────────┴────────────────────────────────────┘

        The 'media' events arrive every ~20ms, each containing 160 bytes
        of µ-law audio (8000 samples/sec ÷ 50 chunks/sec = 160 samples).
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
                    # Extract stream metadata — we need streamSid to send
                    # audio back to the correct Twilio stream.
                    start_data = message.get("start", {})
                    self.stream_sid = start_data.get("streamSid")
                    self.call_sid = start_data.get("callSid", self.call_sid)

                    # Update the logger with the real call_sid now that
                    # Twilio has provided it (replaces the preliminary UUID).
                    if self.call_sid:
                        self.call_logger.call_sid = self.call_sid

                    self.call_logger.info(
                        f"Media stream started | "
                        f"stream_sid={self.stream_sid}",
                        event="twilio.stream.start",
                        stream_sid=self.stream_sid,
                    )
                    self.call_logger.mark_connected()

                    # Tell Cartesia how we'll be sending audio and what
                    # format we expect back. This MUST happen before
                    # forwarding any audio chunks.
                    await self._send_cartesia_session_config()

                elif event_type == "media":
                    # ─── HOT PATH ───────────────────────────────────
                    # This runs ~50 times/second per active call.
                    # Performance is critical here:
                    # - No allocations beyond the transcode buffers
                    # - No awaiting anything except the Cartesia send
                    # - audioop runs in C, not Python
                    # ────────────────────────────────────────────────
                    payload = message["media"]["payload"]
                    await self._transcode_and_forward(payload)

                elif event_type == "stop":
                    self.call_logger.info(
                        "Twilio media stream stopped",
                        event="twilio.stream.stop",
                    )
                    break

            except WebSocketDisconnect:
                # Caller hung up — break out of the loop
                break
            except json.JSONDecodeError as e:
                # Malformed message from Twilio — log and continue.
                # We don't crash on a single bad frame; telephony
                # networks occasionally corrupt packets.
                self.call_logger.warning(
                    f"Malformed JSON from Twilio: {e}",
                    event="twilio.parse_error",
                )

    # =========================================================================
    # Direction 2: Cartesia → Twilio (Agent's voice → Caller's ears)
    # =========================================================================

    async def _forward_cartesia_to_twilio(self) -> None:
        """
        Receive audio and events from Cartesia and forward to Twilio.

        Cartesia's agent streaming can send:
        - Binary frames: Raw audio data (TTS output)
        - Text frames: JSON control messages (text output, errors, etc.)

        We handle both formats because the Cartesia streaming protocol
        version may use either representation depending on configuration.
        """
        if not self._cartesia_ws:
            return

        try:
            async for message in self._cartesia_ws:
                if not self._running:
                    break

                try:
                    if isinstance(message, bytes):
                        # Binary frame: Raw audio from Cartesia TTS.
                        # When configured for µ-law 8kHz output, this
                        # is already in Twilio's native format — just
                        # base64-encode and send. No transcoding needed.
                        audio_b64 = base64.b64encode(message).decode(
                            "ascii"
                        )
                        await self._send_twilio_media(audio_b64)

                    elif isinstance(message, str):
                        data = json.loads(message)
                        msg_type = data.get("type") or data.get("event", "")

                        if msg_type in ("audio", "media_output"):
                            # JSON-wrapped audio payload
                            audio_data = data.get("data") or data.get(
                                "audio", {}
                            ).get("payload")
                            if audio_data:
                                await self._send_twilio_media(audio_data)

                        elif msg_type in ("text", "text_output"):
                            # Agent's text response — log for debugging.
                            # In Stage 2, this can be stored for analytics.
                            text = data.get("text") or data.get(
                                "content", ""
                            )
                            self.call_logger.info(
                                f"Agent says: {text[:120]}",
                                event="agent.text",
                            )

                        elif msg_type == "error":
                            self.call_logger.error(
                                f"Cartesia error: {data.get('message', data)}",
                                event="cartesia.error",
                            )

                        elif msg_type in ("end", "session_end"):
                            self.call_logger.info(
                                "Cartesia session ended",
                                event="cartesia.session.end",
                            )
                            break

                except Exception as e:
                    # Log but don't crash on individual message errors.
                    # One corrupted TTS chunk shouldn't kill the call.
                    self.call_logger.warning(
                        f"Error processing Cartesia message: {e}",
                        event="cartesia.process_error",
                    )

        except websockets.ConnectionClosed:
            # Cartesia disconnected — the agent ended the call or errored
            self.call_logger.info(
                "Cartesia WebSocket closed",
                event="cartesia.ws.closed",
            )

    # =========================================================================
    # Audio Transcoding Engine
    # =========================================================================

    async def _transcode_and_forward(self, base64_ulaw: str) -> None:
        """
        Core audio transcoding: Twilio µ-law → Cartesia PCM.

        This is the most performance-critical method in the entire bridge.
        It runs ~50 times per second per active call. Every microsecond
        of latency here directly impacts the caller's experience.

        Audio Pipeline (per chunk, ~20ms of audio):
        ┌─────────────────────────────────────────────────────────────┐
        │ Step 1: base64.b64decode()                                  │
        │   Base64 → raw bytes. Twilio wraps µ-law in base64 because │
        │   WebSocket JSON can't carry raw binary.                    │
        │   Input:  ~213 bytes (base64)  →  Output: 160 bytes (µ-law)│
        ├─────────────────────────────────────────────────────────────┤
        │ Step 2: audioop.ulaw2lin()                                  │
        │   G.711 µ-law → 16-bit linear PCM. This "expands" the      │
        │   logarithmic companding that µ-law uses to squeeze 14-bit  │
        │   dynamic range into 8 bits. The C implementation in        │
        │   audioop handles this in <10µs for a 20ms chunk.           │
        │   Input: 160 bytes (8-bit µ-law) → Output: 320 bytes (PCM) │
        ├─────────────────────────────────────────────────────────────┤
        │ Step 3: audioop.ratecv()                                    │
        │   Resample 8kHz → 16kHz. Speech recognition models are      │
        │   trained on 16kHz audio. Feeding 8kHz causes accuracy loss.│
        │   ratecv uses sinc interpolation and maintains state across │
        │   chunk boundaries to prevent discontinuity artifacts.      │
        │   Input: 320 bytes (8kHz) → Output: 640 bytes (16kHz)       │
        ├─────────────────────────────────────────────────────────────┤
        │ Step 4: WebSocket send                                      │
        │   Forward the PCM bytes to Cartesia. This is the only await │
        │   point — everything above is synchronous C code.           │
        └─────────────────────────────────────────────────────────────┘

        Why we DON'T buffer multiple chunks before sending:
            Buffering would reduce WebSocket overhead (fewer sends) but
            would add 40-100ms of latency. In voice AI, the STT engine
            needs audio as soon as possible to start recognizing speech.
            The faster it gets audio, the faster it can detect when the
            user finishes speaking, which directly reduces response time.

        Args:
            base64_ulaw: Base64-encoded G.711 µ-law audio from Twilio.
        """
        # Step 1: Base64 decode
        # Twilio wraps raw audio bytes in base64 for safe JSON transport.
        # base64 expands data by ~33%, so 160 bytes of audio becomes
        # ~213 characters in the JSON payload.
        ulaw_bytes = base64.b64decode(base64_ulaw)

        # Step 2: G.711 µ-law → 16-bit linear PCM
        # µ-law is a logarithmic companding algorithm (ITU-T G.711) that
        # maps 14-bit linear samples to 8-bit compressed values. This was
        # designed for the telephone network to maximize voice quality
        # within limited bandwidth. ulaw2lin reverses this compression.
        #
        # The second argument (2) is the output sample width in bytes:
        # 2 = 16-bit signed integers, which is what Cartesia expects.
        pcm_bytes = audioop.ulaw2lin(ulaw_bytes, PCM_SAMPLE_WIDTH)

        # Step 3: Resample 8kHz → 16kHz
        # Speech recognition models (including Cartesia's STT) are trained
        # on 16kHz audio because it captures frequencies up to 8kHz
        # (Nyquist), which includes all formants needed for accurate
        # phoneme recognition. Telephone-quality 8kHz only captures up
        # to 4kHz, missing sibilants (s, f, th) and fricatives.
        #
        # By upsampling, we don't ADD information, but we present the
        # audio in the format the model expects, avoiding resampling
        # artifacts that the model would have to handle internally.
        #
        # STATE PARAMETER: ratecv maintains interpolation state across
        # chunk boundaries. Without this, each chunk is resampled
        # independently, creating discontinuities (audible clicks)
        # at the joins. The state carries the fractional sample position
        # and the FIR filter's internal buffers.
        pcm_resampled, self._upsample_state = audioop.ratecv(
            pcm_bytes,
            PCM_SAMPLE_WIDTH,         # 2 bytes per sample
            CHANNELS,                 # mono
            TWILIO_SAMPLE_RATE,       # input: 8000 Hz
            CARTESIA_INPUT_SAMPLE_RATE,  # output: 16000 Hz
            self._upsample_state,     # carry state across chunks
        )

        # Step 4: Forward to Cartesia
        # Send raw PCM bytes. The session config (sent in Step 2 of the
        # Twilio start event) tells Cartesia the encoding and sample rate.
        if self._cartesia_ws:
            try:
                await self._cartesia_ws.send(pcm_resampled)
            except websockets.ConnectionClosed:
                self.call_logger.warning(
                    "Cartesia WS closed during audio forward",
                    event="cartesia.send_failed",
                )
                self._running = False

    # =========================================================================
    # Cartesia Session Management
    # =========================================================================

    async def _send_cartesia_session_config(self) -> None:
        """
        Send session initialization to Cartesia's agent streaming endpoint.

        This message tells Cartesia:
        1. What audio format we're sending (PCM 16-bit, 16kHz)
        2. What audio format we want back (µ-law 8kHz — Twilio native)

        By requesting µ-law output, we avoid a second transcode step on
        the Agent→Twilio path. Cartesia's TTS generates audio directly
        in the telephony format.
        """
        if not self._cartesia_ws:
            return

        config_message = {
            "event": "start",
            "config": {
                "input_format": {
                    "encoding": "pcm_s16le",
                    "sample_rate": CARTESIA_INPUT_SAMPLE_RATE,
                    "channels": CHANNELS,
                },
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_mulaw",
                    "sample_rate": TWILIO_SAMPLE_RATE,
                },
            },
        }

        await self._cartesia_ws.send(json.dumps(config_message))
        self.call_logger.info(
            f"Cartesia session configured: "
            f"input=PCM-s16le@{CARTESIA_INPUT_SAMPLE_RATE}Hz, "
            f"output=µ-law@{TWILIO_SAMPLE_RATE}Hz",
            event="cartesia.session.configured",
        )

    # =========================================================================
    # Twilio Media Output
    # =========================================================================

    async def _send_twilio_media(self, base64_audio: str) -> None:
        """
        Send an audio chunk back to Twilio as a 'media' event.

        Twilio expects the audio payload as base64-encoded bytes within
        a JSON media event. The streamSid tells Twilio which active
        stream this audio belongs to (important when a call has multiple
        concurrent streams, e.g., conference calls).

        This method also tracks TTFA: the first time we send audio to
        Twilio is the moment the caller hears the agent for the first time.
        """
        if not self.stream_sid:
            # Can't send audio until we know the stream ID from Twilio
            return

        # Track Time-to-First-Audio
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
            # Caller hung up while we were trying to send audio.
            # This is normal and expected — don't log as error.
            self.call_logger.info(
                "Cannot send audio — Twilio WS closed",
                event="twilio.send_closed",
            )
            self._running = False

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def _cleanup(self) -> None:
        """
        Gracefully close all connections and release resources.

        Called in the `finally` block of run(), so it executes regardless
        of how the call ended (normal hangup, error, crash).
        """
        if self._cartesia_ws:
            try:
                await self._cartesia_ws.close()
            except Exception:
                pass  # Best-effort close — don't mask the original error

        self.call_logger.info(
            f"Bridge cleanup complete | TTFA={self.call_logger.ttfa_ms}ms",
            event="bridge.cleanup",
        )
