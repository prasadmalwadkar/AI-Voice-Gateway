"""
HELOC VoIP Bridge — FastAPI Application 
===========================================================================

This module creates the unified server that handles BOTH:
  1. Cartesia's native agent protocol (for CLI/browser testing)
  2. Twilio Media Streams (for production phone calls)

Architecture:
┌──────────────────────────────────────────────────────┐
│              Unified FastAPI Application              │
│                                                      │
│  GET  /voice        → TwiML to connect Twilio stream │
│  WS   /media-stream → Twilio ↔ Cartesia audio bridge│
│  GET  /health       → Health check for load balancers│
│  ...                → Cartesia SDK's native routes   │
└──────────────────────────────────────────────────────┘

Implementation:
    VoiceAgentApp (from Cartesia Line SDK) wraps a FastAPI instance
    internally. We access that instance and add our Twilio-specific
    routes to it. This gives us a single port, single process server
    that serves both Cartesia native clients and Twilio telephony.

    If the VoiceAgentApp's internal structure changes in a future SDK
    version, the fallback creates a standalone FastAPI app. The agent
    definition is shared either way.
"""

import atexit
import base64
import html
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from agent.main_agent import create_voice_app
from app.logging_config import CallLogger, setup_logging
from app.twilio_bridge import TwilioBridge
from app.twilio_line_bridge import TwilioLineBridge
from app.path2_transcript_debug import (
    keepalive_loop,
    path2_transcript_debug_enabled,
    path2_transcript_debug_token,
    register_client,
    unregister_client,
)

# -- Initialize structured logging at module load --
setup_logging()
atexit.register(logging.shutdown)
module_logger = logging.getLogger("voip.main")

_PREFILL_QUERY_FIELDS = frozenset(
    {
        "address_line_1",
        "apt_suite",
        "city",
        "state",
        "zipcode",
        "ownership_type",
        "occupancy_type",
        "property_type",
        "is_currently_for_sale",
        "annual_pre_tax_income",
        "other_income",
        "primary_source_of_income",
        "first_name",
        "middle_name",
        "last_name",
        "suffix",
        "date_of_birth",
        "mobile_phone_number",
        "phone",
        "phone_number",
        "email_address",
        "email",
    }
)


def _normalize_phone_key(phone: str | None) -> list[str]:
    digits = "".join(ch for ch in (phone or "") if ch.isdigit())
    if not digits:
        return []
    keys = [digits]
    if len(digits) == 10:
        keys.extend([f"1{digits}", f"+1{digits}"])
    elif len(digits) == 11 and digits.startswith("1"):
        keys.extend([digits[1:], f"+{digits}"])
    elif phone and phone.startswith("+"):
        keys.append(phone.strip())
    return list(dict.fromkeys(keys))


def _phone_tail10(phone: str | None) -> str:
    digits = "".join(ch for ch in (phone or "") if ch.isdigit())
    return digits[-10:] if len(digits) >= 10 else digits


def _decode_json_payload(value: str, *, b64: bool = False) -> dict[str, Any]:
    if not value:
        return {}
    try:
        raw = value
        if b64:
            padding = "=" * (-len(value) % 4)
            raw = base64.urlsafe_b64decode((value + padding).encode()).decode()
        data = json.loads(raw)
    except Exception as e:
        module_logger.warning("Ignoring invalid customer prefill payload: %s", e)
        return {}
    return data if isinstance(data, dict) else {}


def _load_prefill_lookup() -> dict[str, Any]:
    raw = os.getenv("TWILIO_PREFILL_BY_PHONE_JSON", "").strip()
    if not raw:
        path = os.getenv("TWILIO_PREFILL_BY_PHONE_FILE", "").strip()
        if not path:
            disable_auto = os.getenv(
                "TWILIO_PREFILL_DISABLE_AUTO_CUSTOMERS_JSON", ""
            ).strip().lower() in ("1", "true", "yes", "on")
            if not disable_auto:
                cand = (
                    Path(__file__).resolve().parent.parent
                    / "data"
                    / "prefill"
                    / "customers.json"
                )
                if cand.is_file():
                    path = str(cand)
                    module_logger.info(
                        "Using default phone prefill file %s (set "
                        "TWILIO_PREFILL_DISABLE_AUTO_CUSTOMERS_JSON=1 to skip)",
                        path,
                    )
        if path:
            try:
                raw = Path(path).read_text(encoding="utf-8")
            except OSError as e:
                module_logger.warning("Could not read TWILIO_PREFILL_BY_PHONE_FILE: %s", e)
                raw = ""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        module_logger.warning("Invalid TWILIO_PREFILL_BY_PHONE_JSON: %s", e)
        return {}
    return data if isinstance(data, dict) else {}


def _prefill_for_phone(phone: str | None) -> dict[str, Any]:
    lookup = _load_prefill_lookup()
    for key in _normalize_phone_key(phone):
        value = lookup.get(key)
        if isinstance(value, dict):
            return dict(value)
    caller_tail = _phone_tail10(phone)
    if len(caller_tail) != 10:
        return {}
    for lk, value in lookup.items():
        if not isinstance(value, dict):
            continue
        lk_tail = _phone_tail10(str(lk))
        if lk_tail == caller_tail:
            return dict(value)
        embedded = (
            value.get("mobile_phone_number")
            or value.get("phone_number")
            or value.get("phone")
            or ""
        )
        if _phone_tail10(str(embedded)) == caller_tail:
            return dict(value)
    return {}


async def _request_params(request: Request) -> dict[str, str]:
    params = {str(k): str(v) for k, v in request.query_params.items()}
    if request.method.upper() == "POST":
        try:
            form = await request.form()
            params.update({str(k): str(v) for k, v in form.items()})
        except Exception as e:
            module_logger.debug("No parseable /voice form body: %s", e)
    return params


def _query_prefill(params: dict[str, str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    payload.update(_decode_json_payload(params.get("prefill_json", "")))
    payload.update(_decode_json_payload(params.get("prefill_b64", ""), b64=True))
    for key, value in params.items():
        if key in _PREFILL_QUERY_FIELDS and value != "":
            payload[key] = value
    return payload


def _stream_parameter(name: str, value: str) -> str:
    return (
        f'<Parameter name="{html.escape(name, quote=True)}" '
        f'value="{html.escape(value, quote=True)}" />'
    )


def _build_stream_parameters(params: dict[str, str]) -> tuple[str, list[str]]:
    caller_from = params.get("From") or params.get("caller_from") or ""
    caller_to = params.get("To") or params.get("caller_to") or ""
    payload = _prefill_for_phone(caller_from)
    payload.update(_query_prefill(params))
    field_names = sorted(payload.keys())

    stream_params = []
    if caller_from:
        stream_params.append(_stream_parameter("caller_from", caller_from))
    if caller_to:
        stream_params.append(_stream_parameter("caller_to", caller_to))
    if payload:
        raw_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
        encoded = base64.urlsafe_b64encode(raw_json.encode()).decode().rstrip("=")
        stream_params.append(_stream_parameter("prefill_b64", encoded))
        stream_params.append(_stream_parameter("prefill_fields", ",".join(field_names)))
    return "".join(stream_params), field_names


# =============================================================================
# Extend VoiceAgentApp with Twilio Routes
# =============================================================================
# VoiceAgentApp wraps a FastAPI instance. We extract it and add our routes.
# This approach gives us Cartesia's native functionality (STT, TTS, agent
# lifecycle) PLUS our custom Twilio telephony bridge on the same server.

voice_app = create_voice_app()

# Access the internal FastAPI app from VoiceAgentApp.
# The SDK stores it as `.fastapi_app` (confirmed in cartesia-line v0.2.8 source).
# If this changes in a future SDK version, the fallback creates a standalone app.
try:
    app: FastAPI = voice_app.fastapi_app  # type: ignore[union-attr]
    module_logger.info(
        "Successfully extended VoiceAgentApp — "
        "Cartesia native routes + Twilio routes on same server"
    )
except AttributeError:
    # Fallback: VoiceAgentApp's internal structure changed.
    # Create a standalone FastAPI app. Cartesia native testing can still
    # be done by running `python -m agent.main_agent` separately.
    module_logger.warning(
        "VoiceAgentApp.fastapi_app not accessible — creating standalone FastAPI. "
        "Run `python -m agent.main_agent` separately for Cartesia CLI testing."
    )
    app = FastAPI(
        title="AI Voice Gateway",
        description="AI Voice Gateway — Telephony Agent Bridge",
        version="1.0.0-stage1",
    )


# =============================================================================
# Route: GET /voice — TwiML Response for Twilio
# =============================================================================

@app.get("/voice", response_class=Response)
@app.post("/voice", response_class=Response)
async def voice_twiml(request: Request):
    """
    Return TwiML instructions that tell Twilio to open a bidirectional
    Media Stream WebSocket back to our /media-stream endpoint.

    Flow:
    1. Twilio receives an inbound call (or makes an outbound call)
    2. Twilio fetches this URL to get call handling instructions
    3. We return TwiML with a <Stream> verb pointing to /media-stream
    4. Twilio opens a WebSocket to /media-stream and starts streaming audio

    The <Connect><Stream> TwiML verb is the key to real-time audio access.
    Without it, Twilio only provides post-call recordings.

    The WebSocket URL is built dynamically from the incoming request's
    Host header. This is critical because:
    - In development: the URL is ws://localhost:8000/media-stream
    - Behind ngrok: wss://abc123.ngrok.io/media-stream
    - In production: wss://voip.westcapitallending.com/media-stream
    We MUST use wss:// for Twilio — it rejects insecure ws:// connections.
    """
    host = request.headers.get("host", "localhost:8000")

    # Always use wss:// — Twilio requires TLS for media streams.
    # Even in local dev with ngrok, the tunnel provides TLS.
    ws_url = f"wss://{host}/media-stream"
    params = await _request_params(request)
    stream_parameters, prefill_fields = _build_stream_parameters(params)

    twiml_response = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="{html.escape(ws_url, quote=True)}">'
        f"{stream_parameters}"
        "</Stream>"
        "</Connect>"
        "</Response>"
    )

    module_logger.info(
        "Serving TwiML — stream URL: %s | prefill_fields=%s",
        ws_url,
        prefill_fields,
    )

    return Response(
        content=twiml_response,
        media_type="application/xml",
    )


# =============================================================================
# Route: WS /media-stream — Twilio Media Stream Handler
# =============================================================================

@app.websocket("/media-stream")
async def media_stream_handler(websocket: WebSocket):
    """
    Twilio Media Stream WebSocket endpoint.

    This is the production entry point for all phone calls. When Twilio
    receives the TwiML from /voice, it opens a WebSocket here and begins
    streaming the caller's raw audio. We bridge this audio to the
    Cartesia voice agent and forward the agent's speech back to Twilio.

    Error Handling Strategy:
    ┌──────────────────────┬─────────────────────────────────────────┐
    │ Exception            │ Meaning & Handling                      │
    ├──────────────────────┼─────────────────────────────────────────┤
    │ WebSocketDisconnect  │ Caller hung up. Normal. Log and exit.   │
    │ ConnectionRefusedError│ Cartesia not reachable. Log as error.  │
    │ ConnectionError      │ Network issue. Log as warning.          │
    │ Exception            │ Unexpected. Log full traceback. Exit.   │
    └──────────────────────┴─────────────────────────────────────────┘

    The outer try/except ensures the FastAPI server NEVER crashes from
    a single dropped call. Each call is isolated in its own handler.
    """
    await websocket.accept()

    # Generate a preliminary call ID for logging until Twilio provides
    # the real call_sid in the 'start' event. Using UUID prefix ensures
    # log lines are traceable even before Twilio sends metadata.
    preliminary_call_id = str(uuid.uuid4())[:8]
    call_logger = CallLogger(call_sid=preliminary_call_id)
    call_logger.info("WebSocket accepted — awaiting Twilio stream", event="ws.accepted")

    voice_mode = os.getenv("TWILIO_VOICE_MODE", "path2").strip().lower()

    # -- Path2: Twilio → Ink STT → Line /ws (LlmAgent) → Sonic TTS → Twilio --
    if voice_mode in ("path2", "line", "local"):
        bridge = TwilioLineBridge(
            twilio_ws=websocket,
            call_logger=call_logger,
        )
    else:
        # -- Cartesia Calls API (cloud agent) --
        cartesia_agent_url = os.getenv("CARTESIA_AGENT_WS_URL", "")

        if cartesia_agent_url:
            cartesia_ws_url = cartesia_agent_url
        else:
            cartesia_ws_url = (
                "wss://api.cartesia.ai/agents/stream/agent_6F5z1Fb3qRAAh6WVxj68bH"
            )

        bridge = TwilioBridge(
            twilio_ws=websocket,
            cartesia_ws_url=cartesia_ws_url,
            call_logger=call_logger,
        )

    try:
        await bridge.run()

    except WebSocketDisconnect:
        # Normal: caller hung up — this is NOT an error condition.
        # The bridge's run() method handles this internally, but the
        # outer handler catches it too as a safety net.
        call_logger.info(
            "Call ended normally (caller disconnected)",
            event="call.ended.normal",
        )

    except ConnectionError as e:
        # Network issue between us and Cartesia. This could be:
        # - Cartesia's API is down
        # - Network partition
        # - DNS failure
        # Log as WARNING, not ERROR, because it's external to our code.
        call_logger.warning(
            f"Cartesia connection failed: {e}",
            event="cartesia.connection_failed",
            error=str(e),
        )

    except Exception as e:
        # Catch-all for truly unexpected errors. Log the full traceback
        # so we can diagnose issues from production logs.
        # CRITICAL: We do NOT re-raise here. The server must stay up
        # even if one call's handler crashes.
        call_logger.error(
            f"Unexpected error in media stream handler: "
            f"{type(e).__name__}: {e}",
            event="bridge.fatal",
            exc_info=True,
        )

    finally:
        # Ensure the WebSocket is closed in all cases.
        # FastAPI may have already closed it, so we catch exceptions.
        try:
            await websocket.close()
        except Exception:
            pass

        call_logger.info(
            f"Handler complete | final_ttfa={call_logger.ttfa_ms}ms",
            event="ws.handler.done",
        )


# =============================================================================
# Route: WS /debug/path2-transcripts — Path2 STT transcript fan-out (orchestrator CLI)
# =============================================================================


@app.websocket("/debug/path2-transcripts")
async def path2_transcript_debug_ws(websocket: WebSocket):
    """Live STT events for local debugging (no Twilio protocol — safe for mimic)."""
    await websocket.accept()
    if not path2_transcript_debug_enabled():
        await websocket.close(code=1008)
        return
    expected_token = path2_transcript_debug_token()
    provided_token = websocket.query_params.get("token", "")
    if not expected_token or provided_token != expected_token:
        await websocket.close(code=1008)
        return
    await register_client(websocket)
    try:
        await keepalive_loop(websocket)
    finally:
        await unregister_client(websocket)


# =============================================================================
# Route: GET /health — Health Check
# =============================================================================

@app.get("/health")
async def health_check():
    """
    Health check endpoint for load balancers, Kubernetes probes,
    and uptime monitoring.

    Returns 200 with service metadata. In Stage 2, this can be extended
    to check Cartesia API connectivity and Gemini API key validity.
    """
    return {
        "status": "healthy",
        "service": "AI-Voice-Gateway",
        "version": "1.0.0-stage1",
        "stage": "1",
    }
