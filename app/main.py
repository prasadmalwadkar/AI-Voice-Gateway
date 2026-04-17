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

import logging
import os
import uuid

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from agent.main_agent import create_voice_app
from app.logging_config import CallLogger, setup_logging
from app.twilio_bridge import TwilioBridge

# -- Initialize structured logging at module load --
setup_logging()
module_logger = logging.getLogger("voip.main")


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

    twiml_response = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="{ws_url}" />'
        "</Connect>"
        "</Response>"
    )

    module_logger.info(f"Serving TwiML — stream URL: {ws_url}")

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

    # -- Build the Cartesia agent streaming URL --
    # This URL connects our bridge to Cartesia's cloud infrastructure
    # (or a local VoiceAgentApp instance for development).
    cartesia_api_key = os.getenv("CARTESIA_API_KEY", "")
    cartesia_agent_url = os.getenv("CARTESIA_AGENT_WS_URL", "")

    if cartesia_agent_url:
        # Custom URL: typically ws://localhost:8000 for local VoiceAgentApp
        cartesia_ws_url = cartesia_agent_url
    else:
        # Default: Cartesia's cloud agent streaming endpoint
        cartesia_ws_url = (
            f"wss://api.cartesia.ai/agents/stream/"
            f"?api_key={cartesia_api_key}"
            f"&cartesia_version=2024-11-13"
        )

    # -- Create and run the bridge --
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
