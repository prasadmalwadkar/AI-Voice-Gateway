"""
HELOC VoIP Bridge — Application Entry Point
=============================================

This is the production entry point for the VoIP bridge server.

Usage:
    # Start the server (default: 0.0.0.0:8000)
    python run.py

    # Or with uvicorn directly (for custom options)
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

    # For Cartesia-native testing (no Twilio required)
    python -m agent.heloc_agent

Endpoints once running:
    GET  /voice         → TwiML for Twilio webhook
    WS   /media-stream  → Twilio Media Stream handler
    GET  /health        → Health check
    GET  /docs          → Auto-generated API docs (FastAPI)

Local Development with ngrok:
    1. python run.py
    2. ngrok http 8000
    3. Copy the ngrok HTTPS URL
    4. Set it as your Twilio phone number's webhook:
       https://abc123.ngrok.io/voice
"""

import os

import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file.
# This MUST happen before importing app modules that read os.getenv().
load_dotenv()


def main():
    """Start the HELOC VoIP Bridge server."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    print(
        f"\n"
        f"  ╔══════════════════════════════════════════╗\n"
        f"  ║   HELOC VoIP Bridge — West Capital       ║\n"
        f"  ║   Stage 1: Voice Agent Core              ║\n"
        f"  ╠══════════════════════════════════════════╣\n"
        f"  ║   Server:    http://{host}:{port}         \n"
        f"  ║   TwiML:     http://{host}:{port}/voice   \n"
        f"  ║   Health:    http://{host}:{port}/health   \n"
        f"  ║   API Docs:  http://{host}:{port}/docs     \n"
        f"  ╚══════════════════════════════════════════╝\n"
    )

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        # WebSocket connections require HTTP/1.1 upgrade support.
        # uvicorn handles this natively with its default httptools parser.
        log_level="info",
        # Disable access logs — our structured JSON logger handles logging.
        # uvicorn's default access logs are unstructured and noisy.
        access_log=False,
        # reload=True is useful for development but should be False
        # in production to avoid file-watching overhead.
        reload=False,
    )


if __name__ == "__main__":
    main()
