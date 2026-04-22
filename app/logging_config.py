"""
Structured JSON Logging with TTFA (Time-to-First-Audio) Tracking
=================================================================

Production voice applications need structured, machine-parseable logs for
three reasons:

1. DEBUGGING LIVE CALLS: When a caller reports a bad experience, you need
   to find their call_sid in your logs and trace the exact sequence of
   events — in real time, not by grepping unstructured text.

2. PERFORMANCE MONITORING: TTFA (Time-to-First-Audio) is the single most
   important metric for voice AI UX. It measures the delay between the
   caller connecting and hearing the agent's first words. Anything above
   1.5 seconds feels broken. We track this per-call.

3. ALERTING: Structured JSON logs integrate directly with Datadog,
   CloudWatch, or ELK for automated alerting on error spikes,
   high TTFA percentiles, or connection failures.

Usage:
    from app.logging_config import setup_logging, CallLogger

    setup_logging()
    logger = CallLogger(call_sid="CA123...")
    logger.mark_connected()
    # ... audio processing ...
    logger.mark_first_audio()  # Logs TTFA automatically
"""

import json
import logging
import os
import time
from logging.handlers import RotatingFileHandler
from typing import Optional


class JSONFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.

    Each log line is a self-contained JSON document with standard fields
    (timestamp, level, message) plus optional call-specific fields
    (call_sid, stream_sid, event, ttfa_ms). This format is natively
    ingestible by all major log aggregation platforms.

    Example output:
        {"timestamp": "2026-04-17 12:30:00,000", "level": "INFO",
         "logger": "voip.bridge", "message": "Call connected",
         "call_sid": "CA123...", "event": "call.connected"}
    """

    # Fields that, if present on the LogRecord, should be included in output.
    # These are set via the `extra={}` dict passed to logger.info() etc.
    EXTRA_FIELDS = ("call_sid", "stream_sid", "event", "ttfa_ms")

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach call-specific metadata if present on the record.
        # These fields are injected by CallLogger via the `extra` kwarg.
        for field in self.EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                log_entry[field] = value

        # Include exception info for error-level logs
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class CallLogger:
    """
    Per-call structured logger with automatic TTFA tracking.

    Each phone call gets its own CallLogger instance. This provides:
    - Automatic call_sid injection into every log line
    - TTFA measurement between mark_connected() and mark_first_audio()
    - Clean API that avoids passing `extra={}` everywhere

    Lifecycle:
        logger = CallLogger(call_sid="CA123...")
        logger.mark_connected()        # Call starts
        logger.info("Processing...")    # Regular logging
        logger.mark_first_audio()       # Logs TTFA automatically
        logger.info("Call ended")       # Still includes call_sid

    TTFA Measurement:
        We use time.monotonic() instead of time.time() because:
        - monotonic() is immune to NTP clock adjustments
        - monotonic() has nanosecond resolution on modern kernels
        - We only care about the *duration*, not absolute wall time
    """

    def __init__(self, call_sid: str):
        self.call_sid = call_sid
        # Namespace loggers under "voip.call.<prefix>" for granular filtering.
        # Using first 8 chars of call_sid keeps log output readable while
        # still being unique enough to distinguish concurrent calls.
        self.logger = logging.getLogger(f"voip.call.{call_sid[:8]}")

        # Monotonic timestamps for TTFA calculation
        self._connection_time: Optional[float] = None
        self._first_audio_time: Optional[float] = None

    def mark_connected(self) -> None:
        """Record the moment the caller connects (Twilio stream starts)."""
        self._connection_time = time.monotonic()
        self.info("Call connected", event="call.connected")

    def mark_first_audio(self) -> None:
        """
        Record the moment first audio is sent to the caller.

        Automatically computes and logs TTFA (Time-to-First-Audio) if
        mark_connected() was called previously. TTFA is in milliseconds.

        Industry benchmarks:
          - <500ms:  Excellent (feels instant)
          - 500-1000ms: Good (barely noticeable)
          - 1000-1500ms: Acceptable (slight pause)
          - >1500ms: Poor (caller thinks something is wrong)
        """
        self._first_audio_time = time.monotonic()
        ttfa_ms = None
        if self._connection_time is not None:
            ttfa_ms = round(
                (self._first_audio_time - self._connection_time) * 1000, 2
            )
        self.info(
            "First audio sent to caller",
            event="audio.first_byte",
            ttfa_ms=ttfa_ms,
        )

    @property
    def ttfa_ms(self) -> Optional[float]:
        """Return the TTFA in milliseconds, or None if not yet measured."""
        if self._connection_time and self._first_audio_time:
            return round(
                (self._first_audio_time - self._connection_time) * 1000, 2
            )
        return None

    # -- Convenience logging methods --
    # These inject call_sid automatically so callers don't have to pass it.

    def info(self, message: str, **extra) -> None:
        """Log at INFO level with automatic call_sid injection."""
        self.logger.info(
            message, extra={"call_sid": self.call_sid, **extra}
        )

    def debug(self, message: str, **extra) -> None:
        """Log at DEBUG level with automatic call_sid injection."""
        self.logger.debug(
            message, extra={"call_sid": self.call_sid, **extra}
        )

    def warning(self, message: str, **extra) -> None:
        """Log at WARNING level with automatic call_sid injection."""
        self.logger.warning(
            message, extra={"call_sid": self.call_sid, **extra}
        )

    def error(
        self, message: str, exc_info: bool = False, **extra
    ) -> None:
        """Log at ERROR level with automatic call_sid injection."""
        self.logger.error(
            message,
            exc_info=exc_info,
            extra={"call_sid": self.call_sid, **extra},
        )


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure the root 'voip' logger with JSON formatting.

    Call this once at application startup (in app/main.py or run.py).
    All loggers under the 'voip.*' namespace will inherit this config.

    Args:
        level: Minimum log level. Default INFO. Use DEBUG for local dev.
    """
    root_logger = logging.getLogger("voip")
    if root_logger.handlers:
        root_logger.setLevel(level)
        return

    fmt = JSONFormatter()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    root_logger.addHandler(stream_handler)

    if os.getenv("LOG_FILE_DISABLE", "").strip().lower() not in ("1", "true", "yes"):
        log_path = os.getenv("LOG_FILE_PATH", "").strip() or os.path.join("logs", "voip.log")
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        root_logger.addHandler(file_handler)

    root_logger.setLevel(level)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
