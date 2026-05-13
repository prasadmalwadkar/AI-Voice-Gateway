"""
Intake Queue — Abstract Queue Interface + Filesystem Backend
=============================================================

Publishes validated HELOC intake payloads for asynchronous form filling.
The voice agent collects all answers, then publishes the complete payload
here. A separate consumer process picks it up and fills the WCL form.

Backend abstraction:
    - ``FilesystemQueue`` (default): JSON files in ``data/queue/`` dirs.
      Zero dependencies — perfect for dev and single-server deployments.
    - Easily swappable to Redis Streams, SQS, etc. by implementing
      ``IntakeQueue`` ABC.

Usage::

    from agent.queue_publisher import get_queue, IntakePayload

    queue = get_queue()
    payload = IntakePayload.from_application_state(state, call_sid=...)
    msg_id = await queue.publish(payload)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, List, Optional

logger = logging.getLogger("voip.queue")

# Default queue directories (relative to project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_QUEUE_BASE = _PROJECT_ROOT / "data" / "queue"

# Retry configuration
MAX_RETRIES = int(os.getenv("INTAKE_QUEUE_MAX_RETRIES", "2"))


# =============================================================================
# IntakePayload — Serializable snapshot of a completed intake
# =============================================================================

@dataclass
class IntakePayload:
    """Complete HELOC intake data ready for form filling."""

    # Identifiers
    message_id: str = ""
    call_sid: str = ""
    caller_from: str = ""
    caller_to: str = ""

    # Timestamps
    submitted_at: str = ""
    completed_at: str = ""  # Set by consumer on success

    # Application data (all fields from HelocApplicationState)
    application_data: dict[str, Any] = field(default_factory=dict)

    # Verbatim STT sources per field (for audit)
    voice_field_sources: dict[str, str] = field(default_factory=dict)

    # Processing metadata
    status: str = "pending"  # pending | processing | completed | failed | dlq
    attempt: int = 0
    max_retries: int = MAX_RETRIES
    errors: list[str] = field(default_factory=list)

    # Status callback URL (optional)
    callback_url: str = ""

    @classmethod
    def from_application_state(
        cls,
        state: Any,
        *,
        call_sid: str = "",
        caller_from: str = "",
        caller_to: str = "",
        voice_field_sources: Optional[dict[str, str]] = None,
        callback_url: str = "",
    ) -> "IntakePayload":
        """Build payload from a HelocApplicationState instance."""
        # Export all non-None, non-password fields
        _SENSITIVE = frozenset({"password", "confirm_password"})
        data = state.model_dump(mode="json", exclude_none=True)
        for key in _SENSITIVE:
            data.pop(key, None)

        return cls(
            message_id=str(uuid.uuid4()),
            call_sid=call_sid,
            caller_from=caller_from,
            caller_to=caller_to,
            submitted_at=datetime.now(timezone.utc).isoformat(),
            application_data=data,
            voice_field_sources=dict(voice_field_sources or {}),
            callback_url=callback_url or os.getenv("INTAKE_STATUS_CALLBACK_URL", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntakePayload":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "IntakePayload":
        return cls.from_dict(json.loads(raw))


# =============================================================================
# IntakeQueue — Abstract base class
# =============================================================================

class IntakeQueue(ABC):
    """Abstract queue interface for intake payloads."""

    @abstractmethod
    async def publish(self, payload: IntakePayload) -> str:
        """Publish a payload to the queue. Returns message_id."""
        ...

    @abstractmethod
    async def consume_one(self) -> Optional[IntakePayload]:
        """Pick up the next pending payload. Returns None if queue is empty."""
        ...

    @abstractmethod
    async def ack(self, message_id: str) -> None:
        """Acknowledge successful processing — move to completed."""
        ...

    @abstractmethod
    async def nack(self, message_id: str, error: str) -> None:
        """Negative acknowledge — retry or move to DLQ after max retries."""
        ...

    @abstractmethod
    async def list_pending(self) -> List[IntakePayload]:
        """List all pending payloads (for monitoring/debugging)."""
        ...

    @abstractmethod
    async def list_dlq(self) -> List[IntakePayload]:
        """List all dead-letter payloads."""
        ...


# =============================================================================
# FilesystemQueue — JSON file-based implementation
# =============================================================================

class FilesystemQueue(IntakeQueue):
    """
    Queue backed by JSON files in directories.

    Directory structure::

        data/queue/
        ├── pending/       ← New payloads land here
        ├── processing/    ← Consumer moves here while working
        ├── completed/     ← Successful fills
        └── dlq/           ← Failed after max retries

    Atomicity: Uses os.rename() for atomic moves (same filesystem).
    Concurrency: Single consumer assumed for filesystem backend.
    For multi-consumer, use Redis Streams.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = base_dir or _DEFAULT_QUEUE_BASE
        self.pending_dir = self.base_dir / "pending"
        self.processing_dir = self.base_dir / "processing"
        self.completed_dir = self.base_dir / "completed"
        self.dlq_dir = self.base_dir / "dlq"

        # Ensure directories exist
        for d in (self.pending_dir, self.processing_dir, self.completed_dir, self.dlq_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _payload_path(self, directory: Path, message_id: str) -> Path:
        return directory / f"{message_id}.json"

    def _find_payload_file(self, message_id: str) -> Optional[Path]:
        """Find a payload file across all directories."""
        for d in (self.processing_dir, self.pending_dir, self.completed_dir, self.dlq_dir):
            p = self._payload_path(d, message_id)
            if p.exists():
                return p
        return None

    async def publish(self, payload: IntakePayload) -> str:
        if not payload.message_id:
            payload.message_id = str(uuid.uuid4())

        path = self._payload_path(self.pending_dir, payload.message_id)
        path.write_text(payload.to_json(), encoding="utf-8")

        logger.info(
            "Intake payload published | message_id=%s call_sid=%s fields=%d",
            payload.message_id,
            payload.call_sid,
            len(payload.application_data),
        )
        return payload.message_id

    async def consume_one(self) -> Optional[IntakePayload]:
        # Sort by filename (UUID-based, but check mtime for ordering)
        pending_files = sorted(
            self.pending_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
        )
        if not pending_files:
            return None

        src = pending_files[0]
        try:
            payload = IntakePayload.from_json(src.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("Failed to parse payload %s: %s", src.name, e)
            # Move corrupted file to DLQ
            dst = self._payload_path(self.dlq_dir, src.stem)
            shutil.move(str(src), str(dst))
            return None

        # Move to processing
        payload.status = "processing"
        payload.attempt += 1
        dst = self._payload_path(self.processing_dir, payload.message_id)
        dst.write_text(payload.to_json(), encoding="utf-8")
        src.unlink(missing_ok=True)

        logger.info(
            "Payload picked up for processing | message_id=%s attempt=%d/%d",
            payload.message_id,
            payload.attempt,
            payload.max_retries,
        )
        return payload

    async def ack(self, message_id: str) -> None:
        src = self._payload_path(self.processing_dir, message_id)
        if not src.exists():
            logger.warning("ACK: payload %s not found in processing/", message_id)
            return

        payload = IntakePayload.from_json(src.read_text(encoding="utf-8"))
        payload.status = "completed"
        payload.completed_at = datetime.now(timezone.utc).isoformat()

        dst = self._payload_path(self.completed_dir, message_id)
        dst.write_text(payload.to_json(), encoding="utf-8")
        src.unlink(missing_ok=True)

        logger.info("Payload completed | message_id=%s call_sid=%s", message_id, payload.call_sid)

    async def nack(self, message_id: str, error: str) -> None:
        src = self._payload_path(self.processing_dir, message_id)
        if not src.exists():
            logger.warning("NACK: payload %s not found in processing/", message_id)
            return

        payload = IntakePayload.from_json(src.read_text(encoding="utf-8"))
        payload.errors.append(f"[attempt {payload.attempt}] {error}")

        if payload.attempt >= payload.max_retries:
            # Move to DLQ
            payload.status = "dlq"
            dst = self._payload_path(self.dlq_dir, message_id)
            dst.write_text(payload.to_json(), encoding="utf-8")
            src.unlink(missing_ok=True)
            logger.error(
                "Payload moved to DLQ after %d attempts | message_id=%s error=%s",
                payload.attempt,
                message_id,
                error[:200],
            )
        else:
            # Move back to pending for retry
            payload.status = "pending"
            dst = self._payload_path(self.pending_dir, message_id)
            dst.write_text(payload.to_json(), encoding="utf-8")
            src.unlink(missing_ok=True)
            logger.warning(
                "Payload returned to pending for retry | message_id=%s attempt=%d/%d error=%s",
                message_id,
                payload.attempt,
                payload.max_retries,
                error[:200],
            )

    async def list_pending(self) -> List[IntakePayload]:
        return await self._list_dir(self.pending_dir)

    async def list_dlq(self) -> List[IntakePayload]:
        return await self._list_dir(self.dlq_dir)

    async def _list_dir(self, directory: Path) -> List[IntakePayload]:
        results = []
        for f in sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime):
            try:
                results.append(IntakePayload.from_json(f.read_text(encoding="utf-8")))
            except Exception:
                continue
        return results


# =============================================================================
# Redis Streams Backend (stub — activate with INTAKE_QUEUE_BACKEND=redis)
# =============================================================================

class RedisStreamQueue(IntakeQueue):
    """
    Redis Streams backend.

    Requires env vars:
      REDIS_URL           — connection URL (default: redis://localhost:6379/0)

    Implementation TODOs:
      publish()      → XADD STREAM_KEY * json <payload_json>
      consume_one()  → XREADGROUP GROUP form-fillers <consumer_id>
                        COUNT 1 BLOCK 1000 STREAMS STREAM_KEY >
      ack()          → XACK STREAM_KEY form-fillers <redis_message_id>
      nack()         → if attempts < max_retries: leave in PEL (auto-redeliver)
                       else: XADD DLQ_KEY ... then XACK to purge from PEL
      list_pending() → XRANGE STREAM_KEY - + COUNT 100
      list_dlq()     → XRANGE DLQ_KEY - + COUNT 100

    Why Streams over Pub/Sub: persistent storage, consumer groups, ACK semantics,
    and a pending entries list (PEL) for retries — Pub/Sub is fire-and-forget.
    """

    STREAM_KEY = "intake:queue"
    DLQ_KEY = "intake:dlq"
    GROUP = "form-fillers"

    def __init__(self) -> None:
        self._redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    async def publish(self, payload: IntakePayload) -> str:
        raise NotImplementedError("RedisStreamQueue: implement publish()")

    async def consume_one(self) -> Optional[IntakePayload]:
        raise NotImplementedError("RedisStreamQueue: implement consume_one()")

    async def ack(self, message_id: str) -> None:
        raise NotImplementedError("RedisStreamQueue: implement ack()")

    async def nack(self, message_id: str, error: str) -> None:
        raise NotImplementedError("RedisStreamQueue: implement nack()")

    async def list_pending(self) -> List[IntakePayload]:
        raise NotImplementedError("RedisStreamQueue: implement list_pending()")

    async def list_dlq(self) -> List[IntakePayload]:
        raise NotImplementedError("RedisStreamQueue: implement list_dlq()")


# =============================================================================
# Status Callback — POST webhook on completion/failure
# =============================================================================

async def send_status_callback(payload: IntakePayload) -> None:
    """POST status to callback_url if configured."""
    url = (payload.callback_url or "").strip()
    if not url or url.startswith("#"):
        return

    callback_data = {
        "message_id": payload.message_id,
        "call_sid": payload.call_sid,
        "status": payload.status,
        "submitted_at": payload.submitted_at,
        "completed_at": payload.completed_at,
        "errors": payload.errors,
        "attempt": payload.attempt,
    }

    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=callback_data,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                logger.info(
                    "Status callback sent | url=%s status=%s response=%d",
                    url,
                    payload.status,
                    resp.status,
                )
    except ImportError:
        # aiohttp not installed — fall back to httpx or urllib
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json=callback_data)
                logger.info(
                    "Status callback sent (httpx) | url=%s status=%s response=%d",
                    url,
                    payload.status,
                    resp.status_code,
                )
        except ImportError:
            logger.warning(
                "Status callback skipped — install aiohttp or httpx for webhook support"
            )
    except Exception as e:
        logger.error("Status callback failed | url=%s error=%s", url, str(e)[:200])


# =============================================================================
# Factory — Get the configured queue backend
# =============================================================================

_queue_instance: Optional[IntakeQueue] = None


def get_queue() -> IntakeQueue:
    """Return the singleton queue instance (configured via env vars)."""
    global _queue_instance
    if _queue_instance is not None:
        return _queue_instance

    backend = os.getenv("INTAKE_QUEUE_BACKEND", "filesystem").strip().lower()

    if backend == "filesystem":
        base = os.getenv("INTAKE_QUEUE_DIR", "").strip()
        base_path = Path(base) if base and not base.startswith("#") else None
        _queue_instance = FilesystemQueue(base_dir=base_path)
        logger.info("Intake queue initialized | backend=filesystem dir=%s", _queue_instance.base_dir)
    elif backend == "redis":
        _queue_instance = RedisStreamQueue()
        logger.info("Intake queue initialized | backend=redis stream=%s", RedisStreamQueue.STREAM_KEY)
    else:
        raise ValueError(f"Unknown INTAKE_QUEUE_BACKEND: {backend!r}")

    return _queue_instance


async def publish_completed_intake(
    state: Any,
    *,
    call_sid: str = "",
    caller_from: str = "",
    caller_to: str = "",
    voice_field_sources: Optional[dict[str, str]] = None,
    callback_url: str = "",
) -> str:
    """
    Convenience function: build payload from application state and publish.

    Called by update_application_state when all fields are captured.
    Returns the message_id.
    """
    payload = IntakePayload.from_application_state(
        state,
        call_sid=call_sid,
        caller_from=caller_from,
        caller_to=caller_to,
        voice_field_sources=voice_field_sources,
        callback_url=callback_url,
    )
    queue = get_queue()
    msg_id = await queue.publish(payload)
    return msg_id
