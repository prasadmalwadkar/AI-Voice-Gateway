"""Line session client: TTS normalization, deterministic turns, status checks, queue drain."""
import asyncio

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

def test_normalize_agent_tts_pins_offstage_narration_for_phone_ask():
    """Model echoes property/income while asking phone → speak current_ask only."""
    from app.line_session_client import LineSessionClient

    client = LineSessionClient.__new__(LineSessionClient)
    client._last_tool_ask = "Mobile phone?"
    raw = (
        "I have your income at 80k and property on Main Street. Mobile phone?"
    )
    assert LineSessionClient._normalize_agent_text_for_tts(client, raw) == "Mobile phone?"


# ---------------------------------------------------------------------------
# mark_deterministic_turn tests
# ---------------------------------------------------------------------------

async def test_mark_deterministic_turn_resets_flags():
    """mark_deterministic_turn resets per-turn flags and cancels watchdogs."""
    mock_ws = AsyncMock()
    mock_ws.closed = False
    mock_logger = MagicMock(spec=["debug", "info", "warning", "error", "call_sid"])
    mock_logger.call_sid = "test-call"

    # Build a minimal LineSessionClient-like object to test method behavior.
    # We import and instantiate, then inject mock ws.
    from app.line_session_client import LineSessionClient
    client = LineSessionClient.__new__(LineSessionClient)
    client._ws = mock_ws
    client._closed = False
    client._call_logger = mock_logger
    client._fallback_spoken_text = "old fallback"
    client._ask_spoken_this_turn = True
    client._awaiting_turn_response = True
    client._last_tool_ask = "old ask"
    client._last_spoken_ask_text = ""
    client._last_spoken_ask_ts = 0.0
    client._ask_dedup_window_sec = 3.0
    client._post_fallback_mute_model_sec = 1.8
    client._mute_model_text_until = 0.0
    client._debounce_task = None
    client._text_buffer = "stale buffered text"
    client.tts_queue = asyncio.Queue()
    await client.tts_queue.put("old queued ask")

    # Create mock tasks that are "running"
    mock_task = MagicMock()
    mock_task.done.return_value = False
    client._no_response_watchdog_task = mock_task
    client._turn_watchdog_task = mock_task
    client._pending_ask_fallback_task = None

    await client.mark_deterministic_turn("What is your name?")

    # Verify flags reset
    assert client._fallback_spoken_text == ""
    assert client._ask_spoken_this_turn is True  # Set by _enqueue_ask_if_needed
    assert client._awaiting_turn_response is False
    assert client._text_buffer == ""

    # Verify watchdogs cancelled
    assert mock_task.cancel.call_count == 2

    # Verify stale ask was drained and new ask was queued for TTS
    assert not client.tts_queue.empty()
    queued = client.tts_queue.get_nowait()
    assert queued == "What is your name?"

    # Verify no ws frames sent (no send_user_transcript)
    mock_ws.send.assert_not_called()


async def test_mark_deterministic_turn_no_ask():
    """mark_deterministic_turn with ask_next=None still resets flags, no TTS."""
    from app.line_session_client import LineSessionClient
    client = LineSessionClient.__new__(LineSessionClient)
    client._ws = AsyncMock()
    client._ws.closed = False
    client._closed = False
    client._call_logger = MagicMock(spec=["debug", "info", "warning", "error", "call_sid"])
    client._call_logger.call_sid = "test"
    client._fallback_spoken_text = "old"
    client._ask_spoken_this_turn = True
    client._awaiting_turn_response = True
    client._last_tool_ask = "old"
    client._last_spoken_ask_text = ""
    client._last_spoken_ask_ts = 0.0
    client._ask_dedup_window_sec = 3.0
    client._post_fallback_mute_model_sec = 1.8
    client._mute_model_text_until = 0.0
    client._debounce_task = None
    client._text_buffer = ""
    client.tts_queue = asyncio.Queue()
    client._no_response_watchdog_task = None
    client._turn_watchdog_task = None
    client._pending_ask_fallback_task = None

    await client.mark_deterministic_turn(None)

    assert client._ask_spoken_this_turn is False
    assert client._awaiting_turn_response is False
    assert client._last_tool_ask == ""
    assert client.tts_queue.empty()


async def test_send_user_transcript_status_ping_repeats_current_ask_without_llm():
    """Status check utterance should acknowledge presence and skip LLM frames."""
    from app.line_session_client import LineSessionClient

    client = LineSessionClient.__new__(LineSessionClient)
    client._ws = AsyncMock()
    client._ws.closed = False
    client._closed = False
    client._call_logger = MagicMock(spec=["debug", "info", "warning", "error", "call_sid"])
    client._call_logger.call_sid = "test"
    client._fallback_spoken_text = ""
    client._ask_spoken_this_turn = False
    client._awaiting_turn_response = False
    client._last_tool_ask = "Last name?"
    client._last_spoken_ask_text = ""
    client._last_spoken_ask_ts = 0.0
    client._ask_dedup_window_sec = 3.0
    client._post_fallback_mute_model_sec = 1.8
    client._mute_model_text_until = 0.0
    client.tts_queue = asyncio.Queue()
    client._no_response_watchdog_task = None
    client._turn_watchdog_task = None
    client._pending_ask_fallback_task = None
    client._enqueue_ask_if_needed = AsyncMock(return_value=True)

    await client.send_user_transcript("Hello, you there?")

    client._enqueue_ask_if_needed.assert_awaited_once_with(
        "Yes, I'm here. Last name?",
        event="line.session.status_check_response",
    )
    client._ws.send.assert_not_called()


async def test_clear_pending_speech_drains_stale_tts_queue():
    """Barge-in cleanup should remove stale queued asks before next question."""
    from app.line_session_client import LineSessionClient

    client = LineSessionClient.__new__(LineSessionClient)
    client._call_logger = MagicMock(spec=["debug", "info", "warning", "error", "call_sid"])
    client._debounce_task = None
    client._pending_ask_fallback_task = None
    client._text_buffer = "old buffered words"
    client._fallback_spoken_text = "old fallback"
    client.tts_queue = asyncio.Queue()
    await client.tts_queue.put("old ownership question")
    await client.tts_queue.put("old fallback question")

    await client.clear_pending_speech(reason="test")

    assert client._text_buffer == ""
    assert client._fallback_spoken_text == ""
    assert client.tts_queue.empty()


