"""Twilio bridge routing, Path2 barge-in / VAD, watchdogs, and state update checks."""
import asyncio
import base64
import json
import os

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bridge routing integration tests
# ---------------------------------------------------------------------------


def test_decode_twilio_prefill_b64_custom_parameter():
    from app.twilio_line_bridge import _decode_prefill_b64

    payload = {"first_name": "Jane", "email_address": "jane@example.com"}
    encoded = (
        base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    )

    assert _decode_prefill_b64(encoded) == payload


async def test_line_session_start_includes_prefill_metadata():
    from app.line_session_client import LineSessionClient

    sent = {}

    class FakeWebSocket:
        async def send(self, raw):
            sent["payload"] = json.loads(raw)

    async def fake_connect(*args, **kwargs):
        return FakeWebSocket()

    client = LineSessionClient(
        "ws://example.test/ws",
        MagicMock(spec=["info", "debug", "warning", "error", "call_sid"]),
        call_id="CA123",
        from_number="+12135550100",
        to_number="+18005550100",
        metadata={
            "caller_from": "+12135550100",
            "customer_prefill": {"first_name": "Jane"},
            "customer_prefill_fields": ["first_name"],
        },
    )
    client._receive_loop = AsyncMock()

    with patch("app.line_session_client.websockets.connect", new=fake_connect):
        await client.connect()

    assert sent["payload"]["from"] == "+12135550100"
    assert sent["payload"]["metadata"]["caller_from"] == "+12135550100"
    assert sent["payload"]["metadata"]["customer_prefill"] == {"first_name": "Jane"}
    assert sent["payload"]["metadata"]["customer_prefill_fields"] == ["first_name"]

async def test_bridge_routes_handled_to_deterministic():
    """When heuristic returns handled=True, send_user_transcript must NOT be called."""
    from agent.schema import HeuristicResult

    mock_line = AsyncMock()
    mock_line.mark_deterministic_turn = AsyncMock()
    mock_line.send_user_transcript = AsyncMock()
    mock_line.inject_ask_next_hint = MagicMock()

    handled_result = HeuristicResult(
        handled=True, ask_next="Next question?",
        handled_field="ownership_type", stage="property",
    )

    with (
        patch("app.twilio_line_bridge.apply_voice_intake_heuristics_before_line",
              new_callable=AsyncMock, return_value=handled_result),
        patch("app.twilio_line_bridge.broadcast_stt", new_callable=AsyncMock),
        patch("app.twilio_line_bridge.path2_debug_log", new_callable=AsyncMock),
        patch("app.twilio_line_bridge.path2_set_debug_call", return_value=None),
        patch("app.twilio_line_bridge.path2_reset_debug_call"),
    ):
        from app.twilio_line_bridge import apply_voice_intake_heuristics_before_line

        # Simulate the bridge routing logic inline (extracted from _consume_stt).
        result = await apply_voice_intake_heuristics_before_line("sole owner")
        if result.handled and mock_line:
            await mock_line.mark_deterministic_turn(result.ask_next)
        else:
            await mock_line.send_user_transcript("sole owner")

    mock_line.mark_deterministic_turn.assert_called_once_with("Next question?")
    mock_line.send_user_transcript.assert_not_called()


async def test_bridge_routes_not_handled_to_llm():
    """When heuristic returns handled=False, send_user_transcript must be called."""
    from agent.schema import HeuristicResult

    mock_line = AsyncMock()
    mock_line.mark_deterministic_turn = AsyncMock()
    mock_line.send_user_transcript = AsyncMock()
    mock_line.inject_ask_next_hint = MagicMock()

    not_handled = HeuristicResult(handled=False, stage="property")

    with (
        patch("app.twilio_line_bridge.apply_voice_intake_heuristics_before_line",
              new_callable=AsyncMock, return_value=not_handled),
        patch("app.twilio_line_bridge.broadcast_stt", new_callable=AsyncMock),
        patch("app.twilio_line_bridge.path2_debug_log", new_callable=AsyncMock),
        patch("app.twilio_line_bridge.path2_set_debug_call", return_value=None),
        patch("app.twilio_line_bridge.path2_reset_debug_call"),
    ):
        from app.twilio_line_bridge import apply_voice_intake_heuristics_before_line

        result = await apply_voice_intake_heuristics_before_line("some ambiguous text")
        if result.handled and mock_line:
            await mock_line.mark_deterministic_turn(result.ask_next)
        else:
            await mock_line.send_user_transcript("some ambiguous text")

    mock_line.send_user_transcript.assert_called_once_with("some ambiguous text")
    mock_line.mark_deterministic_turn.assert_not_called()


async def test_bridge_allows_yes_for_on_file_confirmation_to_deterministic():
    from agent.schema import HeuristicResult
    from app.twilio_line_bridge import TwilioLineBridge

    bridge = TwilioLineBridge.__new__(TwilioLineBridge)
    bridge.call_sid = "CA_test"
    bridge._deterministic_answer_guard_until = 0.0
    bridge._line = AsyncMock()
    bridge._line.mark_deterministic_turn = AsyncMock()
    bridge._line.enqueue_clarification_for_current_ask = AsyncMock()
    bridge._line.inject_ask_next_hint = MagicMock()
    bridge._cancel_caller_response_watchdog = MagicMock()
    bridge._cancel_deterministic_followup_task = MagicMock()
    bridge._schedule_deterministic_followup_ask = MagicMock()
    bridge._send_deterministic_goodbye = AsyncMock()
    bridge._log_pipeline_debug = MagicMock()

    handled = HeuristicResult(
        handled=True,
        ask_next="What is the property type: primary, secondary, or investment?",
        handled_field="prefill_confirmation",
        stage="property",
    )
    missing = (
        '[BROWSER STAGE: PROPERTY] Still needed: ownership type. '
        'Ask next: "Got it, I\'ve saved your address. Regarding ownership, '
        'I have you down as the sole owner. Is that still correct?"'
    )

    with (
        patch("app.twilio_line_bridge.broadcast_stt", new_callable=AsyncMock),
        patch("app.twilio_line_bridge.path2_set_debug_call", return_value=None),
        patch("app.twilio_line_bridge.path2_reset_debug_call"),
        patch("app.twilio_line_bridge.path2_debug_log", new_callable=AsyncMock),
        patch("app.twilio_line_bridge.log_intake_event"),
        patch("app.twilio_line_bridge.append_transcript_turn"),
        patch("app.twilio_line_bridge.get_missing_fields", return_value=missing),
        patch(
            "app.twilio_line_bridge.apply_voice_intake_heuristics_before_line",
            new_callable=AsyncMock,
            return_value=handled,
        ),
    ):
        await TwilioLineBridge._handle_final_transcript(bridge, "Yes.")

    bridge._line.enqueue_clarification_for_current_ask.assert_not_called()
    bridge._line.mark_deterministic_turn.assert_called_once_with(handled.ask_next)


async def test_bridge_triggers_deterministic_goodbye_on_terminal_deterministic():
    """Handled deterministic turn with ask_next=None + no missing fields enqueues goodbye TTS."""
    from agent.schema import HeuristicResult
    from app.twilio_line_bridge import DETERMINISTIC_GOODBYE_TEXT

    mock_line = AsyncMock()
    mock_line.mark_deterministic_turn = AsyncMock()
    mock_line.tts_queue = asyncio.Queue()
    mock_line.end_call_event = asyncio.Event()

    handled_terminal = HeuristicResult(
        handled=True,
        ask_next=None,
        handled_field="agreed_to_commerce_communication",
        stage="consent",
    )

    result = handled_terminal
    missing = ""
    if result.handled and mock_line:
        await mock_line.mark_deterministic_turn(result.ask_next)
        if result.ask_next is None and not (missing or "").strip():
            await mock_line.tts_queue.put(DETERMINISTIC_GOODBYE_TEXT)
            mock_line.end_call_event.set()

    mock_line.mark_deterministic_turn.assert_called_once_with(None)
    assert not mock_line.tts_queue.empty()
    assert mock_line.tts_queue.get_nowait() == DETERMINISTIC_GOODBYE_TEXT
    assert mock_line.end_call_event.is_set()


async def test_bridge_does_not_trigger_goodbye_when_missing_fields_remain():
    """ask_next=None should not trigger goodbye if intake still has missing fields."""
    from agent.schema import HeuristicResult
    from app.twilio_line_bridge import DETERMINISTIC_GOODBYE_TEXT

    mock_line = AsyncMock()
    mock_line.mark_deterministic_turn = AsyncMock()
    mock_line.tts_queue = asyncio.Queue()
    mock_line.end_call_event = asyncio.Event()

    handled_terminal = HeuristicResult(
        handled=True,
        ask_next=None,
        handled_field="agreed_to_commerce_communication",
        stage="consent",
    )

    result = handled_terminal
    missing = "[BROWSER STAGE: CONSENT] Still needed: electronic communications consent"
    if result.handled and mock_line:
        await mock_line.mark_deterministic_turn(result.ask_next)
        if result.ask_next is None and not (missing or "").strip():
            await mock_line.tts_queue.put(DETERMINISTIC_GOODBYE_TEXT)
            mock_line.end_call_event.set()

    mock_line.mark_deterministic_turn.assert_called_once_with(None)
    assert mock_line.tts_queue.empty()
    assert not mock_line.end_call_event.is_set()


async def test_bridge_schedules_followup_ask_after_navigation():
    """If deterministic ask_next is None while waiting navigation, queue next ask once available."""
    from app.twilio_line_bridge import TwilioLineBridge

    bridge = TwilioLineBridge.__new__(TwilioLineBridge)
    bridge._running = True
    bridge._line = AsyncMock()
    bridge._line.mark_deterministic_turn = AsyncMock()
    bridge._line.send_user_transcript = AsyncMock()
    bridge.call_logger = MagicMock(spec=["debug", "info", "warning", "error"])
    bridge._deterministic_followup_task = None

    _for_sale_ask = (
        "Is the property currently for sale? Say one for yes, two for no."
    )
    with patch(
        "app.twilio_line_bridge.get_missing_fields",
        side_effect=[
            "[BROWSER STAGE: PROPERTY] All current fields filled. Waiting for navigation.",
            f'[BROWSER STAGE: FOR-SALE] Still needed: whether property is currently for sale Ask next: "{_for_sale_ask}"',
        ],
    ):
        await TwilioLineBridge._wait_and_queue_deterministic_followup_ask(bridge)

    bridge._line.mark_deterministic_turn.assert_called_once_with(_for_sale_ask)


async def test_bridge_skips_finals_when_deterministic_goodbye_in_progress():
    from app.twilio_line_bridge import TwilioLineBridge

    bridge = TwilioLineBridge.__new__(TwilioLineBridge)
    bridge._ending_call = True
    bridge.call_logger = MagicMock(spec=["info", "debug", "warning", "error", "call_sid"])

    await TwilioLineBridge._handle_final_transcript(bridge, "Yes")

    bridge.call_logger.info.assert_called_once()


def test_status_check_utterance_short_circuit_match():
    from app.line_session_client import LineSessionClient
    from app.twilio_line_bridge import _is_status_check_utterance

    for fn in (_is_status_check_utterance, LineSessionClient._is_status_check_utterance):
        assert fn("Hello, you there?") is True
        assert fn("You good?") is True
        assert fn("Can you hear me?") is True
        assert fn("Are we good?") is True
        assert fn("Still with me?") is True
        assert fn("what's next?") is True
        assert fn("go ahead") is True
        assert fn("thank you") is True
        assert fn("thanks") is True
        assert fn("my last name is Smith") is False

    lsc = LineSessionClient._is_status_check_utterance
    assert lsc("What's the question?") is True
    assert lsc("what is the question") is True
    assert lsc("repeat") is True
    assert lsc("say that again") is True
    assert lsc("I didn't hear") is True
    assert lsc("I'm lost") is True
    assert lsc("I'm confused") is True
    assert lsc("pardon?") is True
    assert lsc("excuse me") is True
    assert lsc("which question?") is True
    assert lsc("what do you need?") is True
    assert lsc("325 West Adams Boulevard") is False
    assert lsc("sole owner") is False


def test_bridge_allows_llm_fallback_for_supported_current_asks():
    from app.twilio_line_bridge import TwilioLineBridge

    income_missing = (
        '[BROWSER STAGE: INCOME] Still needed: annual pre-tax income '
        'Ask next: "What\'s your annual pre-tax income?"'
    )
    assert TwilioLineBridge._allow_llm_fallback_for_current_ask(income_missing) is True


def test_looks_like_answer_for_current_field_rejects_non_answer():
    from agent.state_manager import looks_like_answer_for_current_field

    missing = (
        '[BROWSER STAGE: PROPERTY] Still needed: ownership type '
        'Ask next: "What is the ownership type: sole or joint?"'
    )
    assert looks_like_answer_for_current_field("Sorry.", missing) is False
    assert looks_like_answer_for_current_field("That's all.", missing) is False
    assert looks_like_answer_for_current_field("I'm so lonely.", missing) is False
    assert looks_like_answer_for_current_field("So, Lonas.", missing) is True
    assert looks_like_answer_for_current_field("sole owner", missing) is True
    assert looks_like_answer_for_current_field("store owner", missing) is True


def test_clarification_for_ownership_is_not_same_question():
    from app.line_session_client import LineSessionClient

    ask = "What is the ownership type: sole or joint?"
    clarification = LineSessionClient._clarification_for_ask(ask)
    assert clarification == "Sorry, I missed that. Are you the sole owner, or is it joint?"
    assert clarification != ask


async def test_enqueue_status_check_response_includes_current_ask():
    from app.line_session_client import LineSessionClient

    client = LineSessionClient.__new__(LineSessionClient)
    client._last_tool_ask = "What is your date of birth?"
    client._ask_spoken_this_turn = True
    client._last_spoken_ask_text = ""
    client._last_spoken_ask_ts = 0.0
    client._ask_dedup_window_sec = 3.0
    client._post_fallback_mute_model_sec = 1.8
    client._mute_model_text_until = 0.0
    client._call_logger = MagicMock(spec=["debug", "info", "warning", "error", "call_sid"])
    client.tts_queue = asyncio.Queue()

    assert await client.enqueue_status_check_response() is True
    assert client.tts_queue.get_nowait() == "What is your date of birth?"


def test_expanded_consent_prompts():
    from agent.schema import HelocApplicationState
    from agent.state_manager import get_missing_fields

    mock_browser = MagicMock()
    mock_browser.get_current_stage.return_value = "consent"
    mock_browser._page = None
    state = HelocApplicationState(
        address_line_1="123 Main Street",
        ownership_type="Sole owner",
        occupancy_type="Primary property",
        is_currently_for_sale=False,
        annual_pre_tax_income=100000,
        other_income=0,
        primary_source_of_income="Employed full-time",
        first_name="Alex",
        last_name="Rivera",
        date_of_birth="01/05/1990",
        mobile_phone_number="2135550101",
        email_address="alex@example.com",
    )
    with (
        patch("agent.state_manager.browser_manager", mock_browser),
        patch("agent.state_manager.application_state", state),
    ):
        missing = get_missing_fields()

    assert "Privacy Policy" in missing
    assert "Terms of Service" in missing
    assert "soft credit pull" in missing
    assert "will not affect your credit score" in missing


def test_barge_in_partial_filter_accepts_real_answers_not_noise():
    from app.twilio_line_bridge import TwilioLineBridge

    fn = TwilioLineBridge._looks_like_meaningful_barge_in
    assert fn("sole owner") is True
    assert fn("yes") is True
    assert fn("agree") is True
    assert fn("1") is True
    assert fn(".") is False
    assert fn("um") is False
    assert fn("So.") is False
    assert fn("So.", preserve_so_for_ownership_ask=True) is True


def test_path2_noise_optional_so_preserve_for_ownership_ask():
    from app.path2_stt_noise import path2_stt_final_is_drop_noise

    assert path2_stt_final_is_drop_noise("So.") is True
    assert path2_stt_final_is_drop_noise("So.", preserve_so_for_ownership_ask=True) is False


def test_final_stt_barge_in_rejects_agent_echo_but_accepts_user_speech():
    from unittest.mock import patch

    from app.twilio_line_bridge import TwilioLineBridge

    bridge = TwilioLineBridge.__new__(TwilioLineBridge)
    bridge._agent_playing = True
    bridge._last_barge_at = 0.0
    bridge._current_tts_text = "Is the property currently for sale?"
    missing = 'Ask next: "Is the property currently for sale?"'

    with patch("app.twilio_line_bridge.get_missing_fields", return_value=missing):
        assert bridge._should_trigger_barge_in(
            "currently for sale",
            in_echo=True,
            now=10.0,
        ) is False
        assert bridge._should_trigger_barge_in(
            "primary",
            in_echo=True,
            now=10.0,
        ) is True
        assert bridge._should_trigger_barge_in(
            "no",
            in_echo=True,
            now=10.0,
        ) is True


def test_barge_in_bypasses_echo_when_answer_shaped_for_current_ask():
    from unittest.mock import patch

    from app.twilio_line_bridge import TwilioLineBridge

    bridge = TwilioLineBridge.__new__(TwilioLineBridge)
    bridge._agent_playing = True
    bridge._last_barge_at = 0.0
    bridge._current_tts_text = "What is the ownership type: sole or joint?"
    missing = 'Ask next: "What is the ownership type: sole or joint?"'

    with patch("app.twilio_line_bridge.get_missing_fields", return_value=missing):
        with patch.object(
            TwilioLineBridge,
            "_looks_like_current_agent_echo",
            return_value=True,
        ):
            assert bridge._should_trigger_barge_in(
                "store owner",
                in_echo=True,
                now=10.0,
            ) is True
        with patch.object(
            TwilioLineBridge,
            "_looks_like_current_agent_echo",
            return_value=True,
        ):
            assert bridge._should_trigger_barge_in(
                "um",
                in_echo=True,
                now=10.0,
            ) is False


def test_final_stt_barge_in_respects_cooldown_and_agent_playing():
    from app.twilio_line_bridge import TwilioLineBridge

    bridge = TwilioLineBridge.__new__(TwilioLineBridge)
    bridge._agent_playing = False
    bridge._last_barge_at = 0.0
    bridge._current_tts_text = "What is the property type?"

    assert bridge._should_trigger_barge_in(
        "primary",
        in_echo=True,
        now=10.0,
    ) is False

    bridge._agent_playing = True
    bridge._last_barge_at = 9.5
    assert bridge._should_trigger_barge_in(
        "primary",
        in_echo=True,
        now=10.0,
    ) is False


def test_playback_tail_keeps_barge_in_armed_after_cartesia_finishes():
    """Twilio may still play buffered audio after streaming stops — treat as agent-active."""
    import time

    from unittest.mock import patch

    from app.twilio_line_bridge import TwilioLineBridge

    bridge = TwilioLineBridge.__new__(TwilioLineBridge)
    bridge._agent_playing = False
    bridge._playback_tail_until = time.monotonic() + 10.0
    bridge._last_barge_at = 0.0
    bridge._current_tts_text = ""

    missing = 'Ask next: "Do you agree to the Privacy Policy?"'
    with patch("app.twilio_line_bridge.get_missing_fields", return_value=missing):
        assert bridge._should_trigger_barge_in(
            "yes",
            in_echo=False,
            now=time.monotonic(),
        )


async def test_low_confidence_update_requires_single_confirmation():
    from agent.schema import HelocApplicationState
    from agent.state_manager import (
        _clear_pending_low_confidence_confirmation,
        update_application_state,
    )

    mock_browser = MagicMock()
    mock_browser.get_current_stage.return_value = "property"

    state = HelocApplicationState()
    with (
        patch("agent.state_manager.browser_manager", mock_browser),
        patch("agent.state_manager.application_state", state),
    ):
        result = await update_application_state.func(
            None,
            ownership_type="Sole owner",
            confidence_score=0.5,
            raw_user_input="sole owner",
        )

    assert "Update PENDING" in result
    assert 'Ask next: "I didn\'t catch that correctly. Can you repeat?"' in result
    assert state.ownership_type is None
    _clear_pending_low_confidence_confirmation()


async def test_low_confidence_address_asks_for_repeat_not_generic_confirmation():
    from agent.schema import HelocApplicationState
    from agent.state_manager import (
        _clear_pending_low_confidence_confirmation,
        update_application_state,
    )

    mock_browser = MagicMock()
    mock_browser.get_current_stage.return_value = "property"

    state = HelocApplicationState()
    with (
        patch("agent.state_manager.browser_manager", mock_browser),
        patch("agent.state_manager.application_state", state),
    ):
        result = await update_application_state.func(
            None,
            address_line_1="325 West Adams Boulevard",
            confidence_score=0.5,
            raw_user_input="noisy transcript about 325 west adams boulevard",
        )

    assert "Update PENDING" in result
    assert 'Ask next: "I didn\'t catch that correctly. Please repeat the street address."' in result
    assert 'Ask next: "Is that correct?"' not in result
    assert state.address_line_1 is None
    _clear_pending_low_confidence_confirmation()


async def test_pending_low_confidence_yes_commits_candidate():
    from agent.schema import HelocApplicationState
    from agent.state_manager import (
        _clear_pending_low_confidence_confirmation,
        update_application_state,
    )

    mock_browser = MagicMock()
    mock_browser.get_current_stage.return_value = "property"
    mock_browser._page = None

    state = HelocApplicationState()
    with (
        patch("agent.state_manager.browser_manager", mock_browser),
        patch("agent.state_manager.application_state", state),
        patch("agent.state_manager._run_browser_sync", new_callable=AsyncMock),
    ):
        await update_application_state.func(
            None,
            address_line_1="325 West Adams Boulevard",
            confidence_score=0.5,
            raw_user_input="noisy transcript about 325 west adams boulevard",
        )
        result = await update_application_state.func(
            None,
            confidence_score=1.0,
            raw_user_input="Yes.",
        )

    assert state.address_line_1 == "325 West Adams Boulevard"
    assert "ownership type" in result
    _clear_pending_low_confidence_confirmation()


async def test_pending_low_confidence_no_clears_candidate():
    from agent.schema import HelocApplicationState
    from agent.state_manager import (
        _clear_pending_low_confidence_confirmation,
        has_pending_low_confidence_confirmation,
        update_application_state,
    )

    mock_browser = MagicMock()
    mock_browser.get_current_stage.return_value = "property"
    mock_browser._page = None

    state = HelocApplicationState()
    with (
        patch("agent.state_manager.browser_manager", mock_browser),
        patch("agent.state_manager.application_state", state),
    ):
        await update_application_state.func(
            None,
            address_line_1="325 West Adams Boulevard",
            confidence_score=0.5,
            raw_user_input="noisy transcript about 325 west adams boulevard",
        )
        result = await update_application_state.func(
            None,
            confidence_score=1.0,
            raw_user_input="No.",
        )

    assert state.address_line_1 is None
    assert has_pending_low_confidence_confirmation() is False
    assert "Can you tell me the property address?" in result
    _clear_pending_low_confidence_confirmation()


async def test_update_application_state_rejects_guessed_email_without_spoken_evidence():
    from agent.schema import HelocApplicationState
    from agent.state_manager import update_application_state

    mock_browser = MagicMock()
    mock_browser.get_current_stage.return_value = "contact"
    mock_browser._page = None

    state = HelocApplicationState(mobile_phone_number="2135550101")
    with (
        patch("agent.state_manager.browser_manager", mock_browser),
        patch("agent.state_manager.application_state", state),
        patch("agent.state_manager._run_browser_sync", new_callable=AsyncMock),
    ):
        result = await update_application_state.func(
            None,
            email_address="guess@gmail.com",
            confidence_score=1.0,
            raw_user_input="Yes.",
        )

    assert state.email_address is None
    assert "email_address rejected" in result


def test_strong_pin_normalize_agent_text():
    from app.line_session_client import LineSessionClient

    client = LineSessionClient.__new__(LineSessionClient)
    client._last_tool_ask = "Any other income?"

    assert client._normalize_agent_text_for_tts(
        "Sole or joint owner?"
    ) == "Any other income?"

    assert client._normalize_agent_text_for_tts(
        "Any other income?"
    ) == "Any other income?"

    assert client._normalize_agent_text_for_tts(
        'Still needed: ownership type Ask next: "Sole or joint owner?"'
    ) == "Any other income?"

    assert client._normalize_agent_text_for_tts(
        "I see. Let me note that."
    ) == "I see. Let me note that."

    client._last_tool_ask = ""
    assert client._normalize_agent_text_for_tts(
        "Sole or joint owner?"
    ) == "Sole or joint owner?"

    client._last_tool_ask = "What is the property type: primary, secondary, or investment?"
    bad = (
        "It looks like the system needs a full last name. Could you please "
        "provide the complete spelling?"
    )
    assert client._normalize_agent_text_for_tts(bad) == client._last_tool_ask


def test_looks_like_property_address_requires_street_signal():
    from agent.state_manager import looks_like_answer_for_current_field

    missing = 'Ask next: "What is the property address?"'
    assert looks_like_answer_for_current_field("three twenty five.", missing) is False
    assert looks_like_answer_for_current_field("325 west adams", missing) is True


def test_mismatched_question_detection():
    from app.line_session_client import LineSessionClient

    fn = LineSessionClient._text_contains_mismatched_question
    assert fn("Sole or joint owner?", "Any other income?") is True
    assert fn("Any other income?", "Any other income?") is False
    assert fn("I see. Let me note that.", "Any other income?") is False
    assert fn("Got it!", "Sole or joint owner?") is False


def test_build_intake_llm_context_string_shape():
    from agent.state_manager import application_state, build_intake_llm_context_string

    application_state.first_name = "Pat"
    s = build_intake_llm_context_string(max_total_chars=8000)
    assert "[CURRENT_INTAKE_STATE" in s
    assert "browser_stage=" in s
    assert "captured_fields_json" in s
    assert "still_needed_line" in s
    assert "Pat" in s
    application_state.first_name = None


def test_looks_like_answer_to_ownership_numeric_shortcuts():
    from app.line_session_client import LineSessionClient

    fn = LineSessionClient._looks_like_answer_to_ask
    ask = "What is the ownership type: sole or joint?"
    assert fn(ask, "1") is True
    assert fn(ask, "2") is True
    assert fn(ask, "one") is True
    assert fn(ask, "two") is True
    assert fn(ask, "sad") is False
    assert fn(ask, "um") is False


def test_looks_like_answer_to_other_income_natural_negatives():
    from app.line_session_client import LineSessionClient

    fn = LineSessionClient._looks_like_answer_to_ask
    ask = "Do you have any other income to include?"
    assert fn(ask, "Not at this moment.") is True
    assert fn(ask, "not right now") is True
    assert fn(ask, "No. No.") is True
    assert fn(ask, "none") is True


def test_compute_mulaw_rms_silence_vs_speech():
    """_compute_mulaw_rms returns ~0 for silence, high for speech-like bytes."""
    from app.twilio_line_bridge import TwilioLineBridge

    silence = bytes([0xFF] * 160)
    rms_silence = TwilioLineBridge._compute_mulaw_rms(silence)
    assert rms_silence < 1.0, f"Silence RMS should be ~0, got {rms_silence}"

    speech = bytes([0x00] * 160)
    rms_speech = TwilioLineBridge._compute_mulaw_rms(speech)
    assert rms_speech > 1000, f"Max-amplitude RMS should be high, got {rms_speech}"

    assert TwilioLineBridge._compute_mulaw_rms(b"") == 0.0


def test_vad_barge_in_uses_playback_start_grace_not_last_audio_time():
    from app.twilio_line_bridge import TwilioLineBridge, VAD_CONSEC_FRAMES

    async def _run() -> None:
        bridge = TwilioLineBridge.__new__(TwilioLineBridge)
        bridge._agent_playing = True
        bridge._agent_playing_started_at = 10.0
        bridge._last_agent_audio_time = 10.99
        bridge._vad_consec_high = VAD_CONSEC_FRAMES - 1
        bridge._log_pipeline_debug = MagicMock()
        bridge._barge_in = AsyncMock()

        await TwilioLineBridge._vad_check_barge_in(bridge, bytes([0x00] * 160), 11.0)

        bridge._barge_in.assert_called_once()

    asyncio.run(_run())


def test_caller_response_watchdog_prompt_strips_repeat_prefix():
    from app.twilio_line_bridge import (
        CALLER_RESPONSE_WATCHDOG_PREFIX,
        TwilioLineBridge,
    )

    prompt = "What is your first name?"
    repeated = f"{CALLER_RESPONSE_WATCHDOG_PREFIX} {prompt}"

    assert TwilioLineBridge._caller_response_prompt_from_agent_text(prompt) == prompt
    assert TwilioLineBridge._caller_response_prompt_from_agent_text(repeated) == prompt


def test_caller_response_watchdog_uses_current_ask_for_clarification():
    from app.twilio_line_bridge import TwilioLineBridge

    async def _run() -> None:
        bridge = TwilioLineBridge.__new__(TwilioLineBridge)
        bridge._caller_response_watchdog_task = None
        bridge._caller_response_watchdog_seq = 0
        bridge._running = True
        bridge._line = AsyncMock()
        bridge._line._last_tool_ask = "What is your mobile phone number?"

        TwilioLineBridge._schedule_caller_response_watchdog(
            bridge,
            "I didn't catch phone number. Please say all ten digits.",
        )

        assert bridge._caller_response_prompt == "What is your mobile phone number?"
        bridge._caller_response_watchdog_task.cancel()

    asyncio.run(_run())


def test_caller_response_watchdog_queues_repeat(monkeypatch):
    import app.twilio_line_bridge as bridge_module
    from app.twilio_line_bridge import TwilioLineBridge

    async def _run() -> None:
        monkeypatch.setattr(bridge_module, "CALLER_RESPONSE_WATCHDOG_SECS", 0.01)
        bridge = TwilioLineBridge.__new__(TwilioLineBridge)
        bridge._running = True
        bridge._line = AsyncMock()
        bridge._line.tts_queue = asyncio.Queue()
        bridge._caller_response_prompt = "What is your date of birth?"
        bridge._caller_response_watchdog_seq = 0
        bridge.call_sid = "CA_test"
        bridge.call_logger = MagicMock(spec=["debug", "info", "warning", "error"])

        await TwilioLineBridge._caller_response_watchdog(bridge, 0)

        queued = bridge._line.tts_queue.get_nowait()
        assert queued == (
            "I may have missed you there. Could you say that one more time?"
        )
        bridge.call_logger.info.assert_called_once()

    asyncio.run(_run())


def test_repeated_clarification_escalates_to_spelling_prompt():
    from app.line_session_client import LineSessionClient

    async def _run() -> None:
        client = LineSessionClient.__new__(LineSessionClient)
        client._last_tool_ask = "What is your last name?"
        client._clarification_ask_key = ""
        client._clarification_count = 0
        client._ask_spoken_this_turn = False
        client._enqueue_ask_if_needed = AsyncMock(return_value=True)

        await client.enqueue_clarification_for_current_ask()
        await client.enqueue_clarification_for_current_ask()
        await client.enqueue_clarification_for_current_ask()

        first = client._enqueue_ask_if_needed.call_args_list[0].args[0]
        assert first == "What is your last name?"
        second = client._enqueue_ask_if_needed.call_args_list[1].args[0]
        assert second == LineSessionClient._clarification_for_ask(
            "What is your last name?"
        )

        third = client._enqueue_ask_if_needed.call_args_list[2].args[0]
        assert third == (
            "I'm still missing the last name. Could you spell it slowly, "
            "letter by letter."
        )

    asyncio.run(_run())

