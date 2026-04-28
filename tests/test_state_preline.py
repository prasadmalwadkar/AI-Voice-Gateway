"""Pre-line deterministic intake (``apply_voice_intake_heuristics_before_line``)."""
import os

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tests.preline_fixtures import (
    FOR_SALE_ASK_NEXT,
    OCCUPANCY_ASK_NEXT,
    _make_tool_result,
)


async def test_preline_income_garble_passes_low_confidence_for_confirmation():
    """Brittle STT income repairs must not auto-save at nominal high confidence."""
    from agent.schema import HelocApplicationState

    mock_state = HelocApplicationState()
    mock_browser = MagicMock()
    mock_browser.get_current_stage.return_value = "income"
    tool_return = _make_tool_result("Other income?")

    with (
        patch("agent.state_manager.browser_manager", mock_browser),
        patch("agent.state_manager.application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        mock_tool.func = AsyncMock(return_value=tool_return)
        from agent.state_manager import (
            _INCOME_AMBIGUOUS_STT_CONFIDENCE,
            apply_voice_intake_heuristics_before_line,
        )

        await apply_voice_intake_heuristics_before_line("one one five million")

    mock_tool.func.assert_called_once()
    call_kw = mock_tool.func.call_args[1]
    assert call_kw["annual_pre_tax_income"] == 1_500_000
    assert call_kw["confidence_score"] == _INCOME_AMBIGUOUS_STT_CONFIDENCE
@pytest.mark.parametrize(
    "stage,transcript,field,expected_ask,state_kwargs",
    [
        ("property", "sole owner", "ownership_type", OCCUPANCY_ASK_NEXT, {}),
        ("property", "joint", "ownership_type", OCCUPANCY_ASK_NEXT, {}),
        (
            "property",
            "3",
            "occupancy_type",
            FOR_SALE_ASK_NEXT,
            {"address_line_1": "123 Main St", "ownership_type": "Sole owner"},
        ),
        ("for-sale", "no", "is_currently_for_sale", "Annual pre-tax income?", {}),
        ("for-sale", "1", "is_currently_for_sale", "Annual pre-tax income?", {}),
        (
            "consent",
            "yes",
            "agreed_to_privacy_policy",
            "Agree to electronic communications?",
            {},
        ),
        ("name", "Alex", "first_name", "Last name?", {}),
        ("income", "80K", "annual_pre_tax_income", "Other income?", {}),
        (
            "income",
            "3",
            "primary_source_of_income",
            "First name?",
            {"annual_pre_tax_income": 80000, "other_income": 0},
        ),
        ("birthday", "01/05/1990", "date_of_birth", "Phone number?", {}),
        ("contact", "2135550101", "mobile_phone_number", "Email address?", {}),
    ],
)
async def test_preline_heuristic_handled(stage, transcript, field, expected_ask, state_kwargs):
    """Each deterministic field returns HeuristicResult(handled=True) with correct field."""
    from agent.schema import HeuristicResult, HelocApplicationState

    mock_state = HelocApplicationState(**state_kwargs)
    mock_browser = MagicMock()
    mock_browser.get_current_stage.return_value = stage

    tool_return = _make_tool_result(expected_ask)

    with (
        patch("agent.state_manager.browser_manager", mock_browser),
        patch("agent.state_manager.application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        mock_tool.func = AsyncMock(return_value=tool_return)
        from agent.state_manager import apply_voice_intake_heuristics_before_line
        result = await apply_voice_intake_heuristics_before_line(transcript)

    assert isinstance(result, HeuristicResult)
    assert result.handled is True
    assert result.handled_field == field
    assert result.stage == stage
    assert result.ask_next == expected_ask


async def test_preline_heuristic_not_handled_ambiguous():
    """Ambiguous input returns handled=False."""
    from agent.schema import HeuristicResult, HelocApplicationState

    mock_state = HelocApplicationState()
    mock_browser = MagicMock()
    mock_browser.get_current_stage.return_value = "property"

    with (
        patch("agent.state_manager.browser_manager", mock_browser),
        patch("agent.state_manager.application_state", mock_state),
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        from agent.state_manager import apply_voice_intake_heuristics_before_line
        result = await apply_voice_intake_heuristics_before_line("I'm not sure about that")

    assert isinstance(result, HeuristicResult)
    assert result.handled is False


def test_transcript_requests_intake_default_phrases():
    from agent.state_manager import transcript_requests_intake_default

    assert transcript_requests_intake_default("I don't know.") is True
    assert transcript_requests_intake_default("I have no idea.") is True
    assert transcript_requests_intake_default("I can't remember.") is True
    assert transcript_requests_intake_default("I'm not sure.") is True
    assert transcript_requests_intake_default("I am not sure") is True
    assert transcript_requests_intake_default("I'm not sure about that") is False
    assert transcript_requests_intake_default("sole owner") is False


async def test_preline_idk_falls_back_sole_ownership():
    """Decline phrase on ownership step → Sole owner via tool, raw omitted."""
    from agent.schema import HeuristicResult, HelocApplicationState
    from agent.state_manager import (
        DEFAULT_INTAKE_OWNERSHIP,
        apply_voice_intake_heuristics_before_line,
    )

    mock_state = HelocApplicationState(
        address_line_1="123 Main St",
    )
    mock_browser = MagicMock()
    mock_browser.get_current_stage.return_value = "property"
    tool_return = _make_tool_result(OCCUPANCY_ASK_NEXT)

    with (
        patch("agent.state_manager.browser_manager", mock_browser),
        patch("agent.state_manager.application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        mock_tool.func = AsyncMock(return_value=tool_return)
        result = await apply_voice_intake_heuristics_before_line("I don't know.")

    assert result.handled is True
    assert result.handled_field == "ownership_type"
    mock_tool.func.assert_called_once()
    call_kw = mock_tool.func.call_args[1]
    assert call_kw["ownership_type"] == DEFAULT_INTAKE_OWNERSHIP
    assert call_kw["raw_user_input"] is None
    assert call_kw["confidence_score"] == 0.95
    assert result.ask_next == (
        "I'm assuming sole owner as ownership type for now. "
        "What is the property type: primary, secondary, or investment?"
    )


async def test_preline_heuristic_env_killswitch():
    """PATH2_ENABLE_PRELINE_HEURISTICS=0 forces handled=False."""
    from agent.schema import HeuristicResult

    with patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "0"}):
        from agent.state_manager import apply_voice_intake_heuristics_before_line
        result = await apply_voice_intake_heuristics_before_line("sole owner")

    assert isinstance(result, HeuristicResult)
    assert result.handled is False


async def test_preline_heuristic_empty_transcript():
    """Empty transcript returns handled=False."""
    from agent.schema import HeuristicResult

    with patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}):
        from agent.state_manager import apply_voice_intake_heuristics_before_line
        result = await apply_voice_intake_heuristics_before_line("")

    assert result.handled is False


async def test_preline_partial_phone_reprompt_skips_tool():
    """5–9 digits in contact stage → deterministic reprompt, no state update."""
    from agent.schema import HeuristicResult, HelocApplicationState
    from agent.state_manager import (
        PARTIAL_MOBILE_PHONE_REPROMPT,
        apply_voice_intake_heuristics_before_line,
    )

    mock_state = HelocApplicationState()
    mock_browser = MagicMock()
    mock_browser.get_current_stage.return_value = "contact"

    with (
        patch("agent.state_manager.browser_manager", mock_browser),
        patch("agent.state_manager.application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        mock_tool.func = AsyncMock()
        result = await apply_voice_intake_heuristics_before_line("555-1234")

    assert isinstance(result, HeuristicResult)
    assert result.handled is True
    assert result.handled_field == "mobile_phone_number"
    assert result.stage == "contact"
    assert result.ask_next == PARTIAL_MOBILE_PHONE_REPROMPT
    mock_tool.func.assert_not_called()


def test_normalize_agent_tts_pins_offstage_narration_for_phone_ask():
    """Model echoes property/income while asking phone → speak current_ask only."""
    from app.line_session_client import LineSessionClient

    client = LineSessionClient.__new__(LineSessionClient)
    client._last_tool_ask = "Mobile phone?"
    raw = (
        "I have your income at 80k and property on Main Street. Mobile phone?"
    )
    assert LineSessionClient._normalize_agent_text_for_tts(client, raw) == "Mobile phone?"


async def test_preline_heuristic_field_already_filled():
    """If field is already set, returns handled=False (no double-save)."""
    from agent.schema import HeuristicResult, HelocApplicationState

    mock_state = HelocApplicationState(ownership_type="Sole owner", occupancy_type="Primary property")
    mock_browser = MagicMock()
    mock_browser.get_current_stage.return_value = "property"

    with (
        patch("agent.state_manager.browser_manager", mock_browser),
        patch("agent.state_manager.application_state", mock_state),
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        from agent.state_manager import apply_voice_intake_heuristics_before_line
        result = await apply_voice_intake_heuristics_before_line("sole owner")

    assert result.handled is False
async def test_gate_order_idk_reaches_default_not_clarification():
    """'I don't know' during ownership ask must bypass non-answer gate and reach heuristic defaults."""
    from agent.schema import HeuristicResult, HelocApplicationState
    from agent.state_manager import (
        DEFAULT_INTAKE_OWNERSHIP,
        apply_voice_intake_heuristics_before_line,
    )

    mock_state = HelocApplicationState(address_line_1="123 Main St")
    mock_browser = MagicMock()
    mock_browser.get_current_stage.return_value = "property"
    mock_browser._page = None

    tool_return = _make_tool_result(
        "What is the property type: primary, secondary, or investment?"
    )

    with (
        patch("agent.state_manager.browser_manager", mock_browser),
        patch("agent.state_manager.application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        mock_tool.func = AsyncMock(return_value=tool_return)
        result = await apply_voice_intake_heuristics_before_line("I don't know")

    assert result.handled is True
    assert result.handled_field == "ownership_type"
    call_kw = mock_tool.func.call_args[1]
    assert call_kw["ownership_type"] == DEFAULT_INTAKE_OWNERSHIP
    assert call_kw["raw_user_input"] is None
