"""Pre-line deterministic intake (``apply_voice_intake_heuristics_before_line``)."""
import os

import pytest
from unittest.mock import AsyncMock, patch

from tests.preline_fixtures import (
    FOR_SALE_ASK_NEXT,
    OCCUPANCY_ASK_NEXT,
    _make_tool_result,
)

# Maps old browser stage names → conversation stage constants used by
# get_current_conversation_stage(). Tests patch the conv-stage function
# instead of the removed browser_manager.
_BROWSER_TO_CONV_STAGE = {
    "property": "PROPERTY",
    "income": "INCOME",
    "name": "PERSONAL",
    "birthday": "PERSONAL",   # PERSONAL + first/last set → sub-stage "birthday"
    "contact": "CONTACT",
    "consent": "CONSENT",
    "for-sale": "FOR_SALE",
    "primary-home": "PRIMARY_HOME",
}


@pytest.fixture(autouse=True)
def clear_prefill_state_between_tests():
    import agent.state_manager as _sm
    from agent.state_manager import set_customer_prefill

    set_customer_prefill({})
    _sm._current_call_state()._preline_last_handled_at = 0.0
    yield
    set_customer_prefill({})
    _sm._current_call_state()._preline_last_handled_at = 0.0


async def test_preline_income_garble_passes_low_confidence_for_confirmation():
    """Brittle STT income repairs must not auto-save at nominal high confidence."""
    from agent.schema import HelocApplicationState

    mock_state = HelocApplicationState()
    tool_return = _make_tool_result("Other income?")

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value="INCOME"),
        patch("agent.state_manager.application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        mock_tool.func = AsyncMock(return_value=tool_return)
        from agent.deterministic_parsers import _INCOME_AMBIGUOUS_STT_CONFIDENCE
        from agent.state_manager import apply_voice_intake_heuristics_before_line

        await apply_voice_intake_heuristics_before_line("one one five million")

    mock_tool.func.assert_called_once()
    call_kw = mock_tool.func.call_args[1]
    assert call_kw["annual_pre_tax_income"] == 1_500_000
    assert call_kw["confidence_score"] == _INCOME_AMBIGUOUS_STT_CONFIDENCE


async def test_prefill_confirmation_yes_commits_first_then_last_separately():
    from agent.schema import HelocApplicationState
    from agent.state_manager import (
        apply_voice_intake_heuristics_before_line,
        get_missing_fields,
        set_customer_prefill,
    )

    mock_state = HelocApplicationState()

    async def tool_effect(*args, **kwargs):
        if kwargs.get("first_name"):
            mock_state.first_name = kwargs["first_name"]
        if kwargs.get("last_name"):
            mock_state.last_name = kwargs["last_name"]
        return _make_tool_result("What's your date of birth?")

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value="PERSONAL"),
        patch("agent.state_manager.application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch("agent.state_manager._PRELINE_COOLDOWN_SECS", 0.0),
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        set_customer_prefill({"first_name": "Jane", "last_name": "Doe"})
        assert "first name as Jane" in get_missing_fields()
        mock_tool.func = AsyncMock(side_effect=tool_effect)

        result1 = await apply_voice_intake_heuristics_before_line("yes")
        assert result1.handled is True
        mock_tool.func.assert_called_once()
        kw1 = mock_tool.func.call_args[1]
        assert kw1["first_name"] == "Jane"
        assert "last_name" not in kw1
        mock_tool.func.reset_mock()

        assert "last name as Doe" in get_missing_fields()

        await apply_voice_intake_heuristics_before_line("yes")
        mock_tool.func.assert_called_once()
        kw2 = mock_tool.func.call_args[1]
        assert kw2["last_name"] == "Doe"
        assert "first_name" not in kw2


async def test_prefill_last_name_carry_over_thats_correct_commits_last_name():
    from agent.schema import HelocApplicationState
    from agent.state_manager import (
        apply_voice_intake_heuristics_before_line,
        get_missing_fields,
        set_customer_prefill,
    )

    mock_state = HelocApplicationState()

    async def tool_effect_first(*args, **kwargs):
        mock_state.first_name = kwargs["first_name"]
        return _make_tool_result("next")

    async def tool_effect_last(*args, **kwargs):
        mock_state.last_name = kwargs["last_name"]
        return _make_tool_result("What's your date of birth?")

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value="PERSONAL"),
        patch("agent.state_manager.application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch("agent.state_manager._PRELINE_COOLDOWN_SECS", 0.0),
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        set_customer_prefill({"first_name": "Jane", "last_name": "Doe"})
        assert "first name as Jane" in get_missing_fields()
        mock_tool.func = AsyncMock(side_effect=tool_effect_first)

        await apply_voice_intake_heuristics_before_line("yes")
        assert mock_state.first_name == "Jane"
        assert mock_tool.func.call_count == 1

        mock_tool.func.reset_mock()
        mock_tool.func = AsyncMock(side_effect=tool_effect_last)
        assert "last name as Doe" in get_missing_fields()

        rc = await apply_voice_intake_heuristics_before_line("That's correct.")
        assert rc.handled is True
        assert "separate clear yes" not in (rc.ask_next or "").lower()
        mock_tool.func.assert_called_once()
        assert mock_state.last_name == "Doe"


async def test_prefill_confirmation_yes_beats_stale_low_conf_pending():
    from agent.schema import HelocApplicationState
    from agent.state_manager import (
        _set_pending_low_confidence_confirmation,
        apply_voice_intake_heuristics_before_line,
        get_missing_fields,
        set_customer_prefill,
    )

    mock_state = HelocApplicationState(address_line_1="325 West Adams Boulevard")

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value="PROPERTY"),
        patch("agent.state_manager.application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        set_customer_prefill({"ownership_type": "Sole owner"})
        assert "sole owner" in get_missing_fields()
        _set_pending_low_confidence_confirmation({"address_line_1": "325 West Adams Boulevard"}, 0.95)
        mock_tool.func = AsyncMock(return_value=_make_tool_result("Property type?"))
        result = await apply_voice_intake_heuristics_before_line("Yes.")

    assert result.handled is True
    assert result.handled_field == "prefill_confirmation"
    call_kw = mock_tool.func.call_args[1]
    assert call_kw["ownership_type"] == "Sole owner"
    assert "address_line_1" not in call_kw
    assert call_kw["confidence_score"] == 1.0


async def test_prefill_confirmation_no_rejects_and_asks_normal_question():
    from agent.schema import HelocApplicationState
    from agent.state_manager import (
        apply_voice_intake_heuristics_before_line,
        get_missing_fields,
        set_customer_prefill,
    )

    mock_state = HelocApplicationState()

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value="PERSONAL"),
        patch("agent.state_manager.application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        set_customer_prefill({"first_name": "Jane", "last_name": "Doe"})
        assert "first name as Jane" in get_missing_fields()
        mock_tool.func = AsyncMock()
        result = await apply_voice_intake_heuristics_before_line("no")
        followup = get_missing_fields()

    assert result.handled is True
    assert result.handled_field == "prefill_confirmation_rejected"
    assert result.ask_next == "What's your first name?"
    assert 'Ask next: "What\'s your first name?"' in followup
    mock_tool.func.assert_not_called()


async def test_prefill_reset_clears_customer_payload():
    from agent.schema import HelocApplicationState
    from agent.state_manager import (
        get_customer_prefill_fields,
        reset_state,
        set_customer_prefill,
    )

    with patch("agent.state_manager.application_state", HelocApplicationState()):
        set_customer_prefill({"email_address": "jane@example.com"})
        assert get_customer_prefill_fields() == ["email_address"]
        await reset_state()
        assert get_customer_prefill_fields() == []


def test_prefill_initial_ask_greets_first_name_for_missing_address():
    from agent.schema import HelocApplicationState
    from agent.state_manager import get_initial_intake_ask, set_customer_prefill

    mock_state = HelocApplicationState()

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value="PROPERTY"),
        patch("agent.state_manager.application_state", mock_state),
    ):
        set_customer_prefill({"first_name": "Prasad"})
        assert get_initial_intake_ask() == (
            "Hi Prasad — what's your property address for financing?"
        )


def test_prefill_initial_ask_greets_first_name_for_on_file_address():
    from agent.schema import HelocApplicationState
    from agent.state_manager import get_initial_intake_ask, set_customer_prefill

    mock_state = HelocApplicationState()

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value="PROPERTY"),
        patch("agent.state_manager.application_state", mock_state),
    ):
        set_customer_prefill(
            {
                "first_name": "Prasad",
                "address_line_1": "325 West Adams Boulevard",
                "city": "Los Angeles",
            }
        )
        assert get_initial_intake_ask() == (
            "Hi Prasad, I have the property address on file as "
            "325 West Adams Boulevard, Los Angeles. Is that correct?"
        )


def test_prefill_initial_ask_prepends_hi_for_opening_non_address_ask():
    from agent.schema import HelocApplicationState
    from agent.state_manager import get_initial_intake_ask, set_customer_prefill

    mock_state = HelocApplicationState(address_line_1="1 Main St")

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value="PROPERTY"),
        patch("agent.state_manager.application_state", mock_state),
        patch(
            "agent.state_manager.get_missing_fields",
            return_value=(
                '[CONVERSATION STAGE: PROPERTY] Still needed: ownership '
                'Ask next: "Are you the sole owner, or is it jointly owned?"'
            ),
        ),
    ):
        set_customer_prefill({"first_name": "Pat"})
        assert get_initial_intake_ask() == (
            "Hi Pat, Are you the sole owner, or is it jointly owned?"
        )


def test_prefill_initial_ask_when_missing_line_has_no_quoted_ask():
    from agent.schema import HelocApplicationState
    from agent.state_manager import get_initial_intake_ask, set_customer_prefill

    mock_state = HelocApplicationState()

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value="PROPERTY"),
        patch("agent.state_manager.application_state", mock_state),
        patch(
            "agent.state_manager.get_missing_fields",
            return_value="[CONVERSATION STAGE: PROPERTY] Still waiting.",
        ),
    ):
        set_customer_prefill({"first_name": "Pat"})
        assert (
            get_initial_intake_ask()
            == "Hi Pat — what's your property address for financing?"
        )


def test_on_file_confirmation_yes_passes_non_answer_gate():
    from agent.deterministic_parsers import looks_like_answer_for_current_field

    missing = (
        '[CONVERSATION STAGE: PROPERTY] Still needed: ownership type. '
        'Ask next: "Got it, I\'ve saved your address. Regarding ownership, '
        'I have you down as the sole owner. Is that still correct?"'
    )

    assert looks_like_answer_for_current_field("Yes.", missing) is True
    assert looks_like_answer_for_current_field("correct.", missing) is True

    last_name_missing = (
        '[CONVERSATION STAGE: PERSONAL] Still needed: last name. '
        'Ask next: "I have your last name as Malwadkar. Is that still correct?"'
    )
    assert looks_like_answer_for_current_field("Yeah.", last_name_missing) is True
    assert looks_like_answer_for_current_field("correct.", last_name_missing) is True


async def test_preline_occupancy_garbled_final_not_commentary_reprompt():
    """No regex UI-meta path: unparsed occupancy utterance falls through (LLM/tool path)."""
    from agent.schema import HelocApplicationState
    from agent.state_manager import apply_voice_intake_heuristics_before_line

    mock_state = HelocApplicationState(
        address_line_1="123 LN",
        ownership_type="Sole owner",
    )

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value="PROPERTY"),
        patch("agent.state_manager.application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        mock_tool.func = AsyncMock()
        result = await apply_voice_intake_heuristics_before_line("It says Henry Home.")

    assert result.handled is False
    mock_tool.func.assert_not_called()


async def test_preline_occupancy_low_confidence_reprompt():
    """Deepgram-low final on occupancy ask → scripted reprompt, no tool."""
    from agent.schema import HelocApplicationState
    from agent.deterministic_parsers import OCCUPANCY_LOW_CONF_REPROMPT
    from agent.state_manager import apply_voice_intake_heuristics_before_line

    mock_state = HelocApplicationState(
        address_line_1="123 LN",
        ownership_type="Sole owner",
    )

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value="PROPERTY"),
        patch("agent.state_manager.application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        mock_tool.func = AsyncMock()
        result = await apply_voice_intake_heuristics_before_line(
            "Henry Home.",
            stt_confidence=0.50,
        )

    assert result.handled is True
    assert result.ask_next == OCCUPANCY_LOW_CONF_REPROMPT
    assert result.handled_field == "occupancy_type"
    mock_tool.func.assert_not_called()


async def test_preline_final_bad_partial_recovery_occupancy():
    """Final garbage but last partial had occupancy cue → deterministic save."""
    from agent.schema import HelocApplicationState
    from agent.state_manager import apply_voice_intake_heuristics_before_line

    mock_state = HelocApplicationState(
        address_line_1="123 LN",
        ownership_type="Sole owner",
    )

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value="PROPERTY"),
        patch("agent.state_manager.application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        mock_tool.func = AsyncMock(
            return_value=_make_tool_result("Is the property currently listed for sale?")
        )
        result = await apply_voice_intake_heuristics_before_line(
            "Henry Home.",
            stt_confidence=0.95,
            stt_recovery_text="Primary.",
        )

    assert result.handled is True
    assert result.handled_field == "occupancy_type"
    mock_tool.func.assert_called_once()


async def test_preline_it_says_primary_home_still_saves_occupancy():
    """Meta phrase + real answer → deterministic occupancy, not commentary-only."""
    from agent.schema import HelocApplicationState
    from agent.state_manager import apply_voice_intake_heuristics_before_line

    mock_state = HelocApplicationState(address_line_1="123 LN", ownership_type="Sole owner")

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value="PROPERTY"),
        patch("agent.state_manager.application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        mock_tool.func = AsyncMock(
            return_value=_make_tool_result("Is the property currently listed for sale?")
        )
        result = await apply_voice_intake_heuristics_before_line(
            "It says primary home on the screen."
        )

    assert result.handled is True
    assert result.handled_field == "occupancy_type"
    mock_tool.func.assert_called_once()


async def test_preline_redundant_occupancy_echo_skips_llm_and_tool():
    """Echo same occupancy while browser lags — deterministic next ask, no tool."""
    from agent.schema import HelocApplicationState
    from agent.state_manager import apply_voice_intake_heuristics_before_line

    mock_state = HelocApplicationState(
        address_line_1="123 LN",
        ownership_type="Sole owner",
        occupancy_type="Primary property",
    )

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value="PROPERTY"),
        patch("agent.state_manager.application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch(
            "agent.state_manager.get_missing_fields",
            return_value=(
                '[CONVERSATION STAGE: PROPERTY] Still needed: x '
                'Ask next: "What is your annual income?"'
            ),
        ),
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        mock_tool.func = AsyncMock()
        result = await apply_voice_intake_heuristics_before_line("Primary.")

    assert result.handled is True
    assert result.ask_next == "What is your annual income?"
    mock_tool.func.assert_not_called()


async def test_prefill_last_name_confirmation_accepts_repeated_yeah():
    from agent.schema import HelocApplicationState
    from agent.state_manager import (
        apply_voice_intake_heuristics_before_line,
        get_missing_fields,
        set_customer_prefill,
    )

    mock_state = HelocApplicationState(first_name="Prasad")

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value="PERSONAL"),
        patch("agent.state_manager.application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        set_customer_prefill({"last_name": "Malwadkar"})
        assert "I have your last name as Malwadkar" in get_missing_fields()
        mock_tool.func = AsyncMock(return_value=_make_tool_result("What is your date of birth?"))
        result = await apply_voice_intake_heuristics_before_line("Yeah. Yeah.")

    assert result.handled is True
    assert result.handled_field == "prefill_confirmation"
    assert result.ask_next == "What is your date of birth?"
    call_kw = mock_tool.func.call_args[1]
    assert call_kw["last_name"] == "Malwadkar"


@pytest.mark.parametrize(
    "conv_stage,stage,transcript,field,expected_ask,state_kwargs",
    [
        ("PROPERTY", "property", "sole owner", "ownership_type", OCCUPANCY_ASK_NEXT, {}),
        ("PROPERTY", "property", "joint", "ownership_type", OCCUPANCY_ASK_NEXT, {}),
        (
            "PROPERTY",
            "property",
            "3",
            "occupancy_type",
            FOR_SALE_ASK_NEXT,
            {"address_line_1": "123 Main St", "ownership_type": "Sole owner"},
        ),
        ("FOR_SALE", "for-sale", "no", "is_currently_for_sale", "Annual pre-tax income?", {}),
        ("FOR_SALE", "for-sale", "1", "is_currently_for_sale", "Annual pre-tax income?", {}),
        (
            "CONSENT",
            "consent",
            "yes",
            "agreed_to_privacy_policy",
            "Agree to electronic communications?",
            {},
        ),
        ("PERSONAL", "name", "Alex", "first_name", "Last name?", {}),
        ("INCOME", "income", "80K", "annual_pre_tax_income", "Other income?", {}),
        (
            "INCOME",
            "income",
            "3",
            "primary_source_of_income",
            "First name?",
            {"annual_pre_tax_income": 80000, "other_income": 0},
        ),
        # birthday: first_name + last_name in state so sub-stage resolves to "birthday"
        (
            "PERSONAL",
            "birthday",
            "01/05/1990",
            "date_of_birth",
            "Phone number?",
            {"first_name": "Alex", "last_name": "Rivera"},
        ),
        ("CONTACT", "contact", "2135550101", "mobile_phone_number", "Email address?", {}),
    ],
)
async def test_preline_heuristic_handled(conv_stage, stage, transcript, field, expected_ask, state_kwargs):
    """Each deterministic field returns HeuristicResult(handled=True) with correct field."""
    from agent.schema import HeuristicResult, HelocApplicationState

    mock_state = HelocApplicationState(**state_kwargs)
    tool_return = _make_tool_result(expected_ask)

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value=conv_stage),
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
    assert result.ask_next and result.ask_next.endswith(expected_ask)


async def test_consent_decline_privacy_explains_then_terminal():
    """First 'No' on required consent explains mandate + re-asks; second declines closes."""
    from agent.schema import HeuristicResult, HelocApplicationState

    rounds: dict[str, int] = {}
    mock_state = HelocApplicationState(
        address_line_1="123 Main Street",
        ownership_type="Sole owner",
        occupancy_type="Primary property",
        is_currently_for_sale=False,
        annual_pre_tax_income=100_000,
        other_income=0,
        primary_source_of_income="Employed full-time",
        first_name="Alex",
        last_name="Rivera",
        date_of_birth="01/05/1990",
        mobile_phone_number="2135550101",
        email_address="alex@example.com",
    )

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value="CONSENT"),
        patch("agent.state_manager.application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch("agent.state_manager._PRELINE_COOLDOWN_SECS", 0.0),
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        import agent.state_manager as sm

        sm._current_call_state().consent_decline_round = rounds
        mock_tool.func = AsyncMock(return_value="unexpected")
        from agent.state_manager import apply_voice_intake_heuristics_before_line

        r1 = await apply_voice_intake_heuristics_before_line("No.")
        mock_tool.func.assert_not_called()
        assert isinstance(r1, HeuristicResult)
        assert r1.handled is True
        assert r1.terminate_after_tts is False
        assert "required" in r1.ask_next.lower()

        r2 = await apply_voice_intake_heuristics_before_line("No.")
        mock_tool.func.assert_not_called()
        assert r2.handled is True
        assert r2.terminate_after_tts is True
        assert "goodbye" in r2.ask_next.lower()


async def test_preline_heuristic_not_handled_ambiguous():
    """Ambiguous input returns handled=False."""
    from agent.schema import HeuristicResult, HelocApplicationState

    mock_state = HelocApplicationState()

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value="PROPERTY"),
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
    tool_return = _make_tool_result(OCCUPANCY_ASK_NEXT)

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value="PROPERTY"),
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
        "I'm assuming sole owner for now. "
        "Is this your primary home, a second home, or an investment property?"
    )


async def test_primary_home_address_bridges_for_sale_without_pull_up_wording():
    """Avoid 'I'll pull that up' before for-sale ask — bare 'yes' was mis-read as for-sale."""
    from agent.schema import HeuristicResult, HelocApplicationState
    from agent.state_manager import apply_voice_intake_heuristics_before_line

    for_sale_ask = "Is the property currently listed for sale?"
    mock_state = HelocApplicationState(
        address_line_1="123 Main St",
        ownership_type="Sole owner",
        occupancy_type="Secondary property",
    )
    tool_return = _make_tool_result(for_sale_ask)

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value="PRIMARY_HOME"),
        patch("agent.state_manager.application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch("agent.state_manager._use_deterministic_address_parsing", return_value=True),
        patch(
            "agent.state_manager.infer_property_address_from_transcript",
            return_value="5100 Oak St",
        ),
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        mock_tool.func = AsyncMock(return_value=tool_return)
        result = await apply_voice_intake_heuristics_before_line("5100 Oak Street")

    assert isinstance(result, HeuristicResult)
    assert result.handled is True
    assert result.handled_field == "primary_home_address_line_1"
    assert "i'll pull that up" not in (result.ask_next or "").lower()
    assert (result.ask_next or "").startswith("Thanks for that.")
    assert for_sale_ask in (result.ask_next or "")


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
    from agent.deterministic_parsers import PARTIAL_MOBILE_PHONE_REPROMPT
    from agent.state_manager import apply_voice_intake_heuristics_before_line

    mock_state = HelocApplicationState()

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value="CONTACT"),
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


async def test_preline_contact_email_i_dont_know_forwards_to_line_keeps_prefill():
    from agent.schema import HelocApplicationState
    from agent.state_manager import (
        apply_voice_intake_heuristics_before_line,
        get_customer_prefill_fields,
        set_customer_prefill,
    )

    mock_state = HelocApplicationState(mobile_phone_number="2135550100")

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value="CONTACT"),
        patch("agent.state_manager.application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        set_customer_prefill({"email_address": "onfile@example.com"})
        mock_tool.func = AsyncMock()
        result = await apply_voice_intake_heuristics_before_line("I don't know.")

    assert result.handled is False
    mock_tool.func.assert_not_called()
    assert mock_state.email_address is None
    assert get_customer_prefill_fields() == ["email_address"]


async def test_preline_contact_spoken_nine_digit_spell_reprompt():
    """Word-only digit spell producing 9 digits → partial reprompt (not silent LLM fail)."""
    from agent.schema import HelocApplicationState
    from agent.deterministic_parsers import PARTIAL_MOBILE_PHONE_REPROMPT
    from agent.state_manager import apply_voice_intake_heuristics_before_line

    mock_state = HelocApplicationState()

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value="CONTACT"),
        patch("agent.state_manager.application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        mock_tool.func = AsyncMock()
        result = await apply_voice_intake_heuristics_before_line(
            "Four one three two seven five six zero six"
        )

    assert result.handled is True
    assert result.ask_next == PARTIAL_MOBILE_PHONE_REPROMPT
    mock_tool.func.assert_not_called()


async def test_prefill_reject_clears_mobile_phone_from_state():
    """Rejecting on-file mobile confirmation must drop stale digits from application_state."""
    import agent.state_manager as sm
    from agent.schema import HelocApplicationState

    mock_state = HelocApplicationState(mobile_phone_number="4132726066")

    pending = {
        "values": {"mobile_phone_number": "4132726066"},
        "ask": "I have your mobile number as 4132726066. Is that correct?",
    }
    with (
        patch.object(sm, "get_current_conversation_stage", return_value="CONTACT"),
        patch.object(sm, "application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
        patch.object(
            sm,
            "get_missing_fields",
            return_value='Ask next: "What\'s the best mobile number for this application?"',
        ),
    ):
        sm._current_call_state().prefill_confirm_pending = pending
        mock_tool.func = AsyncMock()
        result = await sm.apply_voice_intake_heuristics_before_line("No.")

    assert result.handled_field == "prefill_confirmation_rejected"
    assert mock_state.mobile_phone_number is None
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

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value="PROPERTY"),
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
    tool_return = _make_tool_result(
        "What is the property type: primary, secondary, or investment?"
    )

    with (
        patch("agent.state_manager.get_current_conversation_stage", return_value="PROPERTY"),
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


async def test_preline_primary_home_address_with_filler():
    from agent.schema import HelocApplicationState
    import agent.state_manager as sm

    mock_state = HelocApplicationState()
    tool_return = _make_tool_result("Is the property currently listed for sale?")

    with (
        patch.object(sm, "get_current_conversation_stage", return_value="PRIMARY_HOME"),
        patch.object(sm, "application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        mock_tool.func = AsyncMock(return_value=tool_return)
        result = await sm.apply_voice_intake_heuristics_before_line(
            "Yeah. Um, it's five one zero zero South Rolling Green Avenue, Sioux Falls."
        )

    assert result.handled is True
    assert result.handled_field == "primary_home_address_line_1"
    call_kw = mock_tool.func.call_args[1]
    assert "5100" in call_kw["primary_home_address_line_1"]


async def test_preline_dob_partial_reprompts_for_year():
    from agent.schema import HelocApplicationState
    from agent.deterministic_parsers import PARTIAL_DOB_REPROMPT
    import agent.state_manager as sm

    # first_name + last_name set → PERSONAL sub-stage resolves to "birthday"
    mock_state = HelocApplicationState(first_name="Alex", last_name="Rivera")

    with (
        patch.object(sm, "get_current_conversation_stage", return_value="PERSONAL"),
        patch.object(sm, "application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        mock_tool.func = AsyncMock()
        result = await sm.apply_voice_intake_heuristics_before_line("April seventh.")

    assert result.handled is True
    assert result.handled_field == "date_of_birth"
    assert result.ask_next == PARTIAL_DOB_REPROMPT
    mock_tool.func.assert_not_called()


async def test_preline_other_income_yes_asks_amount():
    from agent.schema import HelocApplicationState
    import agent.state_manager as sm

    mock_state = HelocApplicationState(annual_pre_tax_income=600000)

    with (
        patch.object(sm, "get_current_conversation_stage", return_value="INCOME"),
        patch.object(sm, "application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        mock_tool.func = AsyncMock()
        result = await sm.apply_voice_intake_heuristics_before_line("Yes.")

    assert result.handled is True
    assert result.handled_field == "other_income"
    assert "dollar" in result.ask_next.lower() or "amount" in result.ask_next.lower()
    mock_tool.func.assert_not_called()


async def test_preline_phone_nine_digits_with_filler_reprompt():
    from agent.schema import HelocApplicationState
    from agent.deterministic_parsers import PARTIAL_MOBILE_PHONE_REPROMPT
    import agent.state_manager as sm

    mock_state = HelocApplicationState()

    with (
        patch.object(sm, "get_current_conversation_stage", return_value="CONTACT"),
        patch.object(sm, "application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        mock_tool.func = AsyncMock()
        result = await sm.apply_voice_intake_heuristics_before_line(
            "Uh, it's four one three two seven five six zero six"
        )

    assert result.handled is True
    assert result.handled_field == "mobile_phone_number"
    assert result.ask_next == PARTIAL_MOBILE_PHONE_REPROMPT
    mock_tool.func.assert_not_called()


async def test_preline_prefill_reject_with_inline_phone():
    import agent.state_manager as sm
    from agent.schema import HelocApplicationState

    mock_state = HelocApplicationState(mobile_phone_number="9999999999")

    pending = {
        "values": {"mobile_phone_number": "9999999999"},
        "ask": "I have your mobile number as 9999999999. Is that correct?",
    }
    tool_return = _make_tool_result("What's the best email for this application?")

    with (
        patch.object(sm, "get_current_conversation_stage", return_value="CONTACT"),
        patch.object(sm, "application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
        patch.object(
            sm, "get_missing_fields",
            return_value='Ask next: "What\'s the best mobile number?"',
        ),
    ):
        sm._current_call_state().prefill_confirm_pending = pending
        mock_tool.func = AsyncMock(return_value=tool_return)
        result = await sm.apply_voice_intake_heuristics_before_line(
            "No, it's 4132756066"
        )

    assert result.handled is True
    assert result.handled_field == "prefill_confirmation_rejected"
    assert mock_state.mobile_phone_number is None
    mock_tool.func.assert_called_once()
    call_kw = mock_tool.func.call_args[1]
    assert call_kw["mobile_phone_number"] == "4132756066"


async def test_preline_income_uncertainty_saves_pending():
    from agent.schema import HelocApplicationState
    import agent.state_manager as sm

    mock_state = HelocApplicationState()

    with (
        patch.object(sm, "get_current_conversation_stage", return_value="INCOME"),
        patch.object(sm, "application_state", mock_state),
        patch("agent.state_manager.update_application_state") as mock_tool,
        patch.dict(os.environ, {"PATH2_ENABLE_PRELINE_HEURISTICS": "1"}),
    ):
        mock_tool.func = AsyncMock(
            return_value="Update PENDING — uncertainty detected. Ask the caller to confirm."
        )
        result = await sm.apply_voice_intake_heuristics_before_line(
            "I think it could be around six hundred thousand dollars."
        )

    assert result.handled is True
    assert result.handled_field == "annual_pre_tax_income"
    call_kw = mock_tool.func.call_args[1]
    assert call_kw["annual_pre_tax_income"] == 600_000
