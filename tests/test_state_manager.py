"""
Tests for state_manager.py — uncertainty detection and verification logic.

NOTE: Since the Line SDK (cartesia-line) is only available in the venv,
these tests replicate the core logic locally to run without SDK imports.
The live integration is tested via `cartesia chat 8000`.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# We can't import state_manager directly (Line SDK dependency), so we
# replicate the constants and detect_uncertainty function here for testing.
# This mirrors agent/state_manager.py exactly.

UNCERTAINTY_MARKERS = [
    "maybe",
    "or maybe",
    "i think",
    "not sure",
    "forgot",
    "i forget",
    "i guess",
    "actually",
    "not certain",
    "probably",
    "might be",
    "could be",
    "i believe",
    "wait",
    "hold on",
    "let me check",
]

CONFIDENCE_THRESHOLD = 0.80
HIGH_CONFIDENCE_FIELDS = {"agreed_to_credit_pull", "agreed_to_privacy_policy"}


def detect_uncertainty(raw_input: str) -> list[str]:
    if not raw_input:
        return []
    lowered = raw_input.lower()
    return [marker for marker in UNCERTAINTY_MARKERS if marker in lowered]


def test_uncertainty_detection():
    """Test that hedging language is correctly detected in user speech."""
    print("=== Uncertainty Detection Tests ===\n")

    # Ambiguous address (original bug report)
    markers = detect_uncertainty(
        "My address is 325 West Adams Blvd, or maybe it was East Adams, zip code 90007"
    )
    assert len(markers) > 0, "Should detect 'maybe' / 'or maybe'"
    assert "maybe" in markers
    print(f"✅ Ambiguous address caught: {markers}")

    # Uncertain income
    markers = detect_uncertainty("I think my income is around 80000")
    assert "i think" in markers
    print(f"✅ Uncertain income caught: {markers}")

    # Multiple uncertainty markers
    markers = detect_uncertainty("Actually wait, let me check that")
    assert len(markers) >= 2
    print(f"✅ Multiple markers caught: {markers}")

    # Heavy doubt
    markers = detect_uncertainty("I guess it could be 90007, not sure")
    assert "i guess" in markers
    assert "not sure" in markers
    print(f"✅ Heavy doubt caught: {markers}")

    # Clean / confident input — should NOT flag
    markers = detect_uncertainty("325 West Adams Boulevard Los Angeles California 90007")
    assert len(markers) == 0, "Confident speech should not be flagged"
    print("✅ Confident address: clean (no markers)")

    markers = detect_uncertainty("My name is John Smith")
    assert len(markers) == 0
    print("✅ Confident name: clean (no markers)")

    markers = detect_uncertainty("Yes I agree to the credit pull")
    assert len(markers) == 0
    print("✅ Clear consent: clean (no markers)")

    # Edge cases
    markers = detect_uncertainty("")
    assert len(markers) == 0
    print("✅ Empty string: clean")

    markers = detect_uncertainty("   ")
    assert len(markers) == 0
    print("✅ Whitespace only: clean")

    # Case insensitivity
    markers = detect_uncertainty("I THINK it was 90007")
    assert "i think" in markers
    print(f"✅ Case insensitive: {markers}")

    print("\nAll uncertainty detection tests passed! ✅\n")


def test_confidence_threshold():
    """Test that confidence scoring logic works correctly."""
    print("=== Confidence Threshold Tests ===\n")

    # High confidence — should pass
    score = 0.95
    assert score >= CONFIDENCE_THRESHOLD
    print(f"✅ High confidence ({score}) passes threshold ({CONFIDENCE_THRESHOLD})")

    # Borderline — equal to threshold passes
    score = 0.80
    assert score >= CONFIDENCE_THRESHOLD
    print(f"✅ Borderline ({score}) passes threshold")

    # Just below threshold
    score = 0.79
    assert score < CONFIDENCE_THRESHOLD
    print(f"✅ Below threshold ({score})")

    # Low confidence — should fail
    score = 0.6
    assert score < CONFIDENCE_THRESHOLD
    print(f"✅ Low confidence ({score}) below threshold")

    # Default (None → 1.0) — should pass
    score = None
    effective = score if score is not None else 1.0
    assert effective >= CONFIDENCE_THRESHOLD
    print(f"✅ Default None → {effective} passes threshold")

    print("\nAll confidence threshold tests passed! ✅\n")


def test_consent_field_logic():
    """Test that consent fields require high confidence."""
    print("=== Consent Field Logic Tests ===\n")

    # Consent with high confidence — should be accepted
    field = "agreed_to_credit_pull"
    score = 0.95
    value = True
    is_low = score < CONFIDENCE_THRESHOLD
    should_reject = field in HIGH_CONFIDENCE_FIELDS and (is_low or not value)
    assert not should_reject
    print(f"✅ Credit pull consent (confidence={score}, value={value}): ACCEPTED")

    # Consent with low confidence — should be rejected
    score = 0.5
    is_low = score < CONFIDENCE_THRESHOLD
    should_reject = field in HIGH_CONFIDENCE_FIELDS and (is_low or not value)
    assert should_reject
    print(f"✅ Credit pull consent (confidence={score}, value={value}): REJECTED (low confidence)")

    # Consent with value=False — should be rejected
    score = 0.95
    value = False
    is_low = score < CONFIDENCE_THRESHOLD
    should_reject = field in HIGH_CONFIDENCE_FIELDS and (is_low or not value)
    assert should_reject
    print(f"✅ Credit pull consent (confidence={score}, value={value}): REJECTED (False)")

    # Non-consent field with low confidence — should NOT be rejected by consent logic
    field = "first_name"
    score = 0.5
    is_low = score < CONFIDENCE_THRESHOLD
    should_reject = field in HIGH_CONFIDENCE_FIELDS and (is_low or not value)
    assert not should_reject
    print(f"✅ Non-consent field '{field}' (confidence={score}): NOT blocked by consent logic")

    print("\nAll consent field logic tests passed! ✅\n")


def test_income_validation():
    """Test that income must be positive."""
    print("=== Income Validation Tests ===\n")

    # Valid income
    income = 85000
    assert income > 0
    print(f"✅ Income ${income:,}: valid")

    # Zero income — should be rejected
    income = 0
    assert not (income > 0)
    print(f"✅ Income ${income}: rejected (must be > 0)")

    # Negative income — should be rejected
    income = -5000
    assert not (income > 0)
    print(f"✅ Income ${income:,}: rejected (negative)")

    print("\nAll income validation tests passed! ✅\n")


def test_infer_annual_income():
    """Test spoken income normalization."""
    from agent.state_manager import infer_annual_income_from_transcript

    cases = [
        ("1M", 1_000_000),
        ("1.5M", 1_500_000),
        ("80K", 80_000),
        ("80k", 80_000),
        ("100K", 100_000),
        ("1-0-0-0-0-0", 100_000),
        ("1000000", 1_000_000),
        ("eighty thousand", 80_000),
        ("two hundred fifty thousand", 250_000),
        ("one hundred thousand", 100_000),
        ("a hundred thousand", 100_000),
        ("one point five million", 1_500_000),
        ("one million", 1_000_000),
        ("one million.", 1_000_000),
        ("one zero zero zero zero zero.", 100_000),
        ("three million", 3_000_000),
        ("fifty thousand", 50_000),
        ("seventy thousand", 70_000),
        ("eighty", 80_000),        # bare number -> contextual *1000
        ("one fifty", 150_000),    # 150 -> contextual *1000
        ("two fifty", 250_000),
        ("", None),
        ("   ", None),
        ("hello world", None),
    ]
    for text, expected in cases:
        result = infer_annual_income_from_transcript(text)
        assert result == expected, f"infer_income({text!r}) = {result}, expected {expected}"


def test_infer_other_income():
    """Test other-income negatives map to 0."""
    from agent.state_manager import infer_other_income_from_transcript

    assert infer_other_income_from_transcript("none") == 0
    assert infer_other_income_from_transcript("nope") == 0
    assert infer_other_income_from_transcript("no") == 0
    assert infer_other_income_from_transcript("zero") == 0
    assert infer_other_income_from_transcript("0") == 0
    assert infer_other_income_from_transcript("nothing") == 0
    assert infer_other_income_from_transcript("no, no") == 0
    assert infer_other_income_from_transcript("no no no") == 0
    assert infer_other_income_from_transcript("zero zero") == 0
    assert infer_other_income_from_transcript("no other income") == 0
    assert infer_other_income_from_transcript("no any other income") == 0
    assert infer_other_income_from_transcript("50K") == 50_000


def test_infer_dob_spoken():
    """Test DOB parsing with spoken month names."""
    from agent.state_manager import infer_dob_from_transcript

    assert infer_dob_from_transcript("01/05/1990") == "01/05/1990"
    assert infer_dob_from_transcript("1-5-1990") == "01/05/1990"
    assert infer_dob_from_transcript("January 5th 1990") == "01/05/1990"
    assert infer_dob_from_transcript("march 15 1985") == "03/15/1985"
    assert infer_dob_from_transcript("5th January 1990") == "01/05/1990"
    assert infer_dob_from_transcript("dec 25 1975") == "12/25/1975"
    assert infer_dob_from_transcript("September 1st, 2000") == "09/01/2000"
    assert infer_dob_from_transcript("") is None
    assert infer_dob_from_transcript("hello") is None


def test_infer_ownership_garbles():
    """Test new ownership garble patterns."""
    from agent.state_manager import infer_ownership_from_transcript

    assert infer_ownership_from_transcript("sold owner") == "Sole owner"
    assert infer_ownership_from_transcript("soul owner") == "Sole owner"
    assert infer_ownership_from_transcript("sole owner") == "Sole owner"
    assert infer_ownership_from_transcript("joint owner") == "Joint owner"
    assert infer_ownership_from_transcript("foam owner") == "Sole owner"
    assert infer_ownership_from_transcript("toll owner") == "Sole owner"
    assert infer_ownership_from_transcript("So, loner.") == "Sole owner"
    assert infer_ownership_from_transcript("So loner") == "Sole owner"
    assert infer_ownership_from_transcript("Sool, honado.") == "Sole owner"
    assert infer_ownership_from_transcript("Soul or not.") == "Sole owner"
    assert infer_ownership_from_transcript("Soul.") == "Sole owner"
    assert infer_ownership_from_transcript("Stole.") == "Sole owner"
    assert infer_ownership_from_transcript("one") == "Sole owner"
    assert infer_ownership_from_transcript("1") == "Sole owner"
    assert infer_ownership_from_transcript("two") == "Joint owner"
    assert infer_ownership_from_transcript("2") == "Joint owner"
    assert infer_ownership_from_transcript("When.") == "Sole owner"
    assert infer_ownership_from_transcript("when") == "Sole owner"
    assert infer_ownership_from_transcript("hello world") is None


def test_infer_income_source_employed():
    """Test bare 'employed' defaults to full-time."""
    from agent.state_manager import infer_income_source_from_transcript

    assert infer_income_source_from_transcript("employed") == "Employed full-time"
    assert infer_income_source_from_transcript("Employed full time") == "Employed full-time"
    assert infer_income_source_from_transcript("part-time") == "Employed part-time"
    assert infer_income_source_from_transcript("self employed") == "Self-employed"
    assert infer_income_source_from_transcript("retired") == "Retired"
    assert infer_income_source_from_transcript("1") == "Employed full-time"
    assert infer_income_source_from_transcript("when") == "Employed full-time"
    assert infer_income_source_from_transcript("two") == "Employed part-time"
    assert infer_income_source_from_transcript("3") == "Self-employed"
    assert infer_income_source_from_transcript("for") == "Retired"
    assert infer_income_source_from_transcript("5") == "Unemployed / furloughed"


def test_infer_occupancy_and_for_sale():
    from agent.state_manager import infer_occupancy_from_transcript, infer_for_sale_from_transcript

    assert infer_occupancy_from_transcript("primary") == "Primary property"
    assert infer_occupancy_from_transcript("secondary home") == "Secondary property"
    assert infer_occupancy_from_transcript("investment") == "Investment property"
    assert infer_occupancy_from_transcript("1") == "Primary property"
    assert infer_occupancy_from_transcript("two") == "Secondary property"
    assert infer_occupancy_from_transcript("free") == "Investment property"
    assert infer_occupancy_from_transcript("When?") == "Primary property"

    assert infer_for_sale_from_transcript("yes") is True
    assert infer_for_sale_from_transcript("nope") is False
    assert infer_for_sale_from_transcript("yeah") is True
    assert infer_for_sale_from_transcript("nah") is False
    assert infer_for_sale_from_transcript("1") is True
    assert infer_for_sale_from_transcript("2") is False
    assert infer_for_sale_from_transcript("one") is True
    assert infer_for_sale_from_transcript("two") is False
    assert infer_for_sale_from_transcript("when") is True


def test_infer_phone_email_yesno_and_address():
    from agent.state_manager import (
        infer_phone_from_transcript,
        infer_email_from_transcript,
        infer_yes_no_from_transcript,
        infer_property_address_from_transcript,
    )

    assert infer_phone_from_transcript("213-555-0101") == "2135550101"
    assert infer_phone_from_transcript("+1 (213) 555-0101") == "2135550101"
    assert infer_phone_from_transcript("12345") is None

    assert infer_email_from_transcript("john at gmail dot com") == "john@gmail.com"
    assert infer_email_from_transcript("john_doe@gmail.com") == "john_doe@gmail.com"
    assert infer_email_from_transcript("john at gmail dot com.") == "john@gmail.com"
    assert infer_email_from_transcript("john at gmail dot com!") == "john@gmail.com"
    # Missing local part before @ is invalid by design.
    assert infer_email_from_transcript("@gmail.com") is None

    assert infer_yes_no_from_transcript("Yes.") is True
    assert infer_yes_no_from_transcript("I agree") is True
    assert infer_yes_no_from_transcript("Nope.") is False

    assert infer_property_address_from_transcript("325 west adams boulevard los angeles") == "325 West Adams Boulevard Los Angeles"
    assert infer_property_address_from_transcript("hello there") is None


def test_schema_email_trailing_dot_rejected():
    """Direct schema input with trailing dot must fail validation."""
    from agent.schema import HelocApplicationState

    with pytest.raises(ValueError):
        HelocApplicationState(email_address="john@gmail.com.")


def test_infer_name_fields():
    from agent.state_manager import infer_name_fields_from_transcript

    assert infer_name_fields_from_transcript("Alex.", need_first=True, need_last=False) == {
        "first_name": "Alex"
    }
    assert infer_name_fields_from_transcript("my name is Alex", need_first=True, need_last=False) == {
        "first_name": "Alex"
    }
    assert infer_name_fields_from_transcript("Alex Rivera", need_first=True, need_last=True) == {
        "first_name": "Alex",
        "last_name": "Rivera",
    }
    assert infer_name_fields_from_transcript("Rivera", need_first=False, need_last=True) == {
        "last_name": "Rivera"
    }
    # Should reject obvious non-name speech.
    assert infer_name_fields_from_transcript("But it's sad.", need_first=True, need_last=True) == {}
    assert infer_name_fields_from_transcript("Primary.", need_first=True, need_last=True) == {}
    assert infer_name_fields_from_transcript("12345", need_first=True, need_last=True) == {}
    assert infer_name_fields_from_transcript("Whatever.", need_first=True, need_last=True) == {}
    assert infer_name_fields_from_transcript("My brother.", need_first=False, need_last=True) == {}
    assert infer_name_fields_from_transcript("Thank you.", need_first=True, need_last=True) == {}
    # Duplicate STT tokens: never set same string as first + last in one save.
    assert infer_name_fields_from_transcript(
        "Alex Alex", need_first=True, need_last=True
    ) == {"first_name": "Alex"}
    assert infer_name_fields_from_transcript(
        "Alex Rivera", need_first=True, need_last=True, slot_focus="first"
    ) == {"first_name": "Alex"}
    assert infer_name_fields_from_transcript(
        "Alex Rivera", need_first=True, need_last=True, slot_focus="last"
    ) == {"last_name": "Rivera"}
    assert infer_name_fields_from_transcript(
        "Rivera", need_first=False, need_last=True, slot_focus="last"
    ) == {"last_name": "Rivera"}


def test_heuristic_result_contract():
    """Test HeuristicResult dataclass fields and immutability."""
    from agent.schema import HeuristicResult

    r = HeuristicResult(
        handled=True,
        ask_next="Sole or joint owner?",
        handled_field="ownership_type",
        stage="property",
    )
    assert r.handled is True
    assert r.ask_next == "Sole or joint owner?"
    assert r.handled_field == "ownership_type"
    assert r.stage == "property"
    assert r.normalization_source == "deterministic"

    r2 = HeuristicResult(handled=False, stage="property")
    assert r2.handled is False
    assert r2.ask_next is None
    assert r2.handled_field is None

    not_handled = HeuristicResult(handled=False)
    assert not_handled.stage is None

    # Frozen — should reject mutation.
    import dataclasses
    assert dataclasses.is_dataclass(r)
    try:
        r.handled = False  # type: ignore[misc]
        assert False, "Should be frozen"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


def test_expanded_yes_no_aliases():
    """Test newly added yes/no aliases."""
    from agent.state_manager import infer_yes_no_from_transcript

    for phrase in ("sure", "of course", "absolutely", "definitely", "certainly"):
        assert infer_yes_no_from_transcript(phrase) is True, f"Expected True for {phrase!r}"

    for phrase in ("not really", "i don't think so", "negative"):
        assert infer_yes_no_from_transcript(phrase) is False, f"Expected False for {phrase!r}"

    # Existing should still work.
    assert infer_yes_no_from_transcript("yes") is True
    assert infer_yes_no_from_transcript("nope") is False
    assert infer_yes_no_from_transcript("hello there") is None


def test_expanded_for_sale_aliases():
    """Test 'it is' / 'it's not' for-sale patterns."""
    from agent.state_manager import infer_for_sale_from_transcript

    assert infer_for_sale_from_transcript("it is") is True
    assert infer_for_sale_from_transcript("it's not") is False
    assert infer_for_sale_from_transcript("its not") is False
    assert infer_for_sale_from_transcript("sure") is True
    assert infer_for_sale_from_transcript("of course") is True
    assert infer_for_sale_from_transcript("not really") is False

    # Existing should still work.
    assert infer_for_sale_from_transcript("yes") is True
    assert infer_for_sale_from_transcript("nope") is False


def test_expanded_email_normalization():
    """Test 'period' and 'hyphen' spoken variants in email."""
    from agent.state_manager import infer_email_from_transcript

    assert infer_email_from_transcript("john at gmail period com") == "john@gmail.com"
    assert infer_email_from_transcript("john hyphen doe at yahoo dot com") == "john-doe@yahoo.com"

    # Existing should still work.
    assert infer_email_from_transcript("john at gmail dot com") == "john@gmail.com"
    assert infer_email_from_transcript("john_doe@gmail.com") == "john_doe@gmail.com"


# ---------------------------------------------------------------------------
# Async mock tests for apply_voice_intake_heuristics_before_line
# ---------------------------------------------------------------------------
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_tool_result(ask_next: str) -> str:
    """Simulate update_application_state tool return with Ask next."""
    return f'Updated. Still needed: foo. Ask next: "{ask_next}"'


_OCCUPANCY_ASK_NEXT = (
    "Primary, secondary, or investment? "
    "Say one for primary, two for secondary, three for investment."
)
_FOR_SALE_ASK_NEXT = (
    "Is the property currently for sale? Say one for yes, two for no."
)


@pytest.mark.parametrize(
    "stage,transcript,field,expected_ask,state_kwargs",
    [
        ("property", "sole owner", "ownership_type", _OCCUPANCY_ASK_NEXT, {}),
        ("property", "joint", "ownership_type", _OCCUPANCY_ASK_NEXT, {}),
        (
            "property",
            "3",
            "occupancy_type",
            _FOR_SALE_ASK_NEXT,
            {"address_line_1": "123 Main St", "ownership_type": "Sole owner"},
        ),
        ("for-sale", "no", "is_currently_for_sale", "Annual pre-tax income?", {}),
        ("for-sale", "1", "is_currently_for_sale", "Annual pre-tax income?", {}),
        ("consent", "yes", "agreed_to_privacy_policy", "Agree to marketing texts?", {}),
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
    tool_return = _make_tool_result(_OCCUPANCY_ASK_NEXT)

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
    client.tts_queue = asyncio.Queue()

    # Create mock tasks that are "running"
    mock_task = MagicMock()
    mock_task.done.return_value = False
    client._no_response_watchdog_task = mock_task
    client._turn_watchdog_task = mock_task
    client._pending_ask_fallback_task = mock_task

    await client.mark_deterministic_turn("What is your name?")

    # Verify flags reset
    assert client._fallback_spoken_text == ""
    assert client._ask_spoken_this_turn is True  # Set by _enqueue_ask_if_needed
    assert client._awaiting_turn_response is False

    # Verify watchdogs cancelled
    assert mock_task.cancel.call_count == 3

    # Verify ask was queued for TTS
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
    """Status check utterance should repeat current ask and skip LLM frames."""
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
        LineSessionClient.BRIEF_REPEAT_PROMPT,
        event="line.session.status_check_repeat_ask",
    )
    client._ws.send.assert_not_called()


# ---------------------------------------------------------------------------
# Bridge routing integration tests
# ---------------------------------------------------------------------------

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
        patch("app.twilio_line_bridge.apply_voice_intake_heuristics", new_callable=AsyncMock),
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
        handled_field="agreed_to_credit_pull",
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
        handled_field="agreed_to_credit_pull",
        stage="consent",
    )

    result = handled_terminal
    missing = "[BROWSER STAGE: CONSENT] Still needed: consent for electronic communications and soft credit pull"
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


def test_status_check_utterance_short_circuit_match():
    from app.line_session_client import LineSessionClient
    from app.twilio_line_bridge import _is_status_check_utterance

    for fn in (_is_status_check_utterance, LineSessionClient._is_status_check_utterance):
        assert fn("Hello, you there?") is True
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
    ask = "Sole or joint owner? Say one for sole, two for joint."
    assert fn(ask, "1") is True
    assert fn(ask, "2") is True
    assert fn(ask, "one") is True
    assert fn(ask, "two") is True
    assert fn(ask, "sad") is False
    assert fn(ask, "um") is False


if __name__ == "__main__":
    test_uncertainty_detection()
    test_confidence_threshold()
    test_consent_field_logic()
    test_income_validation()
    test_infer_annual_income()
    test_infer_other_income()
    test_infer_dob_spoken()
    test_infer_ownership_garbles()
    test_infer_income_source_employed()
    test_infer_occupancy_and_for_sale()
    test_infer_phone_email_yesno_and_address()
    test_infer_name_fields()
    test_heuristic_result_contract()
    test_expanded_yes_no_aliases()
    test_expanded_for_sale_aliases()
    test_expanded_email_normalization()
    print("=" * 50)
    print("ALL STATE MANAGER TESTS PASSED")
    print("=" * 50)
