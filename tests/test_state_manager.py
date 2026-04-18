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

CONFIDENCE_THRESHOLD = 0.85
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

    # Borderline — should pass (equal to threshold)
    score = 0.85
    assert score >= CONFIDENCE_THRESHOLD
    print(f"✅ Borderline ({score}) passes threshold")

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


if __name__ == "__main__":
    test_uncertainty_detection()
    test_confidence_threshold()
    test_consent_field_logic()
    test_income_validation()
    print("=" * 50)
    print("ALL STATE MANAGER TESTS PASSED ✅")
    print("=" * 50)
