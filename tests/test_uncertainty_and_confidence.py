"""Uncertainty markers and confidence-threshold behavior (``agent.state_manager``)."""


def test_uncertainty_detection():
    """Hedging language in user speech is detected."""
    from agent.state_manager import detect_uncertainty

    markers = detect_uncertainty(
        "My address is 325 West Adams Blvd, or maybe it was East Adams, zip code 90007"
    )
    assert len(markers) > 0
    assert "maybe" in markers

    markers = detect_uncertainty("I think my income is around 80000")
    assert "i think" in markers

    markers = detect_uncertainty("Actually wait, let me check that")
    assert len(markers) >= 2

    markers = detect_uncertainty("I guess it could be 90007, not sure")
    assert "i guess" in markers
    assert "not sure" in markers

    markers = detect_uncertainty("325 West Adams Boulevard Los Angeles California 90007")
    assert len(markers) == 0

    markers = detect_uncertainty("My name is John Smith")
    assert len(markers) == 0

    markers = detect_uncertainty("Yes I agree to the credit pull")
    assert len(markers) == 0

    markers = detect_uncertainty("")
    assert len(markers) == 0

    markers = detect_uncertainty("   ")
    assert len(markers) == 0

    markers = detect_uncertainty("I THINK it was 90007")
    assert "i think" in markers


def test_confidence_threshold():
    """Confidence scoring vs ``CONFIDENCE_THRESHOLD``."""
    from agent.state_manager import CONFIDENCE_THRESHOLD

    assert 0.95 >= CONFIDENCE_THRESHOLD
    assert 0.85 >= CONFIDENCE_THRESHOLD
    assert 0.84 < CONFIDENCE_THRESHOLD
    assert 0.6 < CONFIDENCE_THRESHOLD
    score = None
    effective = 1.0 if score is None else score
    assert effective >= CONFIDENCE_THRESHOLD


def test_consent_field_logic():
    """Consent fields require high confidence and truthy value."""
    from agent.state_manager import CONFIDENCE_THRESHOLD, HIGH_CONFIDENCE_FIELDS

    field = "agreed_to_privacy_policy"
    score = 0.95
    value = True
    is_low = score < CONFIDENCE_THRESHOLD
    should_reject = field in HIGH_CONFIDENCE_FIELDS and (is_low or not value)
    assert not should_reject

    score = 0.5
    is_low = score < CONFIDENCE_THRESHOLD
    should_reject = field in HIGH_CONFIDENCE_FIELDS and (is_low or not value)
    assert should_reject

    score = 0.95
    value = False
    is_low = score < CONFIDENCE_THRESHOLD
    should_reject = field in HIGH_CONFIDENCE_FIELDS and (is_low or not value)
    assert should_reject

    field = "first_name"
    score = 0.5
    is_low = score < CONFIDENCE_THRESHOLD
    should_reject = field in HIGH_CONFIDENCE_FIELDS and (is_low or not value)
    assert not should_reject

