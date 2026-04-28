"""Unit tests for agent/response_lint.py (no server required)."""

from agent.response_lint import (
    contains_address_leak,
    first_turn_after_address_ok,
    occupancy_ask_without_ownership_cue,
)


def test_occupancy_without_ownership_detected():
    assert occupancy_ask_without_ownership_cue("What is the occupancy type?") is True
    assert occupancy_ask_without_ownership_cue("Primary, secondary, or investment?") is True


def test_sole_joint_ok():
    assert occupancy_ask_without_ownership_cue("Sole or joint owner?") is False


def test_address_leak():
    toks = ["325", "Los Angeles"]
    assert contains_address_leak("I have 325 West Adams", toks) is True
    assert contains_address_leak("Sole or joint owner?", toks) is False


def test_first_turn_lint_fail_on_log_example():
    bad = "What is the occupancy type?"
    ok, reason = first_turn_after_address_ok(bad, ["325", "Los Angeles"])
    assert ok is False
    assert "occupancy asked" in reason


def test_first_turn_lint_ok_short():
    ok, _ = first_turn_after_address_ok("Sole or joint owner?", ["325"])
    assert ok is True
