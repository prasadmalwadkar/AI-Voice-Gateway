"""
Static checks on agent reply text (E2E script + unit tests).

LLMs may ignore prompts; these checks catch regressions in automation.
"""

from __future__ import annotations


def occupancy_ask_without_ownership_cue(text: str) -> bool:
    """
    True if reply pushes occupancy / primary-secondary-investment without sole/joint cues.
    Matches failure mode: 'What is the occupancy type?' before ownership collected.
    """
    t = (text or "").lower()
    asks_occ = (
        "occupancy" in t
        or "primary, secondary, or investment" in t
        or ("primary" in t and "secondary" in t and "investment" in t)
    )
    if not asks_occ:
        return False
    own_cue = (
        "sole" in t
        or "joint" in t
        or "ownership" in t
    )
    return not own_cue


def contains_address_leak(text: str, address_tokens: list[str]) -> bool:
    """True if any token (e.g. street number, city) appears in reply — should stay silent."""
    t = (text or "").lower()
    for tok in address_tokens:
        s = (tok or "").strip()
        if len(s) >= 3 and s.lower() in t:
            return True
    return False


def first_turn_after_address_ok(agent_text: str, address_tokens: list[str]) -> tuple[bool, str]:
    """
    Returns (ok, reason). Fails on address leak or skipping straight to occupancy.
    """
    if contains_address_leak(agent_text, address_tokens):
        return False, "address tokens repeated in agent speech"
    if occupancy_ask_without_ownership_cue(agent_text):
        return False, "occupancy asked without sole/joint/ownership wording"
    return True, ""
