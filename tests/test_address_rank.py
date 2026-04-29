"""Unit tests for address autocomplete suggestion ranking."""

from agent.browser_manager import _score_address_suggestion


def test_rank_boosts_matching_street_number():
    typed = "6123 NEW YORK RD NEW YORK NY"
    hints = ["It's six one two three New York Road, New York"]
    worse = _score_address_suggestion(
        "787 MADISON AVE, NEW YORK, NY 10065",
        typed,
        hints,
    )
    better = _score_address_suggestion(
        "6123 NEW YORK RD, BROOKLYN, NY 11201",
        typed,
        hints,
    )
    assert better > worse


def test_rank_typo_street_still_scores_via_fuzzy():
    typed = "123 E 80TH ST"
    hints = []
    a = _score_address_suggestion(
        "123 EAST 80TH STREET, NEW YORK, NY 10075",
        typed,
        hints,
    )
    assert a >= 0.3
