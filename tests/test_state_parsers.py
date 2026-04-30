"""Transcript parsers: ``infer_*`` helpers and related schema checks."""
import dataclasses

import pytest


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
        ("one one five million", 1_500_000),
        ("it's one one five million", 1_500_000),
        ("two two five million", 225_000_000),
        ("two two five k", 225_000),
        ("one million", 1_000_000),
        ("one million.", 1_000_000),
        ("my annual pre tax income is one million", 1_000_000),
        ("manual pretax income is one million", 1_000_000),
        ("one medium", 1_000_000),
        ("One medium.", 1_000_000),
        ("one fifty k", 150_000),
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


def test_infer_annual_income_meta_flags_fragile_repairs():
    from agent.state_manager import infer_annual_income_meta

    assert infer_annual_income_meta("one one five million") == (1_500_000, True)
    assert infer_annual_income_meta("one point five million") == (1_500_000, False)
    assert infer_annual_income_meta("two two five million") == (225_000_000, True)
    assert infer_annual_income_meta("two two five k") == (225_000, True)


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
    assert infer_other_income_from_transcript("Not at this moment.") == 0
    assert infer_other_income_from_transcript("not right now") == 0
    assert infer_other_income_from_transcript("I don't have any other income") == 0
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
    assert infer_dob_from_transcript("January first nineteen ninety two") == "01/01/1992"
    assert infer_dob_from_transcript("First January nineteen ninety two") == "01/01/1992"
    assert infer_dob_from_transcript("first Jan nineteen ninety two") == "01/01/1992"
    assert infer_dob_from_transcript("My date of birth is 1st Jan 1992") == "01/01/1992"
    assert infer_dob_from_transcript("First ten nineteen ninety two") == "01/01/1992"
    assert infer_dob_from_transcript("") is None
    assert infer_dob_from_transcript("hello") is None


def test_infer_ownership_garbles():
    """Test new ownership garble patterns."""
    from agent.state_manager import infer_ownership_from_transcript

    assert infer_ownership_from_transcript("sold owner") == "Sole owner"
    assert infer_ownership_from_transcript("soul owner") == "Sole owner"
    assert infer_ownership_from_transcript("sole owner") == "Sole owner"
    assert infer_ownership_from_transcript("joint owner") == "Joint owner"
    assert infer_ownership_from_transcript("full owner") == "Sole owner"
    assert infer_ownership_from_transcript("foam owner") == "Sole owner"
    assert infer_ownership_from_transcript("toll owner") == "Sole owner"
    assert infer_ownership_from_transcript("So, loner.") == "Sole owner"
    assert infer_ownership_from_transcript("So loner") == "Sole owner"
    assert infer_ownership_from_transcript("So alone.") == "Sole owner"
    assert infer_ownership_from_transcript("Sool, honado.") == "Sole owner"
    assert infer_ownership_from_transcript("Soul or not.") == "Sole owner"
    assert infer_ownership_from_transcript("Soul.") == "Sole owner"
    assert infer_ownership_from_transcript("Stole.") == "Sole owner"
    assert infer_ownership_from_transcript("one") == "Sole owner"
    assert infer_ownership_from_transcript("1") == "Sole owner"
    assert infer_ownership_from_transcript("two") == "Joint owner"
    assert infer_ownership_from_transcript("2") == "Joint owner"
    assert infer_ownership_from_transcript("I am sole owner") == "Sole owner"
    assert infer_ownership_from_transcript("Yes I am sole owner of the property") == "Sole owner"
    assert infer_ownership_from_transcript("store owner") == "Sole owner"
    assert infer_ownership_from_transcript("stored owner") == "Sole owner"
    assert infer_ownership_from_transcript("join owner") == "Joint owner"
    assert infer_ownership_from_transcript("joined owners") == "Joint owner"
    assert infer_ownership_from_transcript("joist owner") == "Joint owner"
    assert infer_ownership_from_transcript("joynt owner") == "Joint owner"
    assert infer_ownership_from_transcript("jointed owner") == "Joint owner"
    assert infer_ownership_from_transcript("jointly owned") == "Joint owner"
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


def test_infer_occupancy_with_optional_recovery_partial():
    from agent.state_manager import infer_occupancy_with_optional_recovery

    eff, occ = infer_occupancy_with_optional_recovery("Henry Home.", "Primary.")
    assert occ == "Primary property"
    assert eff == "Primary."


def test_stt_confidence_reprompt_gate_unit_interval():
    from agent.state_manager import _stt_confidence_triggers_reprompt

    assert _stt_confidence_triggers_reprompt(None) is False
    assert _stt_confidence_triggers_reprompt(-0.1) is False
    assert _stt_confidence_triggers_reprompt(1.5) is False
    assert _stt_confidence_triggers_reprompt(-2.5) is False
    assert _stt_confidence_triggers_reprompt(0.51) is True
    assert _stt_confidence_triggers_reprompt(0.52) is False


def test_looks_like_answer_wcl_occupancy_ask():
    from agent.state_manager import looks_like_answer_for_current_field

    missing = (
        'Ask next: "Is this your primary home, a second home, '
        'or an investment property?"'
    )
    assert looks_like_answer_for_current_field("Primary.", missing) is True
    assert looks_like_answer_for_current_field("Investment.", missing) is True


def test_infer_occupancy_and_for_sale():
    from agent.state_manager import infer_occupancy_from_transcript, infer_for_sale_from_transcript

    assert infer_occupancy_from_transcript("primary") == "Primary property"
    assert infer_occupancy_from_transcript("it's primary") == "Primary property"
    assert infer_occupancy_from_transcript("yes this is my primary property") == "Primary property"
    assert infer_occupancy_from_transcript("secondary home") == "Secondary property"
    assert infer_occupancy_from_transcript("investment") == "Investment property"
    assert infer_occupancy_from_transcript("1") == "Primary property"
    assert infer_occupancy_from_transcript("two") == "Secondary property"
    assert infer_occupancy_from_transcript("free") == "Investment property"
    assert infer_occupancy_from_transcript("When?") == "Primary property"
    assert infer_occupancy_from_transcript("primer") == "Primary property"
    assert infer_occupancy_from_transcript("prime home") == "Primary property"
    assert infer_occupancy_from_transcript("first home") == "Primary property"
    assert infer_occupancy_from_transcript("main residence") == "Primary property"
    assert infer_occupancy_from_transcript("seccondary") == "Secondary property"
    assert infer_occupancy_from_transcript("second property") == "Secondary property"
    assert infer_occupancy_from_transcript("weekend home") == "Secondary property"
    assert infer_occupancy_from_transcript("ski house") == "Secondary property"
    assert infer_occupancy_from_transcript("vacation house") == "Secondary property"
    assert infer_occupancy_from_transcript("rental home") == "Investment property"
    assert infer_occupancy_from_transcript("income property") == "Investment property"
    assert infer_occupancy_from_transcript("non owner occupied") == "Investment property"
    assert infer_occupancy_from_transcript("landlord") == "Investment property"
    assert infer_occupancy_from_transcript("flip house") == "Investment property"

    assert infer_for_sale_from_transcript("yes") is True
    assert infer_for_sale_from_transcript("it's no") is False
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
    assert infer_phone_from_transcript("Two one three two seven two five five nine nine") == "2132725599"
    assert infer_phone_from_transcript("One three two seven two five five nine nine.") is None
    assert infer_phone_from_transcript("12345") is None

    assert infer_email_from_transcript("john at gmail dot com") == "john@gmail.com"
    assert infer_email_from_transcript("john_doe@gmail.com") == "john_doe@gmail.com"
    assert infer_email_from_transcript("john at gmail dot com.") == "john@gmail.com"
    assert infer_email_from_transcript("john at gmail dot com!") == "john@gmail.com"
    assert infer_email_from_transcript(
        "prasad mahavat kar eighty four at g mail dot com"
    ) == "prasadmahavatkar84@gmail.com"
    assert infer_email_from_transcript(
        "jane smith eight four eight at gmail dot com"
    ) == "janesmith848@gmail.com"
    assert infer_email_from_transcript("m n five six at g mail dot") == "mn56@gmail.com"
    assert infer_email_from_transcript(
        "Prasad. at direct g mail dot com."
    ) == "prasad@directgmail.com"
    # Missing local part before @ is invalid by design.
    assert infer_email_from_transcript("@gmail.com") is None

    assert infer_yes_no_from_transcript("Yes.") is True
    assert infer_yes_no_from_transcript("I agree") is True
    assert infer_yes_no_from_transcript("Yes. I said yes.") is True
    assert infer_yes_no_from_transcript("Yeah. Yeah.") is True
    assert infer_yes_no_from_transcript("Nope.") is False

    assert infer_property_address_from_transcript("325 west adams boulevard los angeles") == "325 West Adams Boulevard Los Angeles"
    assert infer_property_address_from_transcript("four one three four eleventh avenue northeast, seattle") == "4134 11th Avenue"
    assert infer_property_address_from_transcript("three twenty five.") is None
    assert infer_property_address_from_transcript("three twenty five West Adams Boulevard, Los Angeles") == "325 West Adams Boulevard"
    assert infer_property_address_from_transcript("One, two, one, two, three West Adams Boulevard, Boston, Venice.") == "12123 West Adams Boulevard"
    assert infer_property_address_from_transcript("it's 123 11th ave northeast new york") == "123 11th Avenue Northeast New York"
    assert infer_property_address_from_transcript("West Adams, Bilibod, Los Angeles") is None
    assert infer_property_address_from_transcript("123 oak street chicago illinois") == (
        "123 Oak Street Chicago Illinois"
    )
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
    assert infer_name_fields_from_transcript("my name is Alex Rivera", need_first=True, need_last=True) == {
        "first_name": "Alex",
        "last_name": "Rivera",
    }
    assert infer_name_fields_from_transcript("Alex Rivera", need_first=True, need_last=True) == {
        "first_name": "Alex"
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
    assert infer_name_fields_from_transcript("You good?", need_first=True, need_last=True) == {}
    assert infer_name_fields_from_transcript("Can you hear me?", need_first=True, need_last=True) == {}
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


def test_spelled_name_letters_are_joined():
    from agent.state_manager import infer_name_fields_from_transcript

    assert infer_name_fields_from_transcript(
        "p r a s a d",
        need_first=True,
        need_last=False,
        slot_focus="first",
    ) == {"first_name": "Prasad"}
    assert infer_name_fields_from_transcript(
        "em ay el",
        need_first=False,
        need_last=True,
        slot_focus="last",
    ) == {"last_name": "Mal"}

def test_ownership_garbles_soon_sown_sewn_zone():
    from agent.state_manager import infer_ownership_from_transcript

    assert infer_ownership_from_transcript("soon") == "Sole owner"
    assert infer_ownership_from_transcript("so on") == "Sole owner"
    assert infer_ownership_from_transcript("sown") == "Sole owner"
    assert infer_ownership_from_transcript("sewn") == "Sole owner"
    assert infer_ownership_from_transcript("zone") == "Sole owner"


def test_infer_property_address_plain_street():
    from agent.state_manager import infer_property_address_from_transcript

    r = infer_property_address_from_transcript("456 maple avenue boston massachusetts")
    assert r is not None
    assert r.startswith("456 ")
    assert "Maple" in r
    assert "Avenue" in r


def test_infer_property_city_state_from_transcript():
    from agent.state_manager import infer_property_city_state_from_transcript

    raw = (
        "Yeah. Um, it's five one zero zero South Rolling Green Avenue, Sioux Falls, SD."
    )
    assert infer_property_city_state_from_transcript(raw) == {
        "city": "Sioux Falls",
        "state": "SD",
    }
    assert infer_property_city_state_from_transcript("123 Main St") == {}


def test_build_address_autocomplete_type_query_appends_city_state():
    from agent.state_manager import build_address_autocomplete_type_query

    street = "5100 South Rolling Green Avenue"
    raw = (
        "Yeah. Um, it's five one zero zero South Rolling Green Avenue, Sioux Falls, SD."
    )
    assert (
        build_address_autocomplete_type_query(street, raw)
        == "5100 South Rolling Green Avenue Sioux Falls SD"
    )


def test_build_address_autocomplete_type_query_no_commas_returns_street_only():
    from agent.state_manager import build_address_autocomplete_type_query

    assert (
        build_address_autocomplete_type_query("123 Main St", "some transcript without commas")
        == "123 Main St"
    )


def test_build_address_autocomplete_type_query_empty_street():
    from agent.state_manager import build_address_autocomplete_type_query

    assert build_address_autocomplete_type_query("", "Boston, MA") is None


def test_normalize_scaffold_strips_fillers():
    from agent.state_manager import normalize_answer_scaffold

    assert normalize_answer_scaffold("Yeah. Um, five one zero zero Main St") == "five one zero zero Main St"
    assert normalize_answer_scaffold("Uh, four one three") == "four one three"
    assert normalize_answer_scaffold("Okay, sure, 600K") == "600K"

def test_normalize_scaffold_preserves_bare_yes_no():
    from agent.state_manager import normalize_answer_scaffold

    assert normalize_answer_scaffold("Yes.") == "Yes."
    assert normalize_answer_scaffold("No.") == "No."
    assert normalize_answer_scaffold("Yeah.") == "Yeah."
    assert normalize_answer_scaffold("Uh, no.") == "no."

def test_normalize_scaffold_chains():
    from agent.state_manager import normalize_answer_scaffold

    assert normalize_answer_scaffold("Yeah. Um, well, 123 Main St") == "123 Main St"


def test_address_parse_with_filler_lead_in():
    from agent.state_manager import infer_property_address_from_transcript

    r = infer_property_address_from_transcript(
        "Yeah. Um, it's five one zero zero South Rolling Green Avenue, Sioux Falls."
    )
    assert r is not None
    assert r.startswith("5100")

def test_address_parse_uh_its():
    from agent.state_manager import infer_property_address_from_transcript

    r = infer_property_address_from_transcript(
        "Uh, it's one two three four Elm Street."
    )
    assert r is not None
    assert r.startswith("1234")


def test_income_parse_with_uncertainty_words():
    from agent.state_manager import infer_annual_income_meta

    val, fragile = infer_annual_income_meta(
        "I think it could be around six hundred thousand dollars."
    )
    assert val == 600_000

def test_income_parse_per_annum():
    from agent.state_manager import infer_annual_income_meta

    val, fragile = infer_annual_income_meta(
        "It's six hundred thousand dollars per annum."
    )
    assert val == 600_000

def test_income_parse_might_be():
    from agent.state_manager import infer_annual_income_meta

    val, _ = infer_annual_income_meta("It might be about 80K a year.")
    assert val == 80_000


def test_partial_dob_month_day_only():
    from agent.state_manager import infer_partial_dob_month_day

    assert infer_partial_dob_month_day("April seventh.") is True
    assert infer_partial_dob_month_day("January 15") is True
    assert infer_partial_dob_month_day("7th April") is True
    assert infer_partial_dob_month_day("twenty first of March") is True

def test_partial_dob_full_date_not_partial():
    from agent.state_manager import infer_partial_dob_month_day

    assert infer_partial_dob_month_day("April 7th 1990") is False
    assert infer_partial_dob_month_day("01/05/1990") is False

def test_partial_dob_nonsense():
    from agent.state_manager import infer_partial_dob_month_day

    assert infer_partial_dob_month_day("hello there") is False
    assert infer_partial_dob_month_day("") is False


def test_partial_dob_cardinal_month_or_month_only():
    from agent.state_manager import infer_partial_dob_month_day

    assert infer_partial_dob_month_day("Nine February.") is True
    assert infer_partial_dob_month_day("In February.") is True
    assert infer_partial_dob_month_day("February.") is True
    assert infer_partial_dob_month_day("nine february nineteen ninety") is False


def test_transcript_supports_saved_dob_blocks_llm_guess():
    from agent.state_manager import _transcript_supports_saved_dob

    assert _transcript_supports_saved_dob("First gen.", "02/09/1990") is False
    assert _transcript_supports_saved_dob("Nine February.", "02/09/1990") is False
    assert _transcript_supports_saved_dob("February 9th 1990.", "02/09/1990") is True


def test_normalize_scaffold_strips_i_think_for_uncertainty_rescan():
    from agent.state_manager import detect_uncertainty, normalize_answer_scaffold

    cleaned = normalize_answer_scaffold("I think it's one million.")
    assert detect_uncertainty(cleaned) == []


def test_other_income_yes_is_not_zero():
    from agent.state_manager import infer_other_income_from_transcript

    assert infer_other_income_from_transcript("Yes.") is None
    assert infer_other_income_from_transcript("Yes, I do.") is None

def test_other_income_no_returns_zero():
    from agent.state_manager import infer_other_income_from_transcript

    assert infer_other_income_from_transcript("No.") == 0
    assert infer_other_income_from_transcript("Nope.") == 0
    assert infer_other_income_from_transcript("No other income") == 0


def test_extract_inline_phone_after_no():
    from agent.state_manager import _extract_inline_answer_after_no

    result = _extract_inline_answer_after_no(
        "No, it's 4132756066",
        {"mobile_phone_number": "old_number"},
    )
    assert result.get("mobile_phone_number") == "4132756066"

def test_extract_inline_email_after_no():
    from agent.state_manager import _extract_inline_answer_after_no

    result = _extract_inline_answer_after_no(
        "No, it's john@gmail.com",
        {"email_address": "old@old.com"},
    )
    assert result.get("email_address") == "john@gmail.com"

def test_extract_inline_bare_no():
    from agent.state_manager import _extract_inline_answer_after_no

    result = _extract_inline_answer_after_no("No.", {"mobile_phone_number": "123"})
    assert result == {}


def test_spoken_nine_digits_after_filler():
    from agent.state_manager import _spoken_digit_sequence_to_digits, normalize_answer_scaffold

    raw = "Uh, it's four one three two seven five six zero six"
    cleaned = normalize_answer_scaffold(raw)
    digits = _spoken_digit_sequence_to_digits(cleaned)
    assert digits == "413275606"
    assert len(digits) == 9
