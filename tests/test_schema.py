"""HelocApplicationState validation and ``generate_password`` behavior."""

import pytest

from agent.schema import HelocApplicationState


def test_heloc_schema():
    state = HelocApplicationState(
        first_name="John",
        last_name="Smith",
        address_line_1="742 Evergreen Terrace",
        city="Springfield",
        state="IL",
        zipcode="62704",
        occupancy_type="Primary property",
    )

    assert state.first_name == "John"
    assert state.last_name == "Smith"
    assert state.occupancy_type == "Primary property"

    pass_1 = state.generate_password("62704")
    assert len(pass_1) >= 12

    state.first_name = "Ed"
    state.last_name = "Wu"
    pass_2 = state.generate_password("62704")
    assert len(pass_2) >= 12

    state.first_name = "Alexander"
    state.last_name = "Hamilton"
    pass_3 = state.generate_password("62704")
    assert len(pass_3) >= 12

    with pytest.raises(ValueError):
        state.occupancy_type = "My house"

    state.mobile_phone_number = "213-555-0100"
    assert state.mobile_phone_number == "2135550100"

    state.mobile_phone_number = "1-213-555-0100"
    assert state.mobile_phone_number == "2135550100"

    with pytest.raises(ValueError):
        state.mobile_phone_number = "555-1234"

    state.zipcode = "90007"
    assert state.zipcode == "90007"

    with pytest.raises(ValueError):
        state.zipcode = "900"

    state.date_of_birth = "03/15/1990"
    assert state.date_of_birth == "03/15/1990"

    with pytest.raises(ValueError):
        state.date_of_birth = "March 15 1990"

    state.email_address = "john.smith@example.com"
    assert state.email_address == "john.smith@example.com"

    with pytest.raises(ValueError):
        state.email_address = "not-an-email"

    state.state = "ca"
    assert state.state == "CA"

    with pytest.raises(ValueError):
        state.state = "XX"
