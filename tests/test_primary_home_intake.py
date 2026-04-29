"""Primary residence step when HELOC collateral is secondary or investment."""
from unittest.mock import MagicMock, patch

from agent.schema import HelocApplicationState


def test_browser_stage_home_address_url():
    from agent.browser_manager import BrowserManager

    m = BrowserManager()
    page = MagicMock()
    page.url = (
        "https://heloc.westcapitallending.com/account/heloc/register/home-address"
    )
    m._page = page
    assert m.get_current_stage() == "primary-home"


def _secondary_collateral_state() -> HelocApplicationState:
    return HelocApplicationState(
        address_line_1="100 Beach Rd",
        city="Malibu",
        state="CA",
        zipcode="90265",
        ownership_type="Sole owner",
        occupancy_type="Secondary property",
    )


def test_missing_fields_strip_primary_home_until_home_address_url():
    from agent.state_manager import get_missing_fields

    state = _secondary_collateral_state()
    mock_browser = MagicMock()
    mock_browser.get_current_stage.return_value = "property"
    page = MagicMock()
    page.url = "https://heloc.westcapitallending.com/account/heloc/register"
    mock_browser._page = page

    with (
        patch("agent.state_manager.browser_manager", mock_browser),
        patch("agent.state_manager.application_state", state),
    ):
        line = get_missing_fields()

    assert "Waiting for navigation" in line
    assert "primary home" not in line.lower()


def test_missing_fields_shows_primary_home_on_home_address_url():
    from agent.state_manager import get_missing_fields

    state = _secondary_collateral_state()
    mock_browser = MagicMock()
    mock_browser.get_current_stage.return_value = "primary-home"
    page = MagicMock()
    page.url = (
        "https://heloc.westcapitallending.com/account/heloc/register/home-address"
    )
    mock_browser._page = page

    with (
        patch("agent.state_manager.browser_manager", mock_browser),
        patch("agent.state_manager.application_state", state),
    ):
        line = get_missing_fields()

    assert "primary home address" in line.lower()
    assert "Ask next" in line


def test_missing_fields_primary_occupancy_skips_primary_home_stage():
    from agent.state_manager import get_missing_fields

    state = HelocApplicationState(
        address_line_1="1 Main",
        city="LA",
        state="CA",
        zipcode="90001",
        ownership_type="Sole owner",
        occupancy_type="Primary property",
    )
    mock_browser = MagicMock()
    mock_browser.get_current_stage.return_value = "property"
    page = MagicMock()
    page.url = "https://heloc.westcapitallending.com/account/heloc/register"
    mock_browser._page = page

    with (
        patch("agent.state_manager.browser_manager", mock_browser),
        patch("agent.state_manager.application_state", state),
    ):
        line = get_missing_fields()

    assert "Waiting for navigation" in line
    assert "primary home" not in line.lower()


def test_prefill_primary_home_batches_extras():
    from agent.state_manager import (
        _prefill_values_for_field,
        browser_manager,
        set_customer_prefill,
    )

    mock_browser = MagicMock()
    mock_browser.get_current_stage.return_value = "primary-home"
    page = MagicMock()
    page.url = (
        "https://heloc.westcapitallending.com/account/heloc/register/home-address"
    )
    mock_browser._page = page
    st = HelocApplicationState(occupancy_type="Investment property")

    set_customer_prefill(
        {
            "primary_home_address_line_1": "9 Oak St",
            "primary_home_city": "Pasadena",
            "primary_home_state": "CA",
            "primary_home_zipcode": "91101",
        }
    )
    with (
        patch("agent.state_manager.browser_manager", mock_browser),
        patch("agent.state_manager.application_state", st),
    ):
        vals = _prefill_values_for_field("primary_home_address_line_1", 2)

    assert vals.get("primary_home_address_line_1") == "9 Oak St"
    assert vals.get("primary_home_city") == "Pasadena"
    set_customer_prefill({})
