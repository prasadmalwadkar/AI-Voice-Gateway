"""HELOC transcript UI snapshots: current page vs full export."""

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def transcript_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CONVERSATION_TRANSCRIPT_DIR", str(tmp_path))
    return tmp_path


def test_heloc_ui_page_snapshot_only_current_stage(transcript_dir):
    from agent.schema import HelocApplicationState
    from agent.state_manager import _append_heloc_ui_transcript_snapshot

    state = HelocApplicationState(
        address_line_1="1 Main St",
        city="San Jose",
        state="CA",
        zipcode="95111",
        ownership_type="Sole owner",
        occupancy_type="Primary property",
        is_currently_for_sale=True,
    )
    mock_browser = MagicMock()
    mock_browser.get_current_stage.return_value = "for-sale"

    with patch("app.path2_transcript_debug._path2_debug_get_call", return_value="CA_snap"):
        with patch("agent.state_manager.application_state", state):
            with patch("agent.state_manager.browser_manager", mock_browser):
                _append_heloc_ui_transcript_snapshot()

    paths = sorted(transcript_dir.glob("*_CA_snap.jsonl"))
    assert len(paths) == 1
    line = json.loads(paths[0].read_text(encoding="utf-8").strip().split("\n")[0])
    assert line["scope"] == "page"
    assert line["browser_stage"] == "for-sale"
    assert set(line["fields"].keys()) == {"is_currently_for_sale"}
    assert line["fields"]["is_currently_for_sale"] == "True"


def test_heloc_ui_final_snapshot_export_keys(transcript_dir):
    from agent.schema import HelocApplicationState
    from agent.state_manager import append_heloc_ui_transcript_final_snapshot

    import agent.state_manager as sm

    sm._voice_field_user_sources.clear()

    state = HelocApplicationState(
        address_line_1="1 Main St",
        city="San Jose",
        state="CA",
        zipcode="95111",
        ownership_type="Sole owner",
        occupancy_type="Primary property",
        is_currently_for_sale=True,
        annual_pre_tax_income=100000,
        email_address="a@b.co",
        agreed_to_privacy_policy=True,
        agreed_to_commerce_communication=False,
    )

    with patch("agent.state_manager.application_state", state):
        append_heloc_ui_transcript_final_snapshot("CA_fin")

    paths = sorted(transcript_dir.glob("*_CA_fin.jsonl"))
    assert len(paths) == 1
    line = json.loads(paths[0].read_text(encoding="utf-8").strip().split("\n")[0])
    assert line["scope"] == "full"
    assert line["fields"]["email_address"] == "a@b.co"
    assert line["fields"]["agreed_to_privacy_policy"] == "True"
    assert line["fields"]["agreed_to_commerce_communication"] == "False"
    assert "password" not in line["fields"]
    # Full row lists every export key (unset → empty string).
    from agent.state_manager import _HELOC_UI_FULL_EXPORT_KEYS, _SENSITIVE_FIELDS

    expected_keys = frozenset(_HELOC_UI_FULL_EXPORT_KEYS) - _SENSITIVE_FIELDS
    assert frozenset(line["fields"].keys()) == expected_keys
    assert line["fields"].get("first_name", "") == ""
    assert line["fields"].get("primary_home_address_line_1", "") == ""

    vf = transcript_dir / "voice_fields" / "CA_fin.json"
    assert vf.is_file()
    vf_data = json.loads(vf.read_text(encoding="utf-8"))
    assert vf_data["call_sid"] == "CA_fin"
    assert vf_data["fields"] == {}
