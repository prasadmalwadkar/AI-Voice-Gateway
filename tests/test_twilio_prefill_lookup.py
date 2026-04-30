"""Phone-key customer prefill resolution for Twilio stream params."""

import json
import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def clear_prefill_json_env(monkeypatch):
    monkeypatch.delenv("TWILIO_PREFILL_BY_PHONE_JSON", raising=False)
    monkeypatch.delenv("TWILIO_PREFILL_BY_PHONE_FILE", raising=False)


def test_prefill_matches_lookup_key_variants():
    from app.main import _prefill_for_phone

    blob = {"+15551234567": {"first_name": "Sam", "last_name": "Lee"}}
    with patch.dict(os.environ, {"TWILIO_PREFILL_BY_PHONE_JSON": json.dumps(blob)}):
        assert _prefill_for_phone("+15551234567") == blob["+15551234567"]
        assert _prefill_for_phone("+1 (555) 123-4567")["first_name"] == "Sam"


def test_prefill_matches_embedded_mobile_when_lookup_key_differs():
    from app.main import _prefill_for_phone

    blob = {
        "+19999999999": {
            "first_name": "Fixture",
            "last_name": "CustomerA",
            "mobile_phone_number": "4132726066",
            "ownership_type": "Sole owner",
        }
    }
    with patch.dict(os.environ, {"TWILIO_PREFILL_BY_PHONE_JSON": json.dumps(blob)}):
        got = _prefill_for_phone("+14132726066")
        assert got.get("first_name") == "Fixture"
        assert got.get("mobile_phone_number") == "4132726066"


def test_prefill_no_false_match_short_digits():
    from app.main import _prefill_for_phone

    blob = {"+15551112222": {"first_name": "X"}}
    with patch.dict(os.environ, {"TWILIO_PREFILL_BY_PHONE_JSON": json.dumps(blob)}):
        assert _prefill_for_phone("+1555999") == {}
