from urllib.parse import parse_qs, urlparse

import pytest

from app.deepgram_stt_client import (
    _build_deepgram_websocket_url,
    _normalize_flux_message,
)


def test_normalize_flux_update_as_partial():
    out = _normalize_flux_message(
        {
            "event": "Update",
            "turn_index": 0,
            "transcript": "hello there",
            "end_of_turn_confidence": 0.2,
        }
    )

    assert out == {
        "type": "transcript",
        "text": "hello there",
        "is_final": False,
        "provider": "deepgram",
        "deepgram_event": "Update",
        "confidence": 0.2,
        "raw": {
            "event": "Update",
            "turn_index": 0,
            "transcript": "hello there",
            "end_of_turn_confidence": 0.2,
        },
    }


def test_normalize_flux_end_of_turn_as_final():
    out = _normalize_flux_message(
        {
            "event": "EndOfTurn",
            "turn_index": 0,
            "transcript": "complete turn",
            "end_of_turn_confidence": 0.8,
        }
    )

    assert out is not None
    assert out["type"] == "transcript"
    assert out["text"] == "complete turn"
    assert out["is_final"] is True
    assert out["provider"] == "deepgram"
    assert out["deepgram_event"] == "EndOfTurn"


def test_normalize_ignores_unknown_messages():
    assert _normalize_flux_message({"event": "Connected"}) is None


def test_build_deepgram_url_defaults(monkeypatch):
    for name in (
        "DEEPGRAM_STT_MODEL",
        "DEEPGRAM_STT_EOT_THRESHOLD",
        "DEEPGRAM_STT_EAGER_EOT_THRESHOLD",
        "DEEPGRAM_STT_EOT_TIMEOUT_MS",
        "DEEPGRAM_STT_KEYTERMS",
    ):
        monkeypatch.delenv(name, raising=False)

    parsed = urlparse(_build_deepgram_websocket_url())
    qs = parse_qs(parsed.query)

    assert parsed.scheme == "wss"
    assert parsed.netloc == "api.deepgram.com"
    assert parsed.path == "/v2/listen"
    assert qs["model"] == ["flux-general-en"]
    assert qs["encoding"] == ["mulaw"]
    assert qs["sample_rate"] == ["8000"]
    assert qs["eot_threshold"] == ["0.7"]
    assert qs["eager_eot_threshold"] == ["0.4"]
    assert qs["eot_timeout_ms"] == ["3000"]


def test_build_deepgram_url_keyterms(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_STT_KEYTERMS", "West Capital, HELOC, sole owner")

    parsed = urlparse(_build_deepgram_websocket_url())
    qs = parse_qs(parsed.query)

    assert qs["keyterm"] == ["West Capital", "HELOC", "sole owner"]


def test_build_deepgram_url_rejects_invalid_numbers(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_STT_EOT_TIMEOUT_MS", "fast")

    with pytest.raises(ValueError):
        _build_deepgram_websocket_url()
