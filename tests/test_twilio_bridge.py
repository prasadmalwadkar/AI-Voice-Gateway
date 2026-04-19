"""
Tests for Twilio bridge helpers (Cartesia debug redaction / logging gates).

Run: python -m unittest tests.test_twilio_bridge -v
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.twilio_bridge import (  # noqa: E402
    _log_cartesia_frame_if_enabled,
    _redact_cartesia_value,
)


class TestRedactCartesiaValue(unittest.TestCase):
    def test_redacts_media_payload(self):
        out = _redact_cartesia_value(
            {"event": "media_output", "media": {"payload": "a" * 100}}
        )
        self.assertEqual(out["media"]["payload"], "<redacted len=100>")

    def test_preserves_short_strings(self):
        out = _redact_cartesia_value(
            {"event": "ack", "transcript": "hello", "stream_id": "sid_x"}
        )
        self.assertEqual(out["transcript"], "hello")
        self.assertEqual(out["stream_id"], "sid_x")

    def test_redacts_long_strings(self):
        long_text = "x" * 600
        out = _redact_cartesia_value({"note": long_text})
        self.assertEqual(out["note"], "<redacted len=600>")


class TestLogCartesiaFrameIfEnabled(unittest.TestCase):
    def test_no_log_when_env_unset(self):
        log = MagicMock()
        with patch.dict(os.environ, {"CARTESIA_LOG_ALL_EVENTS": ""}):
            _log_cartesia_frame_if_enabled(log, {"event": "clear"})
        log.info.assert_not_called()

    def test_logs_non_media_when_flag_on(self):
        log = MagicMock()
        with patch.dict(os.environ, {"CARTESIA_LOG_ALL_EVENTS": "1"}):
            _log_cartesia_frame_if_enabled(
                log, {"event": "clear", "stream_id": "sid_1"}
            )
        log.info.assert_called_once()
        args, kwargs = log.info.call_args
        self.assertIn("clear", args[0])
        self.assertEqual(kwargs.get("event"), "cartesia.debug_frame")

    def test_skips_media_output_by_default(self):
        log = MagicMock()
        with patch.dict(
            os.environ,
            {"CARTESIA_LOG_ALL_EVENTS": "1", "CARTESIA_LOG_MEDIA_CHUNKS": ""},
        ):
            _log_cartesia_frame_if_enabled(
                log, {"event": "media_output", "media": {"payload": "abc"}}
            )
        log.info.assert_not_called()

    def test_logs_media_output_when_chunks_flag(self):
        log = MagicMock()
        with patch.dict(
            os.environ,
            {"CARTESIA_LOG_ALL_EVENTS": "1", "CARTESIA_LOG_MEDIA_CHUNKS": "1"},
        ):
            _log_cartesia_frame_if_enabled(
                log, {"event": "media_output", "media": {"payload": "x" * 200}}
            )
        log.info.assert_called_once()


if __name__ == "__main__":
    unittest.main()
