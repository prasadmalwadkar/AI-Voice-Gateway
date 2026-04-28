"""Twilio bridge helpers: Cartesia debug redaction, logging gates, STT factory."""

import os
import unittest
from unittest.mock import MagicMock, patch

from app.cartesia_stt_client import CartesiaInkSTTSession
from app.deepgram_stt_client import DeepgramFluxSTTSession
from app.twilio_bridge import _log_cartesia_frame_if_enabled, _redact_cartesia_value
from app.twilio_line_bridge import _create_stt_session


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


class TestPath2STTProviderSelection(unittest.TestCase):
    def test_defaults_to_cartesia(self):
        with patch.dict(os.environ, {}, clear=True):
            session = _create_stt_session(MagicMock())
        self.assertIsInstance(session, CartesiaInkSTTSession)

    def test_selects_deepgram(self):
        with patch.dict(os.environ, {"PATH2_STT_PROVIDER": "deepgram"}):
            session = _create_stt_session(MagicMock())
        self.assertIsInstance(session, DeepgramFluxSTTSession)

    def test_rejects_unknown_provider(self):
        with patch.dict(os.environ, {"PATH2_STT_PROVIDER": "bogus"}):
            with self.assertRaises(RuntimeError):
                _create_stt_session(MagicMock())
