import json

def test_append_transcript_turn_writes_ndjson(tmp_path, monkeypatch):
    monkeypatch.setenv("CONVERSATION_TRANSCRIPT_DIR", str(tmp_path))
    from app.conversation_transcript import append_transcript_turn

    append_transcript_turn("CA_test_001", "user", "hello")
    append_transcript_turn("CA_test_001", "agent", "Hi there.")

    path = tmp_path / "CA_test_001.jsonl"
    assert path.is_file()
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    u = json.loads(lines[0])
    a = json.loads(lines[1])
    assert u["role"] == "user" and u["text"] == "hello"
    assert a["role"] == "agent" and a["text"] == "Hi there."
    assert "ts" in u and "ts" in a


def test_append_transcript_turn_disabled_empty_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CONVERSATION_TRANSCRIPT_DIR", "")
    from app.conversation_transcript import append_transcript_turn

    append_transcript_turn("CA_x", "user", "nope")
    assert not (tmp_path / "CA_x.jsonl").exists()


def test_append_transcript_turn_skips_empty_text(tmp_path, monkeypatch):
    monkeypatch.setenv("CONVERSATION_TRANSCRIPT_DIR", str(tmp_path))
    from app.conversation_transcript import append_transcript_turn

    append_transcript_turn("CA_y", "user", "   ")
    assert not (tmp_path / "CA_y.jsonl").exists()
