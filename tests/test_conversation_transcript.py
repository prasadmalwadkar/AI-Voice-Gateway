import json

def test_append_transcript_turn_writes_ndjson(tmp_path, monkeypatch):
    monkeypatch.setenv("CONVERSATION_TRANSCRIPT_DIR", str(tmp_path))
    from app.conversation_transcript import append_transcript_turn, append_transcript_ui_fields

    append_transcript_turn("CA_test_001", "user", "hello")
    append_transcript_turn("CA_test_001", "agent", "Hi there.")
    append_transcript_ui_fields("CA_test_001", {"address_line_1": "123 Main St"})

    paths = sorted(tmp_path.glob("*_CA_test_001.jsonl"))
    assert len(paths) == 1
    path = paths[0]
    assert path.is_file()
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    u = json.loads(lines[0])
    a = json.loads(lines[1])
    ui = json.loads(lines[2])
    assert u["role"] == "user" and u["text"] == "hello"
    assert a["role"] == "agent" and a["text"] == "Hi there."
    assert ui["role"] == "ui" and ui["fields"]["address_line_1"] == "123 Main St"
    assert ui.get("scope") == "page"
    assert "ts" in u and "ts" in a


def test_append_transcript_ui_scope_full_and_browser_stage(tmp_path, monkeypatch):
    monkeypatch.setenv("CONVERSATION_TRANSCRIPT_DIR", str(tmp_path))
    from app.conversation_transcript import append_transcript_ui_fields

    append_transcript_ui_fields(
        "CA_scope",
        {"is_currently_for_sale": "True"},
        scope="page",
        browser_stage="for-sale",
    )
    append_transcript_ui_fields(
        "CA_scope",
        {"address_line_1": "1 Main", "email_address": "a@b.co"},
        scope="full",
    )

    path = sorted(tmp_path.glob("*_CA_scope.jsonl"))[0]
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    page = json.loads(lines[0])
    full = json.loads(lines[1])
    assert page["scope"] == "page"
    assert page["browser_stage"] == "for-sale"
    assert full["scope"] == "full"
    assert "browser_stage" not in full


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


def test_write_voice_sourced_fields_json(tmp_path, monkeypatch):
    monkeypatch.setenv("CONVERSATION_TRANSCRIPT_DIR", str(tmp_path))
    from app.conversation_transcript import write_voice_sourced_fields_json

    write_voice_sourced_fields_json(
        "CA_voice",
        {"address_line_1": "It's one two three Main.", "annual_pre_tax_income": "One million."},
    )
    path = tmp_path / "voice_fields" / "CA_voice.json"
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["call_sid"] == "CA_voice"
    assert data["fields"]["address_line_1"] == "It's one two three Main."
    assert data["fields"]["annual_pre_tax_income"] == "One million."


def test_write_voice_sourced_fields_disabled_when_transcripts_off(tmp_path, monkeypatch):
    monkeypatch.setenv("CONVERSATION_TRANSCRIPT_DIR", "")
    from app.conversation_transcript import write_voice_sourced_fields_json

    write_voice_sourced_fields_json("CA_z", {"x": "y"})
    assert not (tmp_path / "voice_fields").exists()
