from __future__ import annotations


from app.log_utils import redact_auth, safe_preview_payload, truncate


def test_truncate_adds_ellipsis():
    assert truncate("hello", 5) == "hello"
    assert truncate("hello", 4) == "hel…"


def test_redact_auth_bearer_token():
    s = "Authorization: Bearer SECRET_TOKEN_123"
    redacted = redact_auth(s)
    assert "SECRET_TOKEN" not in redacted
    assert "Bearer ***" in redacted


def test_safe_preview_payload_handles_bytes_and_dict():
    assert safe_preview_payload(b"abc", limit_chars=10) == "<bytes len=3>"
    d = {"a": 1, "b": 2}
    out = safe_preview_payload(d, limit_chars=100)
    assert "keys" in out
