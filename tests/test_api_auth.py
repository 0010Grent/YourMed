import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    # Import after potential env changes in each test (tests set env per-test).
    from app.api_server import app

    return TestClient(app)


def test_triage_401_when_api_key_required_and_missing(client, monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEY", "secret")

    resp = client.post(
        "/v1/triage",
        json={"user_text": "hello", "top_k": 5, "mode": "safe"},
    )

    assert resp.status_code == 401
    body = resp.json()
    assert body["code"] == "UNAUTHORIZED"
    assert body["message"] == "Missing or invalid X-API-Key"
    assert isinstance(body.get("trace_id"), str) and body.get("trace_id")


def test_triage_401_when_api_key_required_and_wrong(client, monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEY", "secret")

    resp = client.post(
        "/v1/triage",
        json={"user_text": "hello", "top_k": 5, "mode": "safe"},
        headers={"X-API-Key": "wrong"},
    )

    assert resp.status_code == 401
    body = resp.json()
    assert body["code"] == "UNAUTHORIZED"
    assert body["message"] == "Missing or invalid X-API-Key"
    assert isinstance(body.get("trace_id"), str) and body.get("trace_id")


def test_triage_200_when_api_key_matches(client, monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEY", "secret")

    # Patch the imported triage_once symbol inside api_server.
    import app.api_server as api_server

    expected_payload = {
        "answer": {"triage_level": "ROUTINE"},
        "evidence": [],
        "rag_query": "q",
        "meta": {"mode": "safe", "created_at": "2025-01-01 00:00:00"},
    }

    def fake_triage_once(**kwargs):
        return expected_payload

    monkeypatch.setattr(api_server, "triage_once", fake_triage_once)

    resp = client.post(
        "/v1/triage",
        json={"user_text": "hello", "top_k": 5, "mode": "safe"},
        headers={"X-API-Key": "secret"},
    )

    assert resp.status_code == 200
    assert resp.json() == expected_payload


def test_triage_200_when_auth_disabled_emits_warning(client, monkeypatch, caplog):
    monkeypatch.delenv("TRIAGE_API_KEY", raising=False)

    import app.api_server as api_server

    expected_payload = {
        "answer": {"triage_level": "ROUTINE"},
        "evidence": [],
        "rag_query": "q",
        "meta": {"mode": "safe", "created_at": "2025-01-01 00:00:00"},
    }

    def fake_triage_once(**kwargs):
        return expected_payload

    monkeypatch.setattr(api_server, "triage_once", fake_triage_once)

    caplog.set_level("WARNING")
    resp = client.post(
        "/v1/triage",
        json={"user_text": "hello", "top_k": 5, "mode": "safe"},
    )

    assert resp.status_code == 200
    assert resp.json() == expected_payload

    assert any("鉴权未启用" in r.message for r in caplog.records)


def test_fast_mode_is_forced_to_safe_when_not_allowed(client, monkeypatch):
    monkeypatch.delenv("TRIAGE_API_KEY", raising=False)
    monkeypatch.delenv("ALLOW_FAST_MODE", raising=False)

    import app.api_server as api_server

    expected_payload = {
        "answer": {"triage_level": "ROUTINE"},
        "evidence": [],
        "rag_query": "q",
        "meta": {"mode": "safe", "created_at": "2025-01-01 00:00:00"},
    }

    def fake_triage_once(**kwargs):
        # API层会把 fast 强制改成 safe
        assert kwargs.get("mode") == "safe"
        return expected_payload

    monkeypatch.setattr(api_server, "triage_once", fake_triage_once)

    resp = client.post(
        "/v1/triage",
        json={"user_text": "hello", "top_k": 5, "mode": "fast"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["meta"]["mode"] == "safe"
    assert body["meta"]["forced_safe"] is True


def test_chat_401_when_api_key_required_and_missing(client, monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEY", "secret")

    resp = client.post(
        "/v1/chat",
        json={"patient_message": "hi"},
    )

    assert resp.status_code == 401
    body = resp.json()
    assert body["code"] == "UNAUTHORIZED"
    assert isinstance(body.get("trace_id"), str) and body.get("trace_id")


def test_chat_persists_session_to_output_dir(client, monkeypatch, tmp_path):
    monkeypatch.delenv("TRIAGE_API_KEY", raising=False)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    # Keep /v1/chat deterministic & offline in tests.
    monkeypatch.setenv("CHAT_SLOT_EXTRACTOR", "rules")

    import app.api_server as api_server

    expected_payload = {
        "answer": {"triage_level": "ROUTINE"},
        "evidence": [],
        "rag_query": "q",
        "meta": {"mode": "safe", "created_at": "2025-01-01 00:00:00"},
    }

    def fake_build_rag_query(user_text: str, clinical_record_text: str = "") -> str:
        return "q"

    def fake_step_retrieve(rag_query: str, top_k: int = 5, trace=None):
        if trace is None:
            trace = []
        return {"rag_query": rag_query, "evidence": [], "rag_status": "ok", "rag_error": None, "trace": trace}

    def fake_step_assess(user_text: str, evidence_list, trace=None):
        if trace is None:
            trace = []
        return {"evidence_block": "", "raw": "{}", "answer": {"triage_level": "ROUTINE"}, "trace": trace}

    def fake_step_safety(answer_json, evidence_block: str, evidence_list, trace=None):
        if trace is None:
            trace = []
        return {"answer": answer_json, "trace": trace}

    def fake_step_build_payload(
        answer_json,
        evidence_list,
        rag_query: str,
        mode: str,
        created_at: str,
        rag_status: str = "unknown",
        rag_error=None,
        trace=None,
    ):
        return expected_payload

    monkeypatch.setattr(api_server, "build_rag_query", fake_build_rag_query)
    monkeypatch.setattr(api_server, "triage_step_retrieve", fake_step_retrieve)
    monkeypatch.setattr(api_server, "triage_step_assess", fake_step_assess)
    monkeypatch.setattr(api_server, "triage_step_safety", fake_step_safety)
    monkeypatch.setattr(api_server, "triage_step_build_payload", fake_step_build_payload)

    # 1st message: inquiry (no triage)
    resp1 = client.post(
        "/v1/chat",
        json={"session_id": "s1", "patient_message": "头痛"},
    )
    assert resp1.status_code == 200
    body1 = resp1.json()
    assert body1["session_id"] == "s1"
    assert body1["act"] in {"INQUIRY", "DIAGNOSIS", "EXPLANATION", "RECOMMENDATION"}

    # 2nd message: force diagnosis by filling essentials quickly
    resp2 = client.post(
        "/v1/chat",
        json={"session_id": "s1", "patient_message": "从昨天开始，持续头痛，6分" , "mode": "safe"},
    )
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2["session_id"] == "s1"

    # Session file exists
    session_file = tmp_path / "sessions" / "s1.json"
    assert session_file.exists()
