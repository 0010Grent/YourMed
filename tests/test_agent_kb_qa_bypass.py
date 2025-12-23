from fastapi.testclient import TestClient


def test_agent_kb_question_bypasses_followups(monkeypatch):
    # Offline deterministic
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.api_server import app

    client = TestClient(app)

    resp = client.post(
        "/v1/agent/chat_v2",
        json={"user_message": "风疹病毒是怎么感染的？"},
    )

    assert resp.status_code == 200
    body = resp.json()

    assert body.get("mode") == "answer"
    assert isinstance(body.get("answer"), str) and body["answer"].strip()

    # In kb_qa strategy, should not emit followups
    assert (body.get("questions") or []) == []
    assert (body.get("next_questions") or []) == []

    trace = body.get("trace") or {}
    assert isinstance(trace, dict)
    assert trace.get("planner_strategy") == "kb_qa"
