import json

from app.services.binding_agent import OpenAICompatibleBindingAgent


class _Response:
    def read(self) -> bytes:
        return json.dumps({"choices": [{"message": {"content": "OK"}}]}).encode("utf-8")

    def close(self) -> None:
        return None


def test_generic_provider_uses_selected_endpoint_model_and_bearer_key():
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response()

    agent = OpenAICompatibleBindingAgent(
        provider_label="OpenAI",
        api_key="user-secret",
        api_key_env="",
        model="gpt-5-mini",
        api_url="https://api.openai.com/v1/chat/completions",
        urlopen=fake_urlopen,
    )

    ok, message = agent.check_connection()

    assert ok is True
    assert "gpt-5-mini" in message
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["authorization"] == "Bearer user-secret"
    assert captured["payload"]["model"] == "gpt-5-mini"


def test_local_compatible_endpoint_can_run_without_authorization_header():
    captured = {}

    def fake_urlopen(request, timeout):
        captured["authorization"] = request.get_header("Authorization")
        return _Response()

    agent = OpenAICompatibleBindingAgent(
        provider_label="Ollama (локально)",
        api_key="",
        api_key_env="",
        requires_api_key=False,
        model="gpt-oss:20b",
        api_url="http://127.0.0.1:11434/v1/chat/completions",
        urlopen=fake_urlopen,
    )

    ok, _message = agent.check_connection()

    assert ok is True
    assert captured["authorization"] is None
