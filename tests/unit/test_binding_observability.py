import os
import threading
import time

from app.services.binding_agent import (
    AgentStep,
    BindingAgentContext,
    BindingAgentMlflowLogger,
    BindingAgentOrchestrator,
    BindingAgentResult,
)
from app.services.binding_agents.privacy import REDACTED, redact_payload, redact_text


os.environ.setdefault("DPLM_BINDING_AGENT_MLFLOW", "0")
os.environ.setdefault("DPLM_BINDING_RESEARCH_WEB", "0")


def _result() -> BindingAgentResult:
    return BindingAgentResult(
        ok=True,
        can_apply=True,
        error="",
        missing=[],
        gesture_label="palm",
        command_name="open_safari",
        mode="single",
        action_spec={"action": "open_app", "app": "Safari"},
        summary=[],
        response_text="Готово.",
        steps=[],
        intent="create_binding",
        intent_block="binding",
    )


def test_privacy_filter_redacts_credentials_recursively():
    secret = "sk-1234567890abcdefghijkl"
    payload = {
        "prompt": f"MISTRAL_API_KEY={secret}",
        "authorization": f"Bearer {secret}",
        "nested": [{"token": secret, "text": f"используй {secret}"}],
    }

    safe = redact_payload(payload)

    assert secret not in str(safe)
    assert safe["authorization"] == REDACTED
    assert safe["nested"][0]["token"] == REDACTED
    assert REDACTED in redact_text(payload["prompt"])


def test_async_mlflow_logger_does_not_block_request_thread(monkeypatch):
    logger = BindingAgentMlflowLogger(enabled=True, async_mode=True)
    completed = threading.Event()

    def slow_log(*_args, **_kwargs):
        time.sleep(0.15)
        completed.set()
        return {"mlflow_run_id": "background"}

    monkeypatch.setattr(logger, "_log_sync", slow_log)
    started = time.perf_counter()
    telemetry = logger.log(
        BindingAgentContext(prompt="test"),
        _result(),
        provider="local",
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 0.1
    assert telemetry["mlflow_queued"] is True
    logger.flush(timeout=1.0)
    assert completed.is_set()


def test_research_is_not_repeated_after_mistral_fallback(monkeypatch):
    monkeypatch.setenv("DPLM_BINDING_AGENT_LOCAL_FIRST", "1")

    class CountingResearch:
        def __init__(self):
            self.calls = 0

        def run(self, _context):
            self.calls += 1
            return AgentStep(
                "Research Agent",
                "need_clarification",
                "Рецепт не найден.",
                {"action_spec": {}, "source": "research_miss"},
            )

    class EmptyModel:
        model = "empty-model"

        def run(self, _context, *, intent, block):
            return (
                AgentStep("Mistral Agent", "ok", "Вернул пустой draft."),
                {
                    "gestureLabel": "palm",
                    "actionSpec": {},
                    "missing": ["действие"],
                    "agentReply": "Нужно уточнить действие.",
                },
            )

    research = CountingResearch()
    result = BindingAgentOrchestrator(
        research_agent=research,
        mistral_agent=EmptyModel(),
    ).run(
        "привяжи palm к включению bluetooth",
        [{"label": "palm"}],
        provider="mistral",
    )

    assert result.can_apply is False
    assert research.calls == 1
    assert len([step for step in result.steps if step.agent == "Research Agent"]) == 1


def test_pipeline_exposes_total_latency_without_mlflow():
    result = BindingAgentOrchestrator().run(
        "привяжи palm к открытию Safari",
        [{"label": "palm"}],
        provider="local",
    )

    assert result.telemetry["latency_ms"] >= 0.0
    assert result.telemetry["budget_remaining_ms"] > 0.0

