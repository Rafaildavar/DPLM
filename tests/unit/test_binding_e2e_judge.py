import json

from app.services.binding_agent import AgentStep, BindingAgentResult
from app.services.binding_agents.e2e_judge import (
    BindingAgentE2ECase,
    BindingAgentLlmJudge,
    JUDGE_CRITERIA,
    MistralJudgeCompletion,
)


def _case(*, min_score: float = 0.9) -> BindingAgentE2ECase:
    return BindingAgentE2ECase(
        case_id="judge-open-safari",
        title="Open Safari",
        prompt="привяжи palm к открытию Safari",
        expected_intent="create_binding",
        expected_block="binding",
        expected_mode="single",
        expected_can_apply=True,
        expected_gesture="palm",
        expected_action="open_app",
        expected_action_spec={"action": "open_app", "app": "Safari"},
        min_judge_score=min_score,
    )


def _result() -> BindingAgentResult:
    return BindingAgentResult(
        ok=True,
        can_apply=True,
        error="",
        missing=[],
        gesture_label="palm",
        command_name="open_safari",
        mode="single",
        action_spec={"action": "open_app", "platform": "macos", "app": "Safari"},
        summary=[],
        response_text="Готово, привязку можно сохранить.",
        steps=[
            AgentStep("Guardrails Agent", "ok", "ok"),
            AgentStep("Intent Agent", "ok", "ok"),
            AgentStep("Reviewer Agent", "ok", "ok"),
        ],
        intent="create_binding",
        intent_block="binding",
    )


def _judge_json(score: float) -> str:
    return json.dumps(
        {
            "criteria": {
                criterion.criterion_id: {
                    "score": score,
                    "reason": "Проверено независимым judge.",
                }
                for criterion in JUDGE_CRITERIA
            }
        },
        ensure_ascii=False,
    )


def test_llm_judge_uses_all_ten_returned_criteria():
    report = BindingAgentLlmJudge(
        lambda _prompt: f"```json\n{_judge_json(0.96)}\n```"
    ).evaluate(_case(), _result())

    assert report.judge_source == "llm"
    assert report.overall_score == 0.96
    assert report.passed is True
    assert len(report.scores) == 10


def test_case_threshold_cannot_be_weakened_by_global_threshold():
    report = BindingAgentLlmJudge(
        lambda _prompt: _judge_json(0.92),
        passing_score=0.9,
    ).evaluate(_case(min_score=0.95), _result())

    assert report.passed is False


def test_llm_failure_falls_back_without_exposing_secret():
    def fail(_prompt):
        raise RuntimeError("token=sk-1234567890abcdefghijkl")

    report = BindingAgentLlmJudge(fail).evaluate(_case(), _result())

    assert report.judge_source == "deterministic_rubric_after_llm_error"
    assert "sk-1234567890abcdefghijkl" not in report.judge_error
    assert report.passed is True


def test_mistral_judge_adapter_requests_json_with_zero_temperature():
    captured = {}

    class Response:
        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {"message": {"content": _judge_json(1.0)}}
                    ]
                }
            ).encode("utf-8")

        def close(self):
            captured["closed"] = True

    def urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return Response()

    completion = MistralJudgeCompletion(
        api_key="judge-key",
        model="judge-model",
        timeout=4.0,
        urlopen=urlopen,
    )
    raw = completion("judge this")

    assert json.loads(raw)["criteria"]
    assert captured["payload"]["temperature"] == 0.0
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["authorization"] == "Bearer judge-key"
    assert captured["timeout"] == 4.0
    assert captured["closed"] is True

