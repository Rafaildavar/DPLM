import os

import pytest

from app.services.binding_agent import BindingAgentOrchestrator
from app.services.binding_agents.e2e_judge import (
    BindingAgentE2ECase,
    BindingAgentLlmJudge,
    JUDGE_CRITERIA,
)


os.environ.setdefault("DPLM_BINDING_AGENT_MLFLOW", "0")
os.environ.setdefault("DPLM_BINDING_AGENT_PROVIDER", "local")


GESTURES = (
    {"label": "palm"},
    {"label": "swipe_up"},
    {"label": "swipe_down"},
    {"label": "ctrlz"},
    {"label": "gun"},
    {"label": "swipe_left"},
)


E2E_CASES = (
    BindingAgentE2ECase(
        case_id="e2e_open_itmo_url",
        title="Привязка жеста к сайту магистратуры ИТМО",
        prompt="привяжи жест gun к открытию сайта https://abiturient.itmo.ru/magistracy",
        expected_intent="create_binding",
        expected_block="binding",
        expected_mode="single",
        expected_can_apply=True,
        expected_gesture="gun",
        expected_action="open_url",
        required_response_markers=("gun", "Открыть сайт"),
        gestures=GESTURES,
        notes="Проверяет готовую URL-привязку после кейса с абитуриентом ИТМО.",
    ),
    BindingAgentE2ECase(
        case_id="e2e_workspace_start_sequence",
        title="Рабочий старт из нескольких действий",
        prompt=(
            "собери рабочий старт: открыть рамблер почту, "
            "открыть приложение джира, включить заметки"
        ),
        expected_intent="build_sequence",
        expected_block="binding",
        expected_mode="sequence",
        expected_can_apply=True,
        expected_gesture="palm",
        expected_action="sequence",
        expected_sequence_steps=3,
        required_response_markers=("palm", "Сценарий"),
        current_gesture="palm",
        gestures=GESTURES,
        notes="Проверяет сценарии, которые пользователь задаёт обычной фразой.",
    ),
    BindingAgentE2ECase(
        case_id="e2e_missing_action_clarification",
        title="Жест указан, действие нужно уточнить",
        prompt="жест свайп вверх",
        expected_intent="create_binding",
        expected_block="binding",
        expected_mode="single",
        expected_can_apply=False,
        expected_gesture="swipe_up",
        expected_missing=("действие",),
        required_response_markers=("Уточните", "команду"),
        gestures=GESTURES,
        notes="Проверяет UX уточнения, когда пользователь назвал только жест.",
    ),
    BindingAgentE2ECase(
        case_id="e2e_validate_wrong_hotkey",
        title="Проверка спорной команды",
        prompt="разве command+z закрывает Telegram?",
        expected_intent="validate_command",
        expected_block="project_question",
        expected_mode="answer",
        expected_can_apply=False,
        required_response_markers=("Нет", "command+q"),
        gestures=GESTURES,
        notes="Проверяет, что агент отвечает на вопрос, а не создаёт неверную привязку.",
    ),
    BindingAgentE2ECase(
        case_id="e2e_unsupported_general_redirect",
        title="Мягкий отказ на общий вопрос",
        prompt="какая завтра погода?",
        expected_intent="unsupported_general_question",
        expected_block="unsupported_general",
        expected_mode="answer",
        expected_can_apply=False,
        required_response_markers=("GestureFlow", "привяз"),
        gestures=GESTURES,
        notes="Проверяет guardrails области ответственности без сухого отказа.",
    ),
)


@pytest.mark.integration
@pytest.mark.parametrize("case", E2E_CASES, ids=lambda item: item.case_id)
def test_binding_agent_e2e_case_passes_llm_as_judge(case):
    assert len(E2E_CASES) == 5
    assert len(JUDGE_CRITERIA) == 10

    result = BindingAgentOrchestrator().run(
        case.prompt,
        list(case.gestures),
        current_gesture=case.current_gesture,
        provider="local",
    )
    report = BindingAgentLlmJudge().evaluate(case, result)

    failed = [
        f"{score.criterion_id}: {score.score} ({score.reason})"
        for score in report.scores
        if score.score < 1.0
    ]
    assert report.passed, (
        f"{case.case_id} failed judge with score {report.overall_score}: "
        + "; ".join(failed)
    )
    assert report.overall_score >= case.min_judge_score
    assert len(report.scores) == 10
    assert all(0.0 <= score.score <= 1.0 for score in report.scores)
    assert "LLM-as-a-judge" in report.prompt
