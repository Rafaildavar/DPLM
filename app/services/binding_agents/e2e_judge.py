"""LLM-as-a-judge rubric for binding-agent end-to-end checks."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from app.services.binding_agent import BindingAgentResult


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).lower()


@dataclass(frozen=True)
class JudgeCriterion:
    criterion_id: str
    title: str
    instruction: str


JUDGE_CRITERIA: tuple[JudgeCriterion, ...] = (
    JudgeCriterion(
        "intent_routing",
        "Intent routing",
        "Оцени, попал ли запрос в правильный intent и intent block.",
    ),
    JudgeCriterion(
        "mode_and_status",
        "Mode and status",
        "Оцени, соответствует ли mode/canApply пользовательскому сценарию.",
    ),
    JudgeCriterion(
        "gesture_resolution",
        "Gesture resolution",
        "Оцени, правильно ли найден жест или корректно запрошено уточнение жеста.",
    ),
    JudgeCriterion(
        "action_resolution",
        "Action resolution",
        "Оцени, правильно ли собран actionSpec, включая URL, hotkey и sequence.",
    ),
    JudgeCriterion(
        "missing_fields",
        "Missing fields",
        "Оцени, совпадают ли missing fields с реальной недостающей частью.",
    ),
    JudgeCriterion(
        "contract_consistency",
        "Contract consistency",
        "Оцени, нет ли противоречия между ok/canApply/missing/actionSpec/ответом.",
    ),
    JudgeCriterion(
        "response_relevance",
        "Response relevance",
        "Оцени, отвечает ли текст агента именно на задачу пользователя.",
    ),
    JudgeCriterion(
        "safety_and_scope",
        "Safety and scope",
        "Оцени, не вышел ли агент за рамки GestureBind и guardrails.",
    ),
    JudgeCriterion(
        "user_next_step",
        "User next step",
        "Оцени, понятно ли пользователю, что делать дальше в UI.",
    ),
    JudgeCriterion(
        "trace_observability",
        "Trace observability",
        "Оцени, виден ли многоагентный путь: guardrails, intent, reviewer.",
    ),
)


@dataclass(frozen=True)
class BindingAgentE2ECase:
    case_id: str
    title: str
    prompt: str
    expected_intent: str
    expected_block: str
    expected_mode: str
    expected_can_apply: bool
    expected_gesture: str = ""
    expected_action: str = ""
    expected_missing: tuple[str, ...] = ()
    expected_sequence_steps: int | None = None
    required_response_markers: tuple[str, ...] = ()
    forbidden_response_markers: tuple[str, ...] = ("dplm", "traceback", "exception")
    current_gesture: str = ""
    gestures: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    min_judge_score: float = 0.9
    notes: str = ""


@dataclass(frozen=True)
class CriterionScore:
    criterion_id: str
    title: str
    score: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "title": self.title,
            "score": self.score,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class JudgeReport:
    case_id: str
    overall_score: float
    passed: bool
    scores: tuple[CriterionScore, ...]
    prompt: str
    judge_source: str = "local_rubric"

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "overall_score": self.overall_score,
            "passed": self.passed,
            "judge_source": self.judge_source,
            "scores": [score.to_dict() for score in self.scores],
            "prompt": self.prompt,
        }


class BindingAgentLlmJudge:
    """Pluggable LLM judge with deterministic fallback for offline tests."""

    def __init__(
        self,
        completion_fn: Callable[[str], str] | None = None,
        *,
        passing_score: float = 0.9,
    ) -> None:
        self.completion_fn = completion_fn
        self.passing_score = passing_score

    def evaluate(
        self,
        case: BindingAgentE2ECase,
        result: BindingAgentResult,
    ) -> JudgeReport:
        prompt = self.build_prompt(case, result)
        if self.completion_fn is not None:
            raw = self.completion_fn(prompt)
            parsed = self._parse_llm_scores(case, raw)
            if parsed:
                overall = self._overall(parsed)
                return JudgeReport(
                    case_id=case.case_id,
                    overall_score=overall,
                    passed=overall >= min(case.min_judge_score, self.passing_score),
                    scores=tuple(parsed),
                    prompt=prompt,
                    judge_source="llm",
                )

        scores = tuple(self._local_scores(case, result))
        overall = self._overall(scores)
        return JudgeReport(
            case_id=case.case_id,
            overall_score=overall,
            passed=overall >= min(case.min_judge_score, self.passing_score),
            scores=scores,
            prompt=prompt,
        )

    def build_prompt(
        self,
        case: BindingAgentE2ECase,
        result: BindingAgentResult,
    ) -> str:
        criteria = [
            {
                "id": criterion.criterion_id,
                "title": criterion.title,
                "instruction": criterion.instruction,
                "score": "0..1",
            }
            for criterion in JUDGE_CRITERIA
        ]
        payload = {
            "task": "Judge GestureBind binding-agent e2e result.",
            "return_format": {
                "criteria": {
                    criterion.criterion_id: {
                        "score": "float 0..1",
                        "reason": "short Russian explanation",
                    }
                    for criterion in JUDGE_CRITERIA
                }
            },
            "criteria": criteria,
            "case": self._case_payload(case),
            "result": self._result_payload(result),
        }
        return (
            "Ты LLM-as-a-judge для агента GestureBind. "
            "Оцени результат строго по 10 критериям и верни только JSON.\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

    def _local_scores(
        self,
        case: BindingAgentE2ECase,
        result: BindingAgentResult,
    ) -> list[CriterionScore]:
        checks = {
            "intent_routing": self._score_intent(case, result),
            "mode_and_status": self._score_mode(case, result),
            "gesture_resolution": self._score_gesture(case, result),
            "action_resolution": self._score_action(case, result),
            "missing_fields": self._score_missing(case, result),
            "contract_consistency": self._score_contract(case, result),
            "response_relevance": self._score_response(case, result),
            "safety_and_scope": self._score_safety(case, result),
            "user_next_step": self._score_next_step(case, result),
            "trace_observability": self._score_trace(result),
        }
        return [
            CriterionScore(
                criterion.criterion_id,
                criterion.title,
                checks[criterion.criterion_id][0],
                checks[criterion.criterion_id][1],
            )
            for criterion in JUDGE_CRITERIA
        ]

    def _score_intent(
        self,
        case: BindingAgentE2ECase,
        result: BindingAgentResult,
    ) -> tuple[float, str]:
        ok = (
            result.intent == case.expected_intent
            and result.intent_block == case.expected_block
        )
        return (
            1.0 if ok else 0.0,
            (
                "intent/block совпали."
                if ok
                else f"Ожидали {case.expected_intent}/{case.expected_block}, "
                f"получили {result.intent}/{result.intent_block}."
            ),
        )

    def _score_mode(
        self,
        case: BindingAgentE2ECase,
        result: BindingAgentResult,
    ) -> tuple[float, str]:
        ok = (
            result.mode == case.expected_mode
            and result.can_apply is case.expected_can_apply
        )
        return (
            1.0 if ok else 0.0,
            "mode/canApply корректны." if ok else "mode или canApply не совпали.",
        )

    def _score_gesture(
        self,
        case: BindingAgentE2ECase,
        result: BindingAgentResult,
    ) -> tuple[float, str]:
        if case.expected_gesture:
            ok = result.gesture_label == case.expected_gesture
            return (
                1.0 if ok else 0.0,
                (
                    "Жест распознан верно."
                    if ok
                    else f"Ожидали жест {case.expected_gesture}, получили {result.gesture_label}."
                ),
            )
        if "жест" in case.expected_missing:
            ok = "жест" in result.missing
            return (1.0 if ok else 0.0, "Запрос жеста корректен." if ok else "Жест не запрошен.")
        ok = not result.gesture_label
        return (1.0 if ok else 0.0, "Жест не нужен для answer-mode." if ok else "Лишний жест в answer-mode.")

    def _score_action(
        self,
        case: BindingAgentE2ECase,
        result: BindingAgentResult,
    ) -> tuple[float, str]:
        action = str(result.action_spec.get("action") or "")
        if not case.expected_action:
            ok = not action
            return (1.0 if ok else 0.0, "ActionSpec ожидаемо пуст." if ok else "ActionSpec лишний.")
        ok = action == case.expected_action
        if ok and case.expected_sequence_steps is not None:
            steps = result.action_spec.get("steps")
            ok = isinstance(steps, list) and len(steps) == case.expected_sequence_steps
        return (
            1.0 if ok else 0.0,
            "ActionSpec соответствует ожиданию." if ok else "ActionSpec не совпал.",
        )

    def _score_missing(
        self,
        case: BindingAgentE2ECase,
        result: BindingAgentResult,
    ) -> tuple[float, str]:
        ok = tuple(result.missing) == tuple(case.expected_missing)
        if ok and result.missing:
            response = _norm(result.response_text)
            ok = any(marker in response for marker in ("уточ", "укаж", "выбер", "нужно"))
        return (
            1.0 if ok else 0.0,
            "Missing fields и уточнение корректны." if ok else "Missing fields или текст уточнения неверны.",
        )

    def _score_contract(
        self,
        case: BindingAgentE2ECase,
        result: BindingAgentResult,
    ) -> tuple[float, str]:
        if result.can_apply:
            ok = bool(result.ok and result.gesture_label and result.action_spec and not result.missing)
        elif result.mode == "answer":
            ok = bool(result.ok and not result.action_spec and not result.missing)
        elif result.missing:
            ok = bool(not result.can_apply and not result.action_spec)
        else:
            ok = result.can_apply is case.expected_can_apply
        return (
            1.0 if ok else 0.0,
            "Контракт результата непротиворечив." if ok else "Контракт результата противоречив.",
        )

    def _score_response(
        self,
        case: BindingAgentE2ECase,
        result: BindingAgentResult,
    ) -> tuple[float, str]:
        response = _norm(result.response_text)
        if not response:
            return (0.0, "Ответ пустой.")
        if not case.required_response_markers:
            return (1.0, "Ответ непустой.")
        hits = sum(1 for marker in case.required_response_markers if _norm(marker) in response)
        ok = hits == len(case.required_response_markers)
        return (
            1.0 if ok else hits / max(1, len(case.required_response_markers)),
            "Ответ содержит ключевые маркеры." if ok else "Не все ключевые маркеры есть в ответе.",
        )

    def _score_safety(
        self,
        case: BindingAgentE2ECase,
        result: BindingAgentResult,
    ) -> tuple[float, str]:
        response = _norm(result.response_text)
        forbidden = [
            marker for marker in case.forbidden_response_markers if _norm(marker) in response
        ]
        scoped = True
        if case.expected_block in {"unsupported_general", "guardrails", "project_question"}:
            scoped = not result.can_apply and not result.action_spec
        ok = not forbidden and scoped
        return (
            1.0 if ok else 0.0,
            "Safety/scope соблюдены." if ok else "Есть forbidden marker или выход за scope.",
        )

    def _score_next_step(
        self,
        case: BindingAgentE2ECase,
        result: BindingAgentResult,
    ) -> tuple[float, str]:
        response = _norm(result.response_text)
        markers = ("можно", "сохран", "заполн", "уточ", "пример", "обычно", "зато")
        ok = result.can_apply or any(marker in response for marker in markers)
        return (
            1.0 if ok else 0.0,
            "Следующий шаг понятен." if ok else "Неясно, что делать дальше.",
        )

    def _score_trace(self, result: BindingAgentResult) -> tuple[float, str]:
        agents = [step.agent for step in result.steps]
        required = {"Guardrails Agent", "Intent Agent", "Reviewer Agent"}
        ok = required.issubset(set(agents))
        return (
            1.0 if ok else 0.0,
            "Trace содержит ключевых агентов." if ok else "Trace неполный.",
        )

    def _parse_llm_scores(
        self,
        case: BindingAgentE2ECase,
        raw: str,
    ) -> tuple[CriterionScore, ...]:
        _ = case
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return ()
        criteria_payload = payload.get("criteria")
        if not isinstance(criteria_payload, dict):
            return ()
        scores: list[CriterionScore] = []
        for criterion in JUDGE_CRITERIA:
            item = criteria_payload.get(criterion.criterion_id)
            if not isinstance(item, dict):
                return ()
            try:
                score = max(0.0, min(1.0, float(item.get("score"))))
            except (TypeError, ValueError):
                return ()
            scores.append(
                CriterionScore(
                    criterion.criterion_id,
                    criterion.title,
                    score,
                    str(item.get("reason") or ""),
                )
            )
        return tuple(scores)

    def _overall(self, scores: tuple[CriterionScore, ...] | list[CriterionScore]) -> float:
        if not scores:
            return 0.0
        return round(sum(score.score for score in scores) / len(scores), 4)

    def _case_payload(self, case: BindingAgentE2ECase) -> dict[str, Any]:
        return {
            "case_id": case.case_id,
            "title": case.title,
            "prompt": case.prompt,
            "expected_intent": case.expected_intent,
            "expected_block": case.expected_block,
            "expected_mode": case.expected_mode,
            "expected_can_apply": case.expected_can_apply,
            "expected_gesture": case.expected_gesture,
            "expected_action": case.expected_action,
            "expected_missing": list(case.expected_missing),
            "expected_sequence_steps": case.expected_sequence_steps,
            "required_response_markers": list(case.required_response_markers),
            "notes": case.notes,
        }

    def _result_payload(self, result: BindingAgentResult) -> dict[str, Any]:
        return {
            "ok": result.ok,
            "can_apply": result.can_apply,
            "error": result.error,
            "missing": list(result.missing),
            "gesture_label": result.gesture_label,
            "command_name": result.command_name,
            "mode": result.mode,
            "action_spec": dict(result.action_spec),
            "response_text": result.response_text,
            "intent": result.intent,
            "intent_block": result.intent_block,
            "steps": [
                {
                    "agent": step.agent,
                    "status": step.status,
                    "message": step.message,
                }
                for step in result.steps
            ],
        }
