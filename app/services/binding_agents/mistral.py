"""Model-backed binding agent adapter."""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from app.services.binding_agent import (
    AGENT_ACTION_LABELS,
    MISTRAL_API_KEY_ENV,
    MISTRAL_API_URL_ENV,
    MISTRAL_ANSWER_TEMPERATURE_ENV,
    MISTRAL_BINDING_TEMPERATURE_ENV,
    MISTRAL_DEFAULT_API_URL,
    MISTRAL_DEFAULT_MODEL,
    MISTRAL_MODEL_ENV,
    AgentStep,
    BindingAgentContext,
    _action_title,
    _clean_value,
    _explicit_gesture_label,
    _extract_json_object,
    _history_text,
    _load_project_dotenv,
    _norm,
    _normalize_missing,
    _normalize_model_action_spec,
    _normalize_summary,
    _short_text,
    _unique_missing,
)

class MistralBindingAgent:
    name = "Mistral Agent"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        api_url: str | None = None,
        timeout: float = 25.0,
        urlopen: Any = None,
    ) -> None:
        self._load_dotenv()
        self.api_key = api_key if api_key is not None else os.getenv(MISTRAL_API_KEY_ENV)
        self.model = model or os.getenv(MISTRAL_MODEL_ENV) or MISTRAL_DEFAULT_MODEL
        self.api_url = api_url or os.getenv(MISTRAL_API_URL_ENV) or MISTRAL_DEFAULT_API_URL
        self.timeout = timeout
        self.urlopen = urlopen or urllib.request.urlopen

    def _load_dotenv(self) -> None:
        _load_project_dotenv()

    def run(
        self,
        context: BindingAgentContext,
        *,
        intent: str = "create_binding",
        block: str = "binding",
    ) -> tuple[AgentStep, dict[str, Any] | None]:
        if not (self.api_key or "").strip():
            return (
                AgentStep(
                    self.name,
                    "need_input",
                    f"Не найден {MISTRAL_API_KEY_ENV}; использую локальный fallback.",
                    {"model": self.model},
                ),
                None,
            )

        payload = self._build_payload(context, intent=intent, block=block)
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            response = self.urlopen(request, timeout=self.timeout)
            try:
                raw = response.read()
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
        except urllib.error.HTTPError as exc:
            detail = self._error_body(exc)
            return (
                AgentStep(
                    self.name,
                    "blocked",
                    f"Mistral API вернул HTTP {exc.code}: {detail}",
                    {"model": self.model},
                ),
                None,
            )
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            return (
                AgentStep(
                    self.name,
                    "blocked",
                    f"Mistral API недоступен: {exc}",
                    {"model": self.model},
                ),
                None,
            )

        try:
            api_response = json.loads(raw.decode("utf-8"))
            content = self._message_content(api_response)
            draft = self._draft_from_content(content, context)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return (
                AgentStep(
                    self.name,
                    "blocked",
                    f"Mistral ответил невалидным draft JSON: {exc}",
                    {"model": self.model},
                ),
                None,
            )
        return (
            AgentStep(
                self.name,
                "ok",
                f"Модель {self.model} подготовила предложение привязки.",
                {
                    "model": self.model,
                    "intent": intent,
                    "block": block,
                    "purpose": "binding_parse",
                    "temperature": payload.get("temperature"),
                    "localPromptPreview": _short_text(
                        str(payload["messages"][1]["content"]),
                        360,
                    ),
                },
            ),
            draft,
        )

    def rewrite_answer(
        self,
        context: BindingAgentContext,
        *,
        intent: str,
        block: str,
        base_answer: str,
    ) -> tuple[AgentStep, str | None]:
        if not (self.api_key or "").strip():
            return (
                AgentStep(
                    self.name,
                    "need_input",
                    f"Не найден {MISTRAL_API_KEY_ENV}; использую локальный ответ.",
                    {
                        "model": self.model,
                        "intent": intent,
                        "block": block,
                        "purpose": "answer_rewrite",
                    },
                ),
                None,
            )

        payload = self._build_answer_payload(
            context,
            intent=intent,
            block=block,
            base_answer=base_answer,
        )
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            response = self.urlopen(request, timeout=self.timeout)
            try:
                raw = response.read()
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
        except urllib.error.HTTPError as exc:
            detail = self._error_body(exc)
            return (
                AgentStep(
                    self.name,
                    "blocked",
                    f"Mistral API вернул HTTP {exc.code}: {detail}",
                    {
                        "model": self.model,
                        "intent": intent,
                        "block": block,
                        "purpose": "answer_rewrite",
                    },
                ),
                None,
            )
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            return (
                AgentStep(
                    self.name,
                    "blocked",
                    f"Mistral API недоступен: {exc}",
                    {
                        "model": self.model,
                        "intent": intent,
                        "block": block,
                        "purpose": "answer_rewrite",
                    },
                ),
                None,
            )

        try:
            api_response = json.loads(raw.decode("utf-8"))
            content = self._message_content(api_response).strip()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return (
                AgentStep(
                    self.name,
                    "blocked",
                    f"Mistral ответил невалидным текстом: {exc}",
                    {
                        "model": self.model,
                        "intent": intent,
                        "block": block,
                        "purpose": "answer_rewrite",
                    },
                ),
                None,
            )
        if not content:
            return (
                AgentStep(
                    self.name,
                    "blocked",
                    "Mistral вернул пустой перефраз ответа.",
                    {
                        "model": self.model,
                        "intent": intent,
                        "block": block,
                        "purpose": "answer_rewrite",
                    },
                ),
                None,
            )
        return (
            AgentStep(
                self.name,
                "ok",
                f"Модель {self.model} перефразировала ответ для интента {intent}.",
                {
                    "model": self.model,
                    "intent": intent,
                    "block": block,
                    "purpose": "answer_rewrite",
                    "temperature": payload.get("temperature"),
                    "localPromptPreview": _short_text(
                        str(payload["messages"][1]["content"]),
                        360,
                    ),
                },
            ),
            content,
        )

    def _build_payload(
        self,
        context: BindingAgentContext,
        *,
        intent: str = "create_binding",
        block: str = "binding",
    ) -> dict[str, Any]:
        labels = [
            str(item.get("label") or "").strip()
            for item in context.gestures
            if str(item.get("label") or "").strip()
        ]
        gestures_text = ", ".join(labels) if labels else "нет известных жестов"
        actions_text = ", ".join(sorted(AGENT_ACTION_LABELS))
        local_prompt = self._local_binding_prompt(context, intent=intent, block=block)
        system_prompt = (
            "Ты семантический агент GestureFlow для привязки жестов к действиям macOS. "
            "Верни только JSON без markdown. Не исполняй команды. "
            "Схема: gestureLabel, commandName, mode, actionSpec, summary, "
            "missing, agentReply. actionSpec.action должен быть одним из: "
            f"{actions_text}. Для открытия приложения используй open_app/app; "
            "для сайта open_url/url; для hotkey key_combination/keys; "
            "для сценария sequence/steps. platform всегда macos. "
            "Для macOS перехода между рабочими столами/Spaces: "
            "налево=['ctrl','left'], направо=['ctrl','right']; не используй fn+arrow. "
            "Если пользователь указал только жест без команды, верни empty "
            "actionSpec и missing=['действие']; не выдумывай action вроде "
            "explain_capabilities. Понимай русские жесты: "
            "свайп вверх=swipe_up, свайп вниз=swipe_down, "
            "свайп влево=swipe_left, свайп вправо=swipe_right."
        )
        user_prompt = (
            f"Локальный перефраз интента для LLM:\n{local_prompt}\n\n"
            f"Известные жесты: {gestures_text}\n"
            f"Выбранный жест в UI: {context.current_gesture or 'нет'}\n"
            f"Диалоговая память:\n{_history_text(context.conversation_history) or 'нет'}\n"
            "Текущий черновик JSON:\n"
            f"{json.dumps(context.draft_state or {}, ensure_ascii=False)}\n"
            f"Фраза пользователя: {context.prompt}"
        )
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._temperature(MISTRAL_BINDING_TEMPERATURE_ENV, 0.25),
            "max_tokens": 700,
        }

    def _build_answer_payload(
        self,
        context: BindingAgentContext,
        *,
        intent: str,
        block: str,
        base_answer: str,
    ) -> dict[str, Any]:
        local_prompt = self._local_answer_prompt(
            context,
            intent=intent,
            block=block,
            base_answer=base_answer,
        )
        system_prompt = (
            "Ты редактор ответов агента GestureFlow. Не решай вопрос заново и не "
            "расширяй область ответственности. Перефразируй локальный черновик "
            "в живом русском tone of voice: дружелюбно, чуть тепло, без сухой "
            "канцелярщины. Сохраняй markdown-формат, смысл, ограничения и "
            "безопасные рамки. Для unsupported_general мягко возвращай разговор "
            "к GestureFlow. Верни только markdown-текст, без JSON."
        )
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": local_prompt},
            ],
            "temperature": self._temperature(MISTRAL_ANSWER_TEMPERATURE_ENV, 0.72),
            "max_tokens": 650,
        }

    def _local_binding_prompt(
        self,
        context: BindingAgentContext,
        *,
        intent: str,
        block: str,
    ) -> str:
        if intent == "build_sequence":
            task = (
                "Пользователь хочет сценарий из нескольких действий. Разбей фразу "
                "на steps и верни actionSpec.action='sequence'."
            )
        elif intent == "update_binding":
            task = (
                "Пользователь хочет изменить существующую привязку. Найди жест, "
                "новое действие и не придумывай недостающие поля."
            )
        else:
            task = (
                "Пользователь хочет создать привязку жеста. Извлеки gestureLabel, "
                "actionSpec и короткий agentReply в tone of voice GestureFlow."
            )
        return (
            f"block={block}; intent={intent}.\n"
            f"{task}\n"
            "Если не хватает жеста или действия, верни missing и не делай вид, "
            "что привязка готова."
        )

    def _local_answer_prompt(
        self,
        context: BindingAgentContext,
        *,
        intent: str,
        block: str,
        base_answer: str,
    ) -> str:
        if block == "unsupported_general":
            task = (
                "Сделай добрый redirect: признай вопрос, легко верни к GestureFlow, "
                "предложи 3-4 полезные возможности и один пример запроса."
            )
        elif intent == "validate_command":
            task = (
                "Перефразируй проверку команды без потери точности: да/нет, почему, "
                "и какая команда подходит лучше."
            )
        else:
            task = (
                "Перефразируй ответ по проекту GestureFlow: живо, понятно, без "
                "одинакового начала и без лишнего маркетинга."
            )
        return (
            f"Интент: {intent}\n"
            f"Блок: {block}\n"
            f"Локальная задача для LLM: {task}\n\n"
            f"Фраза пользователя:\n{context.prompt}\n\n"
            f"Диалоговая память:\n{_history_text(context.conversation_history) or 'нет'}\n\n"
            "Текущий черновик JSON:\n"
            f"{json.dumps(context.draft_state or {}, ensure_ascii=False)}\n\n"
            f"Локальный черновик ответа, который нужно только перефразировать:\n{base_answer}"
        )

    def _temperature(self, env_name: str, default: float) -> float:
        raw = str(os.getenv(env_name) or "").strip()
        if not raw:
            return default
        try:
            return max(0.0, min(1.2, float(raw.replace(",", "."))))
        except ValueError:
            return default

    def _message_content(self, api_response: dict[str, Any]) -> str:
        choice = api_response["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    chunks.append(str(item.get("text") or item.get("content") or ""))
                else:
                    chunks.append(str(item))
            return "".join(chunks)
        return str(content or "")

    def _draft_from_content(
        self,
        content: str,
        context: BindingAgentContext,
    ) -> dict[str, Any]:
        data = _extract_json_object(content)
        gesture = _clean_value(
            str(data.get("gestureLabel") or data.get("gesture_label") or "")
        )
        if not gesture:
            gesture = _explicit_gesture_label(context.prompt) or context.current_gesture
        action_spec = _normalize_model_action_spec(
            data.get("actionSpec") or data.get("action_spec")
        )
        missing = _normalize_missing(data.get("missing"))
        if not gesture:
            missing.append("жест")
        if not action_spec:
            missing.append("действие")
        if gesture:
            missing = [
                item for item in missing if _norm(item) not in {"жест", "gesture"}
            ]
        if action_spec:
            missing = [
                item
                for item in missing
                if _norm(item) not in {"действие", "команда", "action"}
            ]
        command_name = str(data.get("commandName") or data.get("command_name") or "")
        if not command_name and action_spec:
            command_name = (
                f"{gesture}: {_action_title(action_spec)}"
                if gesture
                else _action_title(action_spec)
            )
        mode = str(data.get("mode") or "").strip().lower()
        if mode not in {"single", "sequence"}:
            mode = "sequence" if action_spec.get("action") == "sequence" else "single"
        summary = _normalize_summary(data.get("summary"))
        agent_reply = str(data.get("agentReply") or data.get("agent_reply") or "")
        if not agent_reply:
            agent_reply = (
                f"Mistral: подготовил привязку «{gesture}»."
                if action_spec
                else "Mistral: нужно уточнить жест или действие."
            )
        return {
            "gestureLabel": gesture,
            "commandName": command_name,
            "mode": mode,
            "actionSpec": action_spec,
            "summary": summary,
            "missing": _unique_missing(missing),
            "agentReply": agent_reply,
        }

    def _error_body(self, exc: urllib.error.HTTPError) -> str:
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        except OSError:
            raw = str(exc.reason)
        raw = re.sub(r"\s+", " ", raw).strip()
        return raw[:260] if raw else str(exc.reason)
