# GestureFlow MAS audit

Дата: 2026-06-30

Этот документ фиксирует текущую точку отсчета для многоагентной системы
привязок GestureFlow. Он нужен, чтобы каждый следующий шаг улучшения можно
было сверять с планом, тестами и пользовательскими сценариями.

## Текущий pipeline

Публичный вход остается в `app.services.binding_agent.BindingAgentOrchestrator`.
UI вызывает совместимый wrapper `build_agent_binding_draft`, а результат
возвращается в legacy-формате для Flet-экрана привязок.
Контекст агента включает текущий prompt, список жестов, выбранный жест,
диалоговую историю и `draft_state` последнего черновика.

Порядок обработки:

1. `Guardrails Agent` проверяет вход: длина, секреты, prompt injection.
   Input guardrails v2 возвращают `decision=allow/block`.
2. `Intent Agent` выбирает route:
   - `create_binding`;
   - `update_binding`;
   - `build_sequence`;
   - `validate_command`;
   - `project_question`;
   - `unsupported_general_question`;
   - `guardrail_block`.
   Начиная с шага 3, intent trace содержит `routeMethod`,
   `semanticIntent`, `semanticScore` и, если применимо, `semanticExample`.
3. Для вопросов по проекту и unsupported-вопросов отвечает conversation block.
4. Для binding-route идет цепочка:
   - `Gesture Agent`;
   - `Memory Agent`;
   - `Scenario Agent` или `Action Agent`;
   - `Policy Agent`;
   - `Validation Agent`.
   Начиная с шага 4, concrete agents используют явные tool contracts:
   `resolve_gesture`, `parse_macos_action`, `build_sequence`,
   `validate_binding_contract`, `review_answer_contract`.
5. `Guardrails Agent` проверяет выход. Output guardrails v2 возвращают
   `decision=allow/clarify/block`; неполная привязка является уточнением,
   а не отказом.
6. `Reviewer Agent` оценивает соответствие ответа intent-блоку и пишет
   rubric scores: relevance, contract completeness, safety, tone,
   clarification quality.
7. `BindingAgentMlflowLogger` пишет trace, параметры, метрики, artifacts и
   optional GenAI eval. В MLflow уходят route method, semantic score и
   reviewer rubric metrics.

## Что уже работает

- Агент вынесен из UI в `app/services/binding_agent.py`.
- Конкретные роли вынесены в `app/services/binding_agents/`.
- Каталог trace-skills вынесен в `app/services/binding_agents/skills/`.
- Mistral может использоваться как provider для structured draft и answer rewrite.
- MLflow логирует multi-agent pipeline при включенном окружении.
- UI хранит историю диалога и показывает ответ агента без технической плашки.
- Неполная привязка больше не превращается в guardrails refusal: если есть
  действие, но нет жеста, агент просит уточнить жест.
- Intent routing стал гибридным: rule-based проверки остаются первыми, а
  свободные формулировки может подхватить локальный semantic-router без
  внешних зависимостей.
- Появился отдельный слой tools/contracts: агенты теперь тонко оборачивают
  tool-result в trace step, а не держат всю бизнес-логику внутри `run`.
- Контекст агента получил `draft_state`: уточнения могут продолжать прошлый
  черновик, например `привяжи это к swipe_up` после собранного сценария.
- Guardrails v2 разделяет блокировки и уточнения: missing gesture/action
  проходит как `decision=clarify`, prompt injection и секреты остаются block.
- Reviewer пишет rubric scores, а MLflow логирует `reviewer_*` метрики и
  `intent_semantic_score`.
- UI агента показывает человекочитаемые статусы: например, сценарий без жеста
  отображается как `Выберите жест для сценария`, а не общий technical state.

## Главные проблемы перед улучшением

1. Intent routing пока в основном rule-based. Он быстрый, но плохо понимает
   свободные формулировки. Базовый локальный semantic-router уже добавлен,
   следующий этап - заменить/дополнить его настоящими embeddings.
2. Skills сейчас существуют как Python-каталог для trace, но еще не оформлены
   как самостоятельные skill packs с `SKILL.md`.
3. Tools получили базовый contract-слой. Следующий этап - расширить его
   typed-схемами для LLM/function calling и MLflow eval.
4. Memory теперь получает draft-state для базовых уточнений. Следующий этап -
   сделать полноценный state manager с несколькими черновиками и TTL.
5. Guardrails v2 уже разделяет allow/clarify/block. Следующий этап -
   расширить политику опасных действий и explainability для UI.
6. Reviewer получил базовые рубрики. Следующий этап - добавить eval dataset
   и regression dashboard по этим score.
7. MLflow отражает semantic score и reviewer metrics. Следующий этап -
   связать это с полноценными GenAI scorers/datasets.

## Базовые сценарии для проверки

- `жест palm открывает Safari` -> single binding, ready.
- `сохрани ctrlz как command+z` -> hotkey binding, ready.
- `жест свайп вверх` -> gesture resolved, action missing.
- `свайп вверх открыть Safari` -> gesture synonym resolved.
- `что ты умеешь делать?` -> project question, no binding draft.
- `какая завтра погода?` -> friendly redirect to GestureFlow.
- `разве command+z закрывает Telegram?` -> validation answer.
- `сделай сценарий под названием мое утро ...` -> named sequence,
  missing gesture, no guardrails refusal.
- `привяжи это к swipe_up` после named sequence -> использует прошлый
  `actionSpec` из draft-state и добавляет жест.
- prompt injection marker -> guardrails block.

## Definition of done для следующих шагов

Каждый шаг должен:

- менять только scoped MAS/agent/UI файлы;
- иметь unit-тесты или проверочный сценарий;
- сохранять `tests/unit/test_flet_binding_agent.py`;
- не коммитить посторонние изменения из рабочей директории;
- обновлять этот audit или отдельный architecture doc, если меняется контракт.
