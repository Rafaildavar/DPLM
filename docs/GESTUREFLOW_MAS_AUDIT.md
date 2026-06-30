# GestureFlow MAS audit

Дата: 2026-06-30

Этот документ фиксирует текущую точку отсчета для многоагентной системы
привязок GestureFlow. Он нужен, чтобы каждый следующий шаг улучшения можно
было сверять с планом, тестами и пользовательскими сценариями.

## Текущий pipeline

Публичный вход остается в `app.services.binding_agent.BindingAgentOrchestrator`.
UI вызывает совместимый wrapper `build_agent_binding_draft`, а результат
возвращается в legacy-формате для Flet-экрана привязок.

Порядок обработки:

1. `Guardrails Agent` проверяет вход: длина, секреты, prompt injection.
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
5. `Guardrails Agent` проверяет выход.
6. `Reviewer Agent` оценивает соответствие ответа intent-блоку.
7. `BindingAgentMlflowLogger` пишет trace, параметры, метрики, artifacts и
   optional GenAI eval.

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

## Главные проблемы перед улучшением

1. Intent routing пока в основном rule-based. Он быстрый, но плохо понимает
   свободные формулировки. Базовый локальный semantic-router уже добавлен,
   следующий этап - заменить/дополнить его настоящими embeddings.
2. Skills сейчас существуют как Python-каталог для trace, но еще не оформлены
   как самостоятельные skill packs с `SKILL.md`.
3. Tools получили базовый contract-слой. Следующий этап - расширить его
   typed-схемами для LLM/function calling и MLflow eval.
4. Memory хранит диалог, но еще не держит полноценный draft-state для
   продолжения сценария через уточнения.
5. Guardrails уже разделены на input/output, но политика отказов и уточнений
   требует явной v2-модели.
6. Reviewer проверяет релевантность, но ему нужны более строгие рубрики:
   contract completeness, safety, tone, UI usefulness.
7. MLflow логирует pipeline, но еще не отражает semantic scores, rule hits и
   reviewer-rubric metrics.

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
- prompt injection marker -> guardrails block.

## Definition of done для следующих шагов

Каждый шаг должен:

- менять только scoped MAS/agent/UI файлы;
- иметь unit-тесты или проверочный сценарий;
- сохранять `tests/unit/test_flet_binding_agent.py`;
- не коммитить посторонние изменения из рабочей директории;
- обновлять этот audit или отдельный architecture doc, если меняется контракт.
