# GestureBind MAS architecture

Дата: 2026-06-30

Документ описывает текущую архитектуру многоагентной системы привязок
GestureBind и правила безопасного расширения.

## Цель системы

Агент привязки принимает обычную фразу пользователя и возвращает один из двух
результатов:

- структурированный draft привязки жеста к macOS action или sequence;
- markdown-ответ по проектному/внешнему вопросу без создания привязки.

## Главный pipeline

```text
UI prompt
  -> BindingAgentOrchestrator
  -> Input Guardrails
  -> Intent Agent
       rule route
       semantic route
       optional Mistral structured route
  -> Binding agents or Conversation answer
  -> Output Guardrails
  -> Reviewer Agent
  -> MLflow trace/eval metrics
  -> Flet draft/result UI
```

## Директории

- `app/services/binding_agent.py` - public facade, shared contracts,
  orchestrator, MLflow logger, compatibility wrapper.
- `app/services/binding_agents/intent.py` - hybrid intent router.
- `app/services/binding_agents/semantic_router.py` - local semantic routing.
- `app/services/binding_agents/tools.py` - reusable tool contracts.
- `app/services/binding_agents/binding_pipeline.py` - concrete binding agents.
- `app/services/binding_agents/guardrails.py` - input/output guardrails v2.
- `app/services/binding_agents/reviewer.py` - evaluator/reviewer rubric.
- `app/services/binding_agents/mistral.py` - optional model-backed adapter.
- `app/services/binding_agents/skill_packs/*/SKILL.md` - filesystem skills.
- `app/services/binding_agents/eval_cases.py` - regression/eval dataset.
- `tests/unit/test_flet_binding_agent.py` - main regression suite.

## Intent contract

Every routed request must produce:

- `intent`;
- `block`;
- `route`;
- `routeMethod`: `rule`, `semantic`, or `fallback`;
- optional `semanticIntent`, `semanticScore`, `semanticExample`.

Supported intents:

- `create_binding`;
- `update_binding`;
- `build_sequence`;
- `validate_command`;
- `project_question`;
- `unsupported_general_question`;
- `guardrail_block`.

## Tool contract

Every tool returns `AgentToolResult`:

- `tool`;
- `status`;
- `message`;
- `payload`.

Current tools:

- `resolve_gesture`;
- `parse_macos_action`;
- `build_sequence_action`;
- `validate_binding_contract`;
- `review_answer_contract`.

When adding a tool:

1. Put reusable logic in `tools.py` or a dedicated tools module.
2. Keep agent classes thin: agent `run()` should translate tool result to
   `AgentStep`.
3. Add a unit test for the tool contract.
4. Attach or update a trace skill in `binding_agents/skills/`.
5. Add or update a `SKILL.md` if the behavior changes agent reasoning.

## Skill pack rules

Filesystem skill packs are instructions, not executable tools.

When adding a skill pack:

1. Create `app/services/binding_agents/skill_packs/<pack-id>/SKILL.md`.
2. Add `<pack-id>` to `EXPECTED_SKILL_PACKS`.
3. Include purpose, inputs, behavior, and output contract.
4. Add or update tests that check pack discovery.
5. Keep user-facing tone in `tone-of-voice/SKILL.md`.

## Guardrails rules

Guardrails v2 uses decisions:

- `allow` - result can proceed;
- `clarify` - result is safe but needs missing information;
- `block` - result must not be shown as actionable.

Missing gesture/action is always `clarify`, not refusal.

## Reviewer metrics

Reviewer writes:

- `relevance`;
- `contract_completeness`;
- `safety`;
- `tone`;
- `clarification_quality`.

MLflow logs these as `reviewer_*` metrics together with
`intent_semantic_score`.

## Eval workflow

Before committing MAS changes, run:

```bash
python -m pytest tests/unit/test_flet_binding_agent.py -q --no-cov
```

When adding an intent, scenario, guardrail rule, or parser behavior:

1. Add a case to `BINDING_AGENT_EVAL_CASES`.
2. Add a focused unit test if the behavior is subtle.
3. Check MLflow fields if the trace contract changed.
4. Update `docs/GESTUREBIND_MAS_AUDIT.md` when the current state changes.

## Mistral usage

Mistral is optional. The local deterministic pipeline remains the source of
truth for tests. When Mistral is enabled, prompts include:

- intent/block;
- known gestures;
- selected gesture;
- dialog memory;
- current draft-state;
- local task paraphrase.

Model output must still pass policy, output guardrails, and reviewer checks.
