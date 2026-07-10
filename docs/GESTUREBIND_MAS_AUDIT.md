# GestureBind MAS audit

Дата: 2026-07-09

## Итог текущей итерации

Многоагентный pipeline переведён с набора связанных эвристик на contract-first
архитектуру. Введены typed TaskFrame/ActionCandidate/Result, semantic verifier,
исполняемые skills, CRUD-агент, task-scoped memory, общий latency budget,
privacy-фильтр, независимый golden dataset и LLM-as-a-judge.

Зафиксированные этапы:

- `40e98e5` - TaskFrame contracts;
- `d88b900` - semantic action compiler/verifier;
- `fe01b8f` - executable skills, CRUD and approval lifecycle;
- `fcdf6a2` - structured session memory;
- `e8a6a75` - latency budget, privacy and asynchronous MLflow;
- `b57c686` - 32-case golden evaluation and LLM judge.

## Проверенные свойства

- Exact gesture label beats a colliding alias.
- Explicit user gesture beats an incorrect model gesture.
- `закрыть Telegram` compiles to `quit_app`, never `open_app`.
- Unsupported/abstract actions abstain instead of inventing an application.
- Sequences preserve every resolved step and reject partial silent saves.
- New tasks do not inherit stale gesture/action slots.
- Explicit continuation and `не X, а Y` correction work on the active task.
- Custom aliases and researched recipes persist only after successful save.
- Delete is a confirmation-only mutation; inspect/list are read-only.
- Research runs at most once per request.
- Credentials are redacted before Mistral and MLflow.
- Runtime GenAI evaluation uses independent golden expectations.

The deterministic 10-criterion evaluation currently reports `32/32` passing
cases with mean score `1.000`:

```bash
DPLM_BINDING_AGENT_MLFLOW=0 \
  python scripts/evaluate_binding_agent_mas.py --judge local
```

Full project verification on 2026-07-09:

```text
593 passed, 2 skipped in 45.42s
coverage: 61.42% (required: 59.8%)
```

## Remaining risks

Passing tests do not make incorrect behavior impossible. Current residual risks:

1. The semantic router is a small local character-TF-IDF index, not a
   multilingual production embedding model trained on real user traffic.
2. Rule/action ontology still needs new golden cases as users introduce new
   wording, applications and macOS capabilities.
3. Session memory is process-local and TTL-based; it is not durable across app
   restart and does not yet support several parallel drafts in one UI session.
4. Live web research is intentionally narrow and extracts only allowlisted
   Apple recipes; arbitrary workflows require a safer structured tool catalog.
5. A real Mistral judge is optional and costs network time/money. CI uses the
   deterministic rubric, so periodic external evaluation is still necessary.
6. MLflow is asynchronous. A hard process kill can lose the last queued event;
   graceful app shutdown should call logger flush in a future lifecycle hook.
7. UI delete confirmation is backed by controller tests and service tests, but
   desktop interaction should also be covered by screenshot/interaction tests.
8. All executable skills currently target macOS; another OS needs separate
   manifests, validators and golden cases.

## Next evidence to collect

- anonymized intent/action phrases from real beta sessions;
- confusion/margin distribution for semantic routing;
- p50/p95 cold and warm pipeline latency with live Mistral/MLflow;
- approval/rejection rate for research proposals;
- periodic real-LLM judge reports stored in MLflow;
- end-to-end Flet interaction coverage for save/delete/confirmation.

The detailed design and extension checklist live in
`docs/GESTUREBIND_MAS_ARCHITECTURE.md`.
