# Binding MAS layout

This package contains the GestureBind binding multi-agent system. The public
entry point remains `app.services.binding_agent.BindingAgentOrchestrator`.

## Request path

`guardrails -> session memory -> TaskFrame/intent -> CRUD, answer, or binding
specialists -> executable skill validation -> optional research/Mistral ->
output guardrails -> reviewer -> background MLflow`.

## Files

- `contracts.py`: typed TaskFrame, candidates, context, steps and results.
- `task_frame.py`, `semantic_router.py`, `intent.py`: hybrid routing.
- `session_memory.py`: task-scoped slots with continuation/correction handling.
- `tools.py`, `action_semantics.py`: gesture/action parsing and arbitration.
- `binding_pipeline.py`: thin specialist agents.
- `binding_crud.py`: inspect/list and confirmation-only delete mutations.
- `skills/runtime.py`: executable, versioned action skills.
- `research.py`: source-backed research and approval-safe skill memory.
- `mistral.py`: optional structured parser and answer rewrite adapter.
- `guardrails.py`, `reviewer.py`: safety and quality gates.
- `eval_cases.py`, `e2e_judge.py`: golden data and ten-criterion judge.

New gesture aliases and research recipes are proposals until the user
successfully saves the binding. Raw dialog history is not an executable-memory
source. See `docs/GESTUREBIND_MAS_ARCHITECTURE.md` for the full contract.
