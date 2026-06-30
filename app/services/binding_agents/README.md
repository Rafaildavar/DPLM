# Binding MAS layout

The binding assistant is organized as a small multi-agent system:

- `intent.py` routes requests into project questions, binding changes, and unsupported general questions.
- `guardrails.py` owns input and output safety checks.
- `binding_pipeline.py` contains deterministic binding specialists: gesture, memory, action, scenario, policy, and validation.
- `mistral.py` is the optional model-backed draft adapter.
- `reviewer.py` checks answer relevance before the result reaches the UI.
- `skills/` contains the reusable skill catalog that agents attach to trace rows.

`app.services.binding_agent` remains the public compatibility facade and keeps shared contracts, parsing helpers, MLflow logging, and the orchestrator. It imports the skill registry from `skills/`, which avoids UI import churn while keeping concrete agent roles and reusable capabilities separated.
