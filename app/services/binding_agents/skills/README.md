# Binding MAS skills

Skills are intentionally stored separately from agents.

This mirrors the common MAS layout:

- agents own role behavior and orchestration steps;
- skills describe reusable capabilities/tools that agents can attach to trace rows;
- `runtime.py` selects executable skills and validates their action contracts;
- the registry aggregates skills for UI, MLflow traces, and reviewer context.

Files:

- `base.py` defines the `AgentSkill` contract.
- `guardrails.py` describes input/output safety skills.
- `intent.py` describes routing skills.
- `answer.py` describes project-answer and safe-redirect skills.
- `binding.py` describes gesture, memory, action, research, scenario, policy, and validation skills.
- `review.py` describes reviewer skills.
- `registry.py` exposes `AGENT_SKILLS`, `AGENT_DEFAULT_SKILLS`, and trace helpers.
- `runtime.py` defines versioned actions, OS support, permissions, risk and timeout.

Trace cards are descriptive. An action is actionable only when the executable
runtime selects a matching skill and `validate_action_spec` accepts its schema.
