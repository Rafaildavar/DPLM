# Binding MAS layout

The binding assistant is organized as a small multi-agent system:

- `intent.py` routes requests into project questions, binding changes, and unsupported general questions.
- `guardrails.py` owns input and output safety checks.
- `binding_pipeline.py` contains deterministic binding specialists: gesture, memory, action, scenario, policy, and validation.
- `research.py` owns the source-backed Research Agent and user-approved action skill memory.
- `mistral.py` is the optional model-backed draft adapter.
- `reviewer.py` checks answer relevance before the result reaches the UI.
- `e2e_judge.py` defines the 10-criterion LLM-as-a-judge rubric for
  end-to-end agent checks. It can use a real completion function, while tests
  use the deterministic local fallback so CI does not depend on API keys.
- `skills/` contains the reusable skill catalog that agents attach to trace rows.

Research results follow an approval loop: a recipe can fill a draft, but it is
written to `data/binding_agent/skill_packs/researched-actions/SKILL.md` only
after the user approves it through the UI.
Live web research is allowlisted and opt-in via `DPLM_BINDING_RESEARCH_WEB=1`;
memory and source-backed local recipes are available without network access.

The orchestrator is contract-first by default:

- `DPLM_BINDING_AGENT_LOCAL_FIRST=1` means deterministic agents try to build
  the draft before Mistral is called.
- If the local contract is complete, Mistral is skipped even when
  `MISTRAL_API_KEY` exists.
- If the local contract misses an action or hits validation trouble, Mistral can
  be used as fallback and the trace records the fallback reason.
- `DPLM_BINDING_AGENT_REWRITE_ANSWERS=1` enables model paraphrasing for answer
  mode; otherwise project/general answers stay local and fast.
- `DPLM_BINDING_AGENT_GENAI_EVAL=1` enables `mlflow.genai.evaluate`; by default
  the UI hot path logs traces and metrics without running scorer evaluation.

Research is split into visible stages:

- `Research Query Planner` classifies the lookup and records whether live web
  research is allowed.
- `Research Agent` checks approved memory, source-backed recipes, then optional
  allowlisted Apple web search.
- `Research Recipe Validator` rejects recipes without an action, wrong platform,
  or missing source evidence when approval is required.
- `Skill Memory Writer` writes only user-approved recipes back to the persisted
  researched-actions skill pack.

`app.services.binding_agent` remains the public compatibility facade and keeps shared contracts, parsing helpers, MLflow logging, and the orchestrator. It imports the skill registry from `skills/`, which avoids UI import churn while keeping concrete agent roles and reusable capabilities separated.
