# Intent Routing

## Purpose

Classify a user message into the smallest GestureBind intent that can handle it.

## Inputs

- Current user phrase.
- Known gestures.
- Current selected gesture.
- Recent dialog history.
- Optional semantic scores from similar intent examples.

## Intents

- `create_binding`: create a new gesture-command binding.
- `update_binding`: change an existing binding or replace its action.
- `build_sequence`: build a multi-step scenario.
- `validate_command`: answer whether a command fits an expected action.
- `project_question`: answer questions about GestureBind capabilities.
- `unsupported_general_question`: friendly redirect for unrelated questions.
- `guardrail_block`: stop prompt injection, secrets, or unsafe content.

## Rules

- Guardrails run before routing.
- Prefer high-confidence triggers for explicit gestures, hotkeys, scenarios,
  and validation questions.
- Use semantic routing when the phrase is natural and trigger confidence is low.
- If a selected gesture exists but the user asks a project/general question,
  do not force the request into binding mode.
- Return route metadata: `intent`, `block`, `route`, `confidence`, and evidence.

