# Scenario Planning

## Purpose

Convert a natural language routine into a GestureBind `sequence` action.

## Inputs

- User phrase.
- Optional scenario name.
- Ordered step clauses.
- Existing draft-state from dialog memory.

## Behavior

- Extract names from phrases like `сценарий под названием мое утро`.
- Split steps by explicit markers: `первый шаг`, `второе`, `потом`,
  `затем`, `далее`, semicolon, and command-like commas.
- Parse each step with the macOS action skill.
- Require at least two understandable steps for a scenario.
- If the scenario is clear but gesture is missing, ask only for the gesture.

## Output Contract

Return an action spec:

```json
{"action": "sequence", "platform": "macos", "name": "...", "steps": []}
```

