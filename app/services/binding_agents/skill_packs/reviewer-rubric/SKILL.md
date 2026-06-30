# Reviewer Rubric

## Purpose

Evaluate whether the final agent result fits the selected intent block.

## Checks

- Intent relevance.
- Binding contract completeness.
- Safety and guardrail compliance.
- UI usefulness: the user sees next action, not internal implementation noise.
- Tone of voice: warm, GestureFlow-specific, not dry or generic.

## Verdicts

- `ok`: answer or draft can be shown.
- `need_clarification`: answer is useful but requires a missing field.
- `blocked`: answer violates safety, scope, or contract rules.

## Metrics

Log reviewer scores for relevance, contract completeness, safety, tone, and
clarification quality.

