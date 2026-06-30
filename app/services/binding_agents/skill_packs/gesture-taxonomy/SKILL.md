# Gesture Taxonomy

## Purpose

Resolve Russian and English gesture phrases to GestureFlow class labels.

## Inputs

- Gesture labels from the database.
- User text.
- Current selected gesture.
- Recent dialog memory.

## Behavior

- Normalize Russian/English synonyms: `свайп вверх`, `swipe up`, `up`.
- Prefer exact known labels over typed unknown labels.
- If no exact match exists, return similar known labels and ask the user to pick.
- Use current selected gesture only when the message is clearly about binding.
- Never invent a gesture when the user asks a project or unrelated question.

## Output Contract

- `gesture`: resolved label or empty string.
- `source`: `prompt`, `similar`, `memory`, `selected`, or `typed`.
- `known`: whether the label exists in the current gesture list.
- `suggestions`: similar labels when clarification is needed.

