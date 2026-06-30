# macOS Actions

## Purpose

Map natural language to a safe macOS action specification.

## Supported Actions

- `open_app`
- `open_url`
- `open_path`
- `key_combination`
- `press`
- `wait`
- `notify`
- `media_key`
- `volume_up`
- `volume_down`
- `mute_toggle`
- `brightness_down`
- `brightness_up`
- `lock_screen`
- `screenshot`
- `run_script`

## Rules

- Prefer structured action specs over free text.
- Keep `platform` as `macos`.
- For hotkeys, normalize keys to lowercase canonical names.
- For common products, normalize aliases such as Jira, Notes, ChatGPT, and
  Rambler Mail.
- If action is ambiguous, ask a clarification instead of guessing.

## Output Contract

Return `action_spec` with only fields required by the action executor.

