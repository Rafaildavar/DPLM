# GestureBind MAS skill packs

Skill packs are filesystem instructions for the binding multi-agent system.
They complement the Python `skills/` trace registry:

- Python skills are compact cards attached to trace rows.
- Filesystem skill packs are richer instructions that can be loaded into
  routers, reviewers, evaluators, or LLM prompts.

Every pack uses a `SKILL.md` entrypoint so new capabilities can be added
without hiding behavior inside one large prompt.

