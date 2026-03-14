# Agent Guidelines

## Workflow
- Prefer worktrees or task-specific branches for multi-file edits.
- Run tests and linters before handing work back.

## Documentation
- Preserve doc front matter (`id`, `type`, `purpose`, scope, tags, etc.) and keep sections aligned with the docs guide.
- Validate doc front matter against `docs/front-matter-schema.json` if present.
- For docs-specific workflows and permissions, see `docs/AGENTS.md` (details on when to write to different doc directories).

## Guardrails
- Respect human-owned files flagged in `docs/philosophy/`.
- Keep generated content ASCII unless the project explicitly opts in.
