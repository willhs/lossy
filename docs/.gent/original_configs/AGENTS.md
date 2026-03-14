---
id: agent-guidelines-docs
type: reference
purpose: "Agent-specific guidelines for contributing to the docs/ directory."
scope: ["documentation", "agent-workflows"]
non_goals: []
tags: ["agents", "documentation", "guidelines"]
related: ["index.md", "../AGENTS.md"]
---

## Purpose
This document outlines what agents can and cannot do when contributing to docs/.

**Agents can edit freely**: `docs/work/tasks/`, `docs/research/experiments/`, `docs/work/notes/`

**Agents need approval**: `docs/philosophy/`, `docs/design/`, `docs/design/adr/`, `docs/ops/`, `docs/meta/`

**Always use templates** when creating new documents: `adr.md`, `task.md`, `plan.md`, `experiment.md`, `vision.md`

**Validate front matter** against `docs/meta/front-matter-schema.json` before committing.

See `docs/index.md` for the full documentation guide.
