---
id: adr-002-stateless-cli
type: decision
purpose: "Record the decision to build lossy as stateless CLI commands rather than a server or workflow engine."
scope: ["design", "architecture"]
non_goals: []
tags: ["adr", "architecture"]
related: ["design/architecture.md"]
---

# Context

The pipeline has three stages (encode, decode, compare). We need to decide how they're orchestrated — as independent CLI commands, a single monolithic script, a web service, or a workflow engine (e.g. Airflow, Prefect).

# Decision

Each stage is a standalone CLI command. Stages communicate via files on disk (scene manifest JSON, video clips). No server, no database, no orchestrator.

# Consequences (Positive/Negative)

**Positive:**
- Dead simple to run, debug, and resume — re-run any stage independently.
- No infrastructure to maintain.
- Easy to manually intervene between stages (edit the manifest, swap clips, etc.).
- Aligns with the project's one-off experiment nature.

**Negative:**
- No automatic retries or progress tracking for long decode runs.
- No parallelism built in (could be added per-stage without changing the overall model).
- User must manually chain stages.

# Alternatives Considered

- **Single script**: Less flexible — can't re-run decode without re-encoding.
- **Web service**: Unnecessary complexity for a single-user experiment.
- **Workflow engine**: Massive overkill. The pipeline has three steps.

# Links

- [Architecture overview](../architecture.md)
