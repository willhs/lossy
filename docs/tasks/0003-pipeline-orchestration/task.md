---
id: task-0003
type: spec
purpose: "Add a single CLI command that chains encode → decode → stitch automatically, eliminating manual stage chaining."
tags: ["cli", "pipeline", "orchestration"]
related: ["docs/design/adr/002-stateless-cli-pipeline.md", "docs/design/architecture.md"]
created: 2026-03-16
updated: 2026-03-16
---

# Pipeline Orchestration Command

## Goal

Add a `pipeline` (or `run-all`) CLI command that runs the full lossy pipeline — encode stage1, encode stage2, decode, stitch — in sequence with a single invocation. Each stage's existing resume/skip logic should still apply, so a failed run can be re-invoked to pick up where it left off.

## Context

Currently the user must manually chain four commands:

```bash
python encode.py stage1 media/film.mp4 -o output/film
python encode.py stage2 output/film
python decode.py output/film --strategy fal-seedance
python decode.py output/film --strategy fal-seedance --stitch
```

This is fine for development and debugging, but for a full-film run it's friction. ADR-002 chose stateless CLI commands over an orchestrator — this task adds a thin convenience layer on top without changing that model. Each stage remains independently runnable; the pipeline command simply calls them in order.

## Scope

### Must Do
- New CLI entry point (`pipeline.py` or similar) that accepts a source video, output dir, and strategy, then runs all four stages in order
- Stop on first stage failure with a clear error message indicating which stage failed
- Pass through relevant options to each stage (e.g., `--strategy`, `--limit`, detection method)
- Print a summary at the end (stages completed, total time, output paths)

### Might Do
- `--skip` flag to skip specific stages (e.g., `--skip encode` to re-run decode+stitch only)
- `--dry-run` to print the commands that would be executed without running them

### Won't Do
- Parallel execution of stages (they're sequential by nature)
- Web UI or daemon mode
- Changing how individual stages work internally
- Adding a package manager or CLI framework (keep it as standalone scripts)

## Constraints

- Must work with the existing Python 3.11 venv and no new dependencies
- Must preserve ADR-002's principle: each stage remains independently runnable
- Must work with all existing strategies (replicate-wan, fal-seedance, fal-seedance-pro, runpod-wan)

## References

- [ADR-002: Stateless CLI Pipeline](../../design/adr/002-stateless-cli-pipeline.md) — the pipeline command is a convenience layer, not a replacement for independent stages
- [Architecture overview](../../design/architecture.md)

## Success Criteria

- [ ] Single command runs full pipeline from source video to reconstructed output
- [ ] Each stage's existing resume/skip logic works when pipeline is re-run
- [ ] Failure in any stage stops the pipeline with a clear error
- [ ] All existing strategies work through the pipeline command
- [ ] Existing stage commands continue to work independently (no regressions)
