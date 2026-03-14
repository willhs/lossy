---
id: adr-001-scene-manifest
type: decision
purpose: "Record the decision to use a JSON scene manifest as the intermediate format between encode and decode."
scope: ["design", "architecture"]
non_goals: []
tags: ["adr", "format"]
related: ["design/architecture.md"]
---

# Context

The pipeline needs an intermediate representation — the "compressed" form of the film that sits between encoding and decoding. Options range from a simple text file to a structured data format to a database.

# Decision

Use a single JSON file (the "scene manifest") as the intermediate format. It contains metadata about the source and an ordered list of scene objects, each with timestamps, text description, dialogue, mood, and camera notes.

# Consequences (Positive/Negative)

**Positive:**
- Human-readable and editable — scenes can be manually tweaked before decoding.
- Easy to version control and diff.
- No dependencies (no database, no binary format).
- Naturally supports the "manual encoding" path where someone writes descriptions by hand.

**Negative:**
- No binary data — can't embed reference frames or audio samples in the manifest itself.
- JSON has no schema enforcement at runtime without extra tooling.
- Large films with many scenes could produce unwieldy files (unlikely to matter in practice).

# Alternatives Considered

- **Plain text (one description per line)**: Too unstructured — loses timestamp and metadata.
- **SQLite database**: Overkill for a linear sequence of scenes with no relational queries.
- **YAML**: Viable but JSON is more universal for tooling and less ambiguous to parse.

# Links

- [Architecture overview](../architecture.md)
