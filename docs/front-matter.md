---
id: front-matter
type: reference
purpose: "Describe the doc front matter schema and how to use it."
scope: ["docs"]
non_goals: []
tags: ["front-matter", "schema"]
related: ["front-matter-schema.json"]
---

## Purpose
Explain the required YAML front matter format for persistent docs and point to the schema for validation.

Use the JSON Schema in `front-matter-schema.json` to validate doc headers before committing changes.

## Types
- **spec**: Defines intended behaviour, interfaces, or constraints that guide implementation work.
- **decision**: Records a significant choice and the rationale behind it so it can be revisited later.
- **note**: Captures context, research, or running notes that do not fit a formal specification.
- **runbook**: Documents operational procedures, recovery steps, and other on-call guidance.
- **reference**: Summarises stable facts, definitions, or canonical lists that are consulted frequently.

## Examples
```yaml
---
id: customer-interviews
type: note
purpose: "Capture insights from the latest customer interviews."
tags: ["research"]
related: []
---
```

```yaml
---
id: billing-retries
type: spec
purpose: "Define the retry policy for failed billing runs."
scope: ["payments", "scheduler"]
non_goals: ["UI changes"]
---
```
