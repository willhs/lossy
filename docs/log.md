---
id: log
type: reference
purpose: "Append-only record of docs/ operations (ingests, large reorganisations, schema changes) — gives future agents a chronological breadcrumb trail."
scope: ["docs", "meta"]
tags: ["log", "ingest", "history"]
related: ["index.md"]
---

# Docs Log

Append-only. Newest entries at the bottom. Format:

```
## [YYYY-MM-DD] <op> | <title>
<one-paragraph note: what happened, paths touched, raw source if relevant>
```

Operations: `ingest`, `reorg`, `schema`, `prune`.

---

## [2026-04-18] ingest | AI Video Generation Speed Benchmarks 2025 (Apatero)
Ingested third-party benchmark blog post into `research/0017-video-gen-speed-benchmarks/research.md`. Raw source: `sources/ai-video-generation-speed-benchmarks-2025.md`. Covers LTX-2 vs Wan 2.2 vs Kling/Runway/Pika across local GPUs and cloud rental — relevant to `strategies_video.py` backend choices and the concurrent-runpod work.
