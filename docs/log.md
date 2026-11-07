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

## [2026-09-06] update | Full-run go/no-go decided: GO, gated on defects 5 and 6

The `[HUMAN] Decide whether the remaining ~2000 shots proceed` subtask had been open
since 2026-08-23 and was the project's actual critical path — the 2026-09-06 dream
report flagged it as the bottleneck for the second night running.

Decided: **go**. Defects 5 (identity bleed with >=3 stacked characters, 196/2069 shots)
and 6 (non-humanoid characters rendering as people in costume) land first — both are
small prompt/registry changes covering 9.5% of shots. Defect 7 (~1.5% speaker-attribution
error floor) and defects 11/12 (2.3% ambience failures, 98.8% FS mux peak) are accepted
as known v1 limits; 11/12 get revisited before mastering, not before generating.

Board reflects this: `0b8d08c0` moved idea → week and rescoped to just 5 + 6, with the
full run (`0725e309`) now declaring a dependency on it.

## [2026-09-12] update | Defects 5 + 6 landed, full run unblocked

`aac8ee2` fixed both defects gating the 2026-09-06 GO decision: `CharacterIdentityMixin`
now caps identity blocks to the two most prominent characters per shot (defect 5,
identity bleed on 196/2069 shots), and `IDENTITY_DESCRIPTION_SPEC` now describes
droids/robots as mechanical rather than "humanoid" (defect 6). Both spot-checked
against the real `star_wars_iv_v2` `characters.json`. `docs/work/roadmap.md` updated
to move this out of "Now" into "Done" — resuming the full run for the remaining
~2000 shots is the project's only remaining "Now" item.

## [2026-10-09] update | AGENTS.md is the single agent instructions file

`ab9775d` removed the `CLAUDE.md` and `docs/CLAUDE.md` stubs; `AGENTS.md` (a symlink to `.gent/rules.md`) is now the only agent instructions file. `docs/design/architecture.md` now points at it. References to `CLAUDE.md` in `docs/research/0024` and `docs/tasks/*/plan.md` are historical and left as written.
## [2026-10-15] reorg | CLAUDE.md retired in favour of AGENTS.md
Commit ab9775d (2026-10-09) removed `CLAUDE.md` and `docs/CLAUDE.md`; `AGENTS.md` (-> `.gent/rules.md`) is now the single agent instructions file. Updated the live reference in `design/architecture.md`. Historical references in `research/0024` and `tasks/0012`, `0014`, `0016` left as-is (point-in-time records).

## [2026-11-07] ingest | Video model refresh: LTX-2.5 recommended, Seinfeld S01E03 first
`research/0025-video-model-refresh-nov-2026/` surveys models since Sep 2026 and pilots LTX-2.3 vs LTX-2.5 vs Kandinsky 6 on a fixed 20-shot hard benchmark (spend $2.29 of a $10 cap, 0 pods left). Blind scores: LTX-2.3 2.1, LTX-2.5 2.7-2.8, K6 2.9 (but ~3x cost, 480p). Recommends a `runpod-ltx25` strategy (two-stage distilled, conv VAE: ~$15/film, ~$2.70/episode community) and Seinfeld S01E03 as the first end-to-end run (~$4.50-7). Harness in `tools/model_pilot/`. Also found: 690 shots (690-1379) already rendered on LTX-2.3 in Aug, pre defect-5/6; the encode's `keyframes/` dir is misnumbered from ~#405.
