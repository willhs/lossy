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

## [2026-11-07] code | `runpod-ltx25` strategy
`RunPodLtx25Strategy` (`strategies_video.py`): LTX-2.5 distilled int8, two-stage + conv VAE, per research 0025; registered in `config.py` and `decode.py`. Pod setup upgrades ComfyUI to >= v0.32 (tag v0.39.2) and downloads weights from the `comfyicu/LTX-2.5` mirror pinned by revision and sha256 (`RunPodSession.download_pinned_models`, `.part` then verify then move). Frame counts are 8k+1. Fixed a shared bug found by the live smoke test: `_attempt_one_clip` deleted any clip the pod returned as `.mp4` (raw path equalled clip path). New `tools/seam_ratio.py` for the 0023 chaining measurement. Smoke-test results: `research/0025-video-model-refresh-nov-2026/smoke-test.md`.

## [2026-11-07] cleanup | Docs tidy after the weekend burn
Folded research 0025, `runpod-ltx25` and the SECURE-rate fix into `design/architecture.md` (strategy table; `runpod-ltx2` marked superseded), `ops/runbook.md`, `work/roadmap.md` (new Now item, Done entry) and `README.md`/`docs/index.md` (Wan/LTX-2 defaults flagged as superseded). Note: no Seinfeld S01E03 run results exist yet — research 0025 only recommends it as the first run; roadmap tracks it as pending. Historical research notes 0012/0017/0022/0023 left as point-in-time records.

## [2026-11-07] lint | Docs audit
0 broken inline links; 16 pre-existing frontmatter findings across 93 files (13 non-schema `type`/`id` values in research/experiments/design, 3 files without frontmatter) left for a human decision.

## [2026-11-08] research | 0028 style drift
`research/0028-style-drift/` classifies all 427 clips of the Seinfeld S01E03 runpod-ltx25 run (Gemini 3.1 flash-lite, ~$0.18, no GPU) and finds the photoreal/cartoon drift is driven by Kramer's identity block (55/62 shots cartoon vs 25/237 without; 3/15 when the 2-character cap drops it), plus stacked Elaine+George blocks. No prompt carries a live-action anchor; the negative prompt is inert at LTX-2.5's cfg 1.0; seed, length, pod and chaining are not drivers (chained parts never flip). Proposes a per-film style prefix and photographic identity-block spec, and a 21-clip ~$0.44 A/B (`ab_manifest.json`) awaiting Will's approval.
