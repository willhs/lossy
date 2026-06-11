---
id: principles
type: reference
purpose: "Define the guiding principles for product and engineering decisions."
scope: ["philosophy"]
non_goals: []
tags: ["principles"]
related: ["philosophy/vision.md"]
---

## Purpose
Enumerate principles that guide decisions when goals and data leave room for interpretation. Listed roughly in priority order — when two principles collide, the one above wins.

## Principles

### 1. Comically lossy, but as good as it can be.
lossy compresses films into a tiny number of bytes of text and reconstructs them with AI video. The result is unavoidably degraded — that's the joke and the point — but within that constraint, every byte should carry its weight. Push on prompt quality, encoder fidelity, model choice, and reference conditioning. Do *not* fight the lossiness itself with upscaling, interpolation, or post-hoc smoothing: the interesting artifact is what survives the round-trip through language, not how cleanly we can paper over what didn't.

### 2. Cost first.
Every external API call costs real money, and a full-film decode runs hundreds to thousands of calls. Defaults pick the cheapest viable option: 480p, short clips, fewest frames sampled, smallest model that produces any output. "Cheaper and visibly worse" beats "pricier and a bit nicer". Before adding a backend or feature, state what a full-film run costs with it on.

### 3. Stateless CLI stages (ADR-002).
Stages communicate by files in the per-film output directory — nothing else. No shared database, no long-running server, no in-memory state that survives a CLI invocation, no cross-stage Python imports for runtime data. `pipeline.py` is a subprocess chainer, not an importer. This is how the project stays resumable, debuggable from the file tree, and easy to parallelize.

### 4. One source of truth for the on-disk contract.
`manifest.py` owns filename conventions and the `shots.json` schema. Readers go through its helpers (`load_shots`, `clips_dir`, `clip_filename`, ...) instead of reinventing the check. When the contract changes, it changes in exactly one place and every stage picks it up. If you find yourself writing `f"{idx:04d}.mp4"` or re-checking `format == "v2"` inline, stop and add a helper.

### 5. Single-shot experiment, not a platform.
lossy is a blog-post project: encode a handful of films, compare results, write up what was learned. It is not a product, not a service, and has no external consumers. Do not design for multi-tenant use, plugin ecosystems, hypothetical future strategies, or backwards compatibility with old output directories. When in doubt, delete the abstraction and hardcode the thing. The one load-bearing extension point is the strategy pattern for video/audio backends — that earns its keep because swapping backends is the core of the experiment.
