---
id: 0021-training-data-contamination
type: note
purpose: "Establish whether the models in the pipeline reconstruct films from the compressed text or partly from memorised training data, and what that means for the lossy-codec claim."
scope: ["research", "encode", "describe-pass", "evaluation", "benchmark-validity"]
non_goals: []
tags: ["research", "contamination", "world-knowledge", "gemini", "benchmark", "character-identity"]
related: ["research/0020-wan22-identity-i2v-verification/research.md", "research/0011-character-continuity/research.md"]
---

## Question

The project's claim is that a film is compressed to under 1MB of text and regenerated from that text. If the models already know the film, the reconstruction is not text alone — it is text plus whatever the encoder and decoder remember. Two questions follow:

1. Does world knowledge measurably enter the pipeline, and where?
2. Is the effect specific to famous films, making Star Wars a poor benchmark, or is it general?

## Finding 1: world knowledge overrides visual evidence, not just annotates it

In `sw_r2_leia`, shot 6 is the Leia hologram. The describe pass wrote:

> "A blue, translucent holographic figure of **Obi-Wan Kenobi**, wearing a hooded robe…"

The figure is visibly female. No dialogue transcript was present in that run, so the model had only the sampled frames and its own knowledge. It recognised the iconic hologram-plea moment and described the scene it remembered rather than the frames it was given.

The prior baseline encode of the same clip wrote "a blue, translucent holographic figure of **a woman** in a hooded robe". Both descriptions contain "hooded robe"; the **only** meaningful difference is the name.

Downstream, the reconstruction of shots 6–7 renders a dark hooded figure that reads as Darth Vader (roughly 26–40s in
`output/comparisons/sw_r2_leia_before_after_vstack.mp4`, ~15s of screen time — the longest continuous stretch in the clip). A single wrong noun phrase in the compressed representation propagated into a visibly wrong character.

This is the important distinction: world knowledge did not add a label on top of an accurate description, it **replaced** the description.

## Finding 2: the previous-shot context feature makes wrong names sticky

Commit `98a3154` threads the previous shot's description into the next shot's describe call for continuity. Shot 7's description came out **byte-identical** to shot 6's — the feature working exactly as designed, and therefore preserving the error for the whole beat.

The mechanism that prevents drift also propagates mistakes. Tracked as a fix in the sticky-names task; the leading candidate is to carry setting, lighting and style continuity in the context but **not** character identity, since identity already has its own locked-string mechanism at decode time (see `0020`).

## Finding 3: control test on a less-famous title — and a confound

Ran `stage1` + `stage2` on `media/bcs_s01e01_segment.mp4` → `output/bcs_control_after` (27 shots, $0.0071).

Naming in the fresh run is both frequent and **internally inconsistent**:

| Visual subject | Names assigned |
|---|---|
| Man in a brown suit | "Saul Goodman" (shots 6, 8, 9, 22–25) **and** "Jimmy McGill" (shots 12, 14–16, 18, 20) |
| Man in pinstripe suit, purple tie | "Howard Hamlin" (shots 11, 13) **and** "Jimmy McGill" (shot 15) |

17 of 27 shots carry a proper name, against 4 of 27 in the older encode of the same segment.

Jimmy McGill and Saul Goodman are the same person under two canonical names. The describer knows both and cannot hold still on one — a direct illustration of the codec leaking training data rather than reading frames.

**The comparison is confounded.** The two runs used different models:

- older `bcs_s01e01` encode — `gemini-3.1-flash-lite-preview` (`EVAL_MODEL`)
- new `bcs_control_after` — `gemini-2.5-flash-lite` (`ENCODE_MODEL`, the pinned production model)

So the jump from 4 to 17 named shots cannot be attributed to the context feature; model and code both changed. What the run does establish is that **the currently pinned production model produces unstable character naming**, on a title far less represented in training data than Star Wars. Contamination is therefore not a Star Wars-only phenomenon.

A counterweight against over-reading: the `sw_r2_leia` baseline that labelled the hologram *correctly* also ran on `gemini-2.5-flash-lite`. 2.5 is not uniformly worse. A clean experiment is still owed.

## Finding 4: the older, less-famous material stayed visually grounded

The `bcs_gene` encode (14 shots, black-and-white Omaha/Cinnabon sequence) contains **zero** proper names. The same actor, in a series the model demonstrably recognises, described purely as "a man with a mustache wearing a baseball cap and an apron".

The distinguishing factor is not the title's fame but the **shot's** fame. Endlessly-reproduced frames trigger recall; ordinary footage from the same production does not.

## Implications for the codec claim

For an ordinary shot, the pipeline behaves as advertised: the text is the channel. For an iconic shot, some of the reconstruction arrives through the decoder's memory instead of the compressed representation, and there is no clean way to separate the two contributions from the output alone.

This does not invalidate the project, but it does mean:

- Star Wars flatters and distorts results in ways that are hard to bound. It is the most contaminated plausible choice.
- Reported compression results should be accompanied by a less-famous control title.
- The failure is interesting in its own right: with a famous enough film the system stops being purely lossy and starts being partly generative-from-memory.

## Model-choice consequence for the planned full run

`config.py:48-49` pins `ENCODE_MODEL = "gemini-2.5-flash-lite"` and `EVAL_MODEL = "gemini-3.1-flash-lite-preview"`; both were introduced in the config-consolidation commit `891992d`. The existing `star_wars_iv_v2` full-film encode was produced with **3.1-preview**.

As things stand, a full run today would decode a film encoded with one model using a pipeline pinned to another. This should be settled — and the film re-encoded consistently — before spending the ~$30 of pod time.

## Open questions

1. **Deconfounded A/B** — hold the model fixed, toggle the previous-shot context feature, measure naming rate and naming conflicts. Encode-only, pennies.
2. **Which model for the full run** — 2.5 vs 3.1-preview on naming stability and description quality.
3. **Does a name in the prompt change the rendered face?** The Obi-Wan/Vader case suggests yes. Testable cheaply by rendering one shot with the name present and absent, holding the rest of the description constant.
4. **Registry as authority** — would gating propagated names against `characters.json` remove the Saul/Jimmy flip, or does the registry inherit the same instability upstream?
