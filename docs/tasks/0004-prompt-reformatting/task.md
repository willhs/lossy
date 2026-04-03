---
id: task-0004
type: spec
purpose: "Improve video generation quality by reformatting structured Gemini descriptions into model-efficient prompts with cinematography vocabulary, front-loaded content, and negative prompts."
tags: ["decoder", "prompts", "video-generation", "quality"]
related: ["docs/research/0003-prompt-to-video/research.md", "docs/research/0002-shot-to-prompt/research.md", "docs/design/adr/004-replicate-wan22-for-video-generation.md", "docs/design/adr/005-runpod-self-hosted-wan-strategy.md", "docs/research/experiments/0001-first-e2e-decode-test.md"]
created: 2026-03-17
updated: 2026-03-17
---

# Prompt Reformatting for Video Gen Models

## Goal

Replace the current `format_prompt()` function with model-aware formatters that distill the same structured Gemini description data into prompts optimized for each video generation backend. A/B test against the current format on ~15 shots to measure the impact.

## Context

The encoder (Gemini Flash-Lite) produces structured JSON descriptions per shot: shot_type, camera_movement, subjects, action, lighting, color_palette, mood, setting. The decoder's `format_prompt()` (`decode.py:718-769`) flattens these into a single text string for video generation.

Current prompts are 150-200 words of human-readable prose. They work, but use inefficient vocabulary for video models:

- **Narrative filler** — phrases like "The camera slowly zooms out, revealing more of the vastness of space" where "Slow dolly out revealing Star Destroyer scale against planet" conveys the same visual information in fewer tokens.
- **Metadata labels** — "Color palette:", "Mood:" waste tokens on semantic labels the model wasn't trained on.
- **Casual camera terms** — "slow zoom out" instead of professional terms like "dolly out", "tracking shot", "crane up" that the model's training data (professional film descriptions) used.
- **No negative prompts** — Wan benefits significantly from a default negative prompt (Chinese-language, matching its training data). Currently empty string in both Replicate and RunPod strategies.

### Technical constraints discovered during research

- Wan's T5 (UmT5-XXL) encoder truncates at **512 tokens**; quality degrades around **320-350 tokens** (~200-250 words) due to cross-attention training distribution.
- Seedance prefers shorter prompts (~30-60 words) with single action verbs and intensity adverbs. Does not support negative prompts.
- No production video model supports structured multi-aspect inputs (separate foreground/background/style fields). Single text field only.
- ComfyUI's Conditioning Combine node could split a prompt into two encoding paths — unproven with Wan's DiT architecture but architecturally plausible.
- Prompt weighting syntax (`(word:1.5)`) is unreliable with T5-based encoders.

### What "reformatting" means

Not shortening — distilling. The same structured fields (shot_type, camera_movement, subjects, action, lighting, color_palette, mood, setting) expressed in model-efficient vocabulary, ordered by importance (subject/action first, atmosphere last), without filler or labels.

## Scope

### Must Do
- New Wan-optimized formatter: `Subject → Action → Camera → Style/Atmosphere` using professional cinematography vocabulary
- New Seedance-optimized formatter: shorter, single action verb, intensity adverbs
- Add default Chinese negative prompt to ReplicateWanStrategy and RunPodWanStrategy
- Make prompt formatting strategy-aware (each strategy selects its own formatter)
- A/B decode test: ~15 shots (indices 10-24) comparing current vs new Wan formatter
- Experiment doc in `docs/research/experiments/` documenting results

### Might Do
- Test ComfyUI Conditioning Combine for RunPod strategy — split subject+action / environment+style into two CLIPTextEncode nodes
- Compare multiple negative prompt variants (Chinese default vs English vs combined)

### Won't Do
- Changing the encoder (Gemini system prompt stays as-is — structured descriptions are valuable)
- LLM-based prompt rewriting at decode time (adds cost and latency)
- Image-to-video conditioning (separate task)
- Prompt weighting or regional prompting (unreliable with T5)

## Constraints

- Zero additional API cost for the reformatting itself (it's just string manipulation)
- A/B test will cost ~$0.75 (15 clips at $0.05 each on Replicate)
- Must work with existing `prompts.json` files — no re-encoding required
- Must not break existing strategies or the stitch pipeline

## References

- [Prompt-to-video research](../../research/0003-prompt-to-video/research.md) — open question #3 is this task
- [Shot-to-prompt research](../../research/0002-shot-to-prompt/research.md) — how Gemini descriptions are generated
- [ADR-004: Replicate Wan 2.2](../../design/adr/004-replicate-wan22-for-video-generation.md)
- [ADR-005: RunPod self-hosted Wan](../../design/adr/005-runpod-self-hosted-wan-strategy.md)
- [Experiment 0001: First E2E decode test](../../research/experiments/0001-first-e2e-decode-test.md) — baseline results to compare against

## Success Criteria

- [ ] New Wan formatter produces prompts using cinematography vocabulary, front-loaded subject/action, no metadata labels
- [ ] New Seedance formatter produces ~60-word prompts with single action verbs
- [ ] Wan strategies include default negative prompt
- [ ] Each strategy uses its own formatter via the strategy interface
- [ ] A/B test completed on ~15 shots with visual comparison documented
- [ ] Experiment doc written with side-by-side observations and recommendation
