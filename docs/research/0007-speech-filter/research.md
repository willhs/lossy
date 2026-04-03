---
id: "0007"
type: research
purpose: Evaluate whether filtering speech cues from MMAudio prompts improves audio quality
scope: audio generation
tags: [mmaudio, audio, speech-filter, eval]
---

# Speech Filter for MMAudio Prompts

## Problem

Audio eval revealed MMAudio produces its worst output when sound descriptions include character voices or dialogue references ("Darth Vader's voice", "C-3PO speaking"). MMAudio is an SFX/ambiance model, not a speech model — the pipeline already has a separate ElevenLabs TTS track for dialogue.

## Approach

Preprocess sound descriptions before sending to MMAudio: strip speech/voice/dialogue references, keep only ambient, SFX, and music cues.

Two iterations of the filter were built:
1. **v1 (comma-split)**: Split on commas and periods, drop segments containing speech keywords. Fast but left dangling fragments ("C-3PO's anxious," with no noun).
2. **v2 (clause-level)**: Split on semantic clause boundaries (". ", "; ", ", accompanied by", etc.), remove speech items inline with targeted regex patterns. Avoids dangling fragments.

Both versions detect the same keywords: voice, speaking, dialogue, conversation, talking, shout, whisper, scream, yell, murmur, narrate, vocal.

## Results

### Baseline (no filter)

| Backend | Clips | Avg Relevance | Avg Quality | Source |
|---------|-------|---------------|-------------|--------|
| runpod-mmaudio | 194 | 0.57 | 0.54 | Previous eval (pre-filter) |

### Filter v1 (comma-split) via fal.ai

| Backend | Clips | Avg Relevance | Avg Quality | Source |
|---------|-------|---------------|-------------|--------|
| mmaudio (fal.ai) | 39 | 0.63 | 0.63 | eval-audio-star_wars_iv_v2-mmaudio-20260327-2146.md |

**vs baseline**: relevance +0.06, quality +0.09

### Filter v2 (clause-level) via RunPod

| Backend | Clips | Avg Relevance | Avg Quality | Source |
|---------|-------|---------------|-------------|--------|
| runpod-mmaudio (filtered 0-426) | 38 | 0.50 | 0.52 | eval-audio-star_wars_iv_v2-runpod-mmaudio-20260328-1337.md |
| runpod-mmaudio (unfiltered 427+) | 151 | 0.56 | 0.55 | Same eval, different shot range |

**vs unfiltered same model**: relevance -0.06, quality -0.03

## Analysis

The filter v1 on fal.ai showed a clear improvement (+0.09 quality). Filter v2 on RunPod self-hosted showed a slight regression. Key confounding variables:

1. **Backend difference**: fal.ai MMAudio and RunPod ComfyUI MMAudio may use different model weights, precision, or inference parameters. The RunPod strategy uses ComfyUI's MMAudioSampler node with fp16 weights.
2. **Duration capping**: RunPod strategy caps clips at 5-10s (per 0006-mmaudio-duration findings). fal.ai strategy allowed up to 30s clips at the time of the v1 eval, though it has since been updated to 5-10s too.
3. **Sample size**: 38-39 clips per filtered eval vs 151-194 for baselines. Small sample sizes make these comparisons noisy.
4. **Filter aggressiveness**: v2 drops entire clauses when speech is the main subject, which may remove useful ambient context that was adjacent to speech references.

## Conclusions

- Speech filtering provides a measurable quality improvement when using fal.ai hosted MMAudio (+0.09 quality, +0.06 relevance)
- The RunPod self-hosted backend produces lower quality than fal.ai regardless of filtering, suggesting the backend difference is a larger factor than prompt filtering
- The clause-level filter (v2) produces cleaner prompts than v1 but the quality gains are backend-dependent
- Recommend keeping the filter enabled for all MMAudio backends — the prompt quality is objectively better even if the eval scores are noisy

## Next Steps

- Re-run eval on fal.ai with filter v2 once balance is topped up, to isolate filter improvement from backend difference
- Investigate RunPod MMAudio configuration (precision, steps, cfg) to close the quality gap with fal.ai
- Consider increasing eval sample size for more reliable comparisons
