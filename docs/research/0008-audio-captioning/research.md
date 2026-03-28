---
id: "0008"
type: research
purpose: Evaluate whether CLAPCap or Whisper Audio Captioning can replace the YAMNet + Gemini pipeline for generating sound descriptions
scope: audio encoding
tags: [audio, captioning, clapcap, whisper, yamnet, encoder, eval]
---

# Audio Captioning Model Evaluation

## Problem

The current encode stage2 pipeline generates sound descriptions in two steps:

1. **YAMNet** classifies each shot's audio into coarse AudioSet labels (521 classes, 6 buckets)
2. **Gemini** receives labels + visual keyframes and writes a natural-language `sound` field

This has three known issues:

- **Hallucination**: Gemini invents sounds from visual context. A shot showing an explosion may get "massive detonation" even if the audio is quiet dialogue.
- **Speech leakage**: Gemini descriptions include character voices and dialogue references, causing poor MMAudio output. The speech filter (0007) is a workaround, not a fix.
- **Heavy dependency**: TensorFlow Hub is pulled in solely for YAMNet (~500MB install).

Local audio captioning models generate natural-language descriptions directly from audio waveforms, potentially solving all three problems.

## Approach

Evaluated two models on 19 representative shots from the star_wars_iv_v2 encode, spanning all audio buckets (music, effects, speech, other, silence) with durations from 0.8s to 11.5s:

1. **CLAPCap** (`msclap` PyPI package, MIT license, ~200M params) -- primary candidate
2. **Whisper Audio Captioning** (`MU-NLPC/whisper-small-audio-captioning`, CC BY-NC, 244M params) -- secondary candidate

CoNeTTE was excluded (requires Python <3.11).

The eval script (`tools/eval_audio_captioning.py`) extracts audio clips via ffmpeg, runs each model, and produces a side-by-side comparison against YAMNet labels and Gemini sound descriptions.

## Results

### Summary Metrics

| Metric | CLAPCap | Whisper AC | Gemini (baseline) |
|--------|---------|------------|-------------------|
| Avg inference time | 2.47s/clip | 1.16s/clip | ~2s/shot (API) |
| Speech leak rate | 7/19 (37%) | 5/19 (26%) | ~45% (from 0007 data) |
| Avg caption length | 53 chars | 46 chars | ~90 chars |
| Descriptive captions (>20 chars) | 18/19 (95%) | 17/19 (89%) | 19/19 (100%) |
| Projected full-film time (1750 shots) | ~72 min | ~34 min | ~58 min |
| Degenerate outputs | 1/19 (5%) | 0/19 (0%) | 0/19 (0%) |
| Output format | Natural language | AudioSet-style keyword lists | Natural language |

### Audio Grounding Assessment

CLAPCap captions are **poorly grounded** in actual audio content. The model produces generic, plausible-sounding descriptions that frequently don't match what's actually in the clip:

- **Shot 188** (silence/faint strings): CLAPCap says "A motor vehicle engine revs" -- completely wrong
- **Shot 63** (blaster fire/explosions): CLAPCap says "A machine is running in the background" -- misses the dominant sounds
- **Shot 373** (dishes clinking, small room): CLAPCap says "Wind is blowing in the distance as a car passes by" -- wrong setting entirely
- **Shot 1098** (tense orchestral): Degenerate output -- "A fast-paced, fast-paced, fast-paced..." repeated, 12.8s inference time

The model appears to be guessing common AudioCaps/Clotho patterns rather than genuinely describing the audio.

### Speech Leakage

CLAPCap leaked speech references in 37% of captions, including shots where speech was not the primary audio event:

| Shot | Bucket | Has Dialogue | CLAPCap Caption |
|------|--------|-------------|-----------------|
| 210 | speech | No | "A dog barking and a man talking in the background" |
| 432 | other | Yes | "A man is speaking while music plays in the background" |
| 753 | music | Yes | "A man is speaking over a radio" |
| 813 | effects | Yes | "A man is speaking" |
| 1138 | effects | No | "A man is yelling while a stream of water is flowing" |
| 1178 | other | Yes | "A group of people are talking and splashing" |
| 1585 | speech | Yes | "A man is speaking over a loudspeaker" |

This is **not an improvement** over the Gemini pipeline. CLAPCap would still require the speech filter, defeating one of the primary motivations for switching.

### Descriptiveness for MMAudio

CLAPCap captions are significantly less descriptive than Gemini's:

**Shot 5** (opening crawl, full orchestral score):
- Gemini: "orchestral film score, grand and triumphant"
- CLAPCap: "Loud music is playing"

**Shot 1411** (lightsaber duel):
- Gemini: "The sound of blaster fire, the hum of a lightsaber, metallic clanging, and the hissing of steam or smoke"
- CLAPCap: "A music is playing and music is being played"

**Shot 1039** (laser blast, mechanical sounds):
- Gemini: "A sharp, high-pitched laser blast sound followed by a loud, metallic explosion and the sound of debris and sparks scattering"
- CLAPCap: "A video game is playing and music is playing"

CLAPCap consistently collapses rich audio into generic categories ("music is playing", "a machine is running"). These captions would produce worse MMAudio output than the current Gemini descriptions, which at least mention specific sound types.

### Throughput

- Per-clip: 1.3-3.7s on M3 Pro CPU (excluding the 12.8s degenerate output)
- Projected for 1750 shots: ~72 minutes (local CPU, no API cost)
- Gemini comparison: ~58 minutes (API, ~$0.50 cost for the full film)

Throughput is acceptable but not a meaningful advantage given the quality gap.

### Whisper Audio Captioning

Whisper Audio Captioning produced **AudioSet-style keyword lists** rather than natural language captions, despite being described as a captioning model. Outputs are prefixed with "audioset" fragments and contain comma-separated labels:

- **Shot 5** (orchestral score): "audiosetells, music, music mood, tender music"
- **Shot 63** (blaster fire): "audiosetrain, vehicle horn, rail transport, train, sounds of things"
- **Shot 1098** (tense orchestral): "audioset keywords are explosion, boom"
- **Shot 210** (cave acoustics): "audioset" (empty -- no useful output)

This makes Whisper AC functionally equivalent to YAMNet (keyword classification) rather than a replacement for Gemini (natural language descriptions). It cannot be used directly as the `sound` field for MMAudio prompts. Its keyword outputs are also less accurate than YAMNet's -- e.g., labeling the opening orchestral crawl as "tender music" rather than "Orchestra, Theme music".

Additionally, the CC BY-NC license makes it unsuitable for production use.

## Comparison: Key Examples

### CLAPCap wins (none convincing)

No shot in the eval produced a CLAPCap caption that was clearly better than Gemini's. The closest:

- **Shot 1178** (trash compactor): CLAPCap mentions "splashing around in the water" which is somewhat grounded in the actual audio (liquid sounds), while Gemini's "liquid splashing, sloshing" is equally good but more specific. However, CLAPCap also leaks speech ("people are talking").

### Gemini wins (most shots)

- **Shot 5**: Gemini correctly identifies "orchestral film score, grand and triumphant" vs CLAPCap's "Loud music is playing"
- **Shot 63**: Gemini identifies "blaster fire, laser blasts" vs CLAPCap's "A machine is running"
- **Shot 813**: Gemini describes "mechanical hum, electronic beeps, heavy breathing" vs CLAPCap's "A man is speaking"

### Edge cases

- **Shot 1098**: CLAPCap degenerate output (repetition loop), 12.8s inference. This would need detection and fallback logic in production.
- **Shot 188**: Both wrong -- Gemini says "silence" (closer), CLAPCap says "motor vehicle engine revs" (completely wrong for faint strings).

## Recommendation

**Do not adopt either model.** Keep the current YAMNet + Gemini pipeline.

CLAPCap fails on all four evaluation criteria:

1. **Audio grounding**: Poor. Captions are generic guesses, not grounded descriptions of actual audio content.
2. **Speech-freedom**: Worse than expected. 37% speech leak rate means the speech filter would still be needed.
3. **MMAudio descriptiveness**: Significantly worse than Gemini. Captions collapse to generic categories.
4. **Throughput**: Acceptable but irrelevant given quality problems.

Whisper Audio Captioning is even less viable -- it produces keyword lists rather than natural language captions, making it functionally a worse YAMNet rather than a Gemini replacement. Its CC BY-NC license is also restrictive.

Both models appear limited by their training data (AudioCaps/Clotho) which consists of short, simple audio events -- not the complex film soundscapes in this use case. Gemini's ability to combine visual context with audio labels produces richer, more useful descriptions despite the hallucination and speech leakage problems.

### What would change this assessment

- A larger audio captioning model (e.g., future CLAP variants, Pengi, or audio-specific LLMs) that handles complex polyphonic audio
- A model explicitly trained on film/media audio rather than AudioCaps/Clotho
- Fine-tuning CLAPCap on film audio data (significant effort, unlikely to be worthwhile)

### Integration notes (not recommended, for reference)

If a better model emerges, it would slot into `encode.py` by:
1. Replacing the YAMNet classification step (`encode.py:355-482`) with direct captioning
2. Modifying the Gemini prompt in `generate_descriptions()` (`encode.py:538-744`) to use the caption as context instead of (or alongside) YAMNet labels
3. Alternatively, bypassing Gemini entirely and using the caption directly as the `sound` field -- but only if the model's descriptive quality matches Gemini's

No follow-up integration task is recommended at this time.
